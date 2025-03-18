#
# Author: Ye Li, <pkupkupku@pku.edu.com> (this email address is valid till 2027 June)
#

from functools import reduce
import numpy
from pyscf import lib
# from pyscf import scf
from pyscf import symm
# from pyscf import ao2mo
from pyscf.lib import logger
# from pyscf.tdscf import rhf
from pyscf.tdscf import uhf
from pyscf.tdscf import rhf
from pyscf.scf import hf_symm
from pyscf.scf import uhf_symm
# from pyscf.scf import _response_functions
from pyscf.data import nist
from pyscf import __config__
import pyscf_losc

# import py_losc
# from pyscf_losc import utils
# from pyscf_losc import losc_options

OUTPUT_THRESHOLD = getattr(__config__, 'tdscf_rhf_get_nto_threshold', 0.3)
REAL_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_pick_eig_threshold', 1e-4)
MO_BASE = getattr(__config__, 'MO_BASE', 1)


def _get_x_sym_table(mf):
    '''Irrep (up to D2h symmetry) of each coefficient in X[nocc,nvir]'''
    mol = mf.mol
    mo_occ = mf.mo_occ
    orbsym = hf_symm.get_orbsym(mol, mf.mo_coeff)
    orbsym = orbsym % 10  # convert to D2h irreps
    return orbsym[mo_occ==2,None] ^ orbsym[mo_occ==0]



# region TDA-GSC-Restricted

def gen_tda_operation_postGSC_R(mf, losc_data, fock_ao=None, singlet=True, wfnsym=None):
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    # assert (mo_coeff.dtype == numpy.double)
    mo_energy = losc_data['Orblosc'][0]
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    occidx = numpy.where(mo_occ==2)[0]
    viridx = numpy.where(mo_occ==0)[0]
    nocc = len(occidx)
    nvir = len(viridx)
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]


    curvature       =   losc_data['curvature']
    select_CO_idx   =   losc_data['select_CO_idx']
    select_CO     =   select_CO_idx[0]
    kappa         =   numpy.zeros((nao,nao), dtype=float)
    if select_CO is not None:
        kappa[select_CO[0]:select_CO[1], select_CO[0]:select_CO[1]] = 1 * curvature[0]
    delta_e_ia    =   kappa[:nocc, nocc:] # in shape(nocc,nvir)


    if wfnsym is not None and mol.symmetry:
        if isinstance(wfnsym, str):
            wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
        wfnsym = wfnsym % 10  # convert to D2h subgroup
        x_sym = _get_x_sym_table(mf)
        sym_forbid = x_sym != wfnsym

    if fock_ao is None:
        e_ia = hdiag = mo_energy[viridx] - mo_energy[occidx,None]
        e_ia -= delta_e_ia
    else:
        fock = reduce(numpy.dot, (mo_coeff.conj().T, fock_ao, mo_coeff))
        foo = fock[occidx[:,None],occidx]
        fvv = fock[viridx[:,None],viridx]
        hdiag = fvv.diagonal() - foo.diagonal()[:,None]

    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = hdiag.ravel()

    mo_coeff = numpy.asarray(numpy.hstack((orbo,orbv)), order='F')
    vresp = mf.gen_response(singlet=singlet, hermi=0)

    def vind(zs):
        zs = numpy.asarray(zs).reshape(-1,nocc,nvir)
        if wfnsym is not None and mol.symmetry:
            zs = numpy.copy(zs)
            zs[:,sym_forbid] = 0

        # *2 for double occupancy
        dmov = lib.einsum('xov,qv,po->xpq', zs*2, orbv.conj(), orbo)
        v1ao = vresp(dmov)
        v1ov = lib.einsum('xpq,po,qv->xov', v1ao, orbo.conj(), orbv)
        if fock_ao is None:
            v1ov += numpy.einsum('xia,ia->xia', zs, e_ia)
        else:
            v1ov += lib.einsum('xqs,sp->xqp', zs, fvv)
            v1ov -= lib.einsum('xpr,sp->xsr', zs, foo)
        if wfnsym is not None and mol.symmetry:
            v1ov[:,sym_forbid] = 0
        return v1ov.reshape(v1ov.shape[0],-1)

    return vind, hdiag


class TDA_postGSC_R(rhf.TDA):

    def __init__(self, mf, losc_data):
        super().__init__(mf)
        self.losc_data = losc_data
        pass

    def gen_vind(self, mf=None, losc_data=None):
        if mf is None: 
            mf = self._scf
        if losc_data is None:
            losc_data = self.losc_data
        return gen_tda_operation_postGSC_R(mf, losc_data, singlet=self.singlet, wfnsym=self.wfnsym)


# region TDA-GSC
def gen_tda_operation_postGSC(mf, losc_data, fock_ao=None, wfnsym=None):
    '''A x

    Kwargs:
        wfnsym : int or str
            Point group symmetry irrep symbol or ID for excited CIS wavefunction.
    '''
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    assert (mo_coeff[0].dtype == numpy.double)
    mo_energy = losc_data['Orblosc']
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff[0].shape
    occidxa = numpy.where(mo_occ[0]>0)[0]
    occidxb = numpy.where(mo_occ[1]>0)[0]
    viridxa = numpy.where(mo_occ[0]==0)[0]
    viridxb = numpy.where(mo_occ[1]==0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:,occidxa]
    orbob = mo_coeff[1][:,occidxb]
    orbva = mo_coeff[0][:,viridxa]
    orbvb = mo_coeff[1][:,viridxb]

    curvature       =   losc_data['curvature']
    select_CO_idx   =   losc_data['select_CO_idx']
    select_CO_a     =   select_CO_idx[0]
    select_CO_b     =   select_CO_idx[1]
    kappa_a         =   numpy.zeros((nao,nao), dtype=float)
    kappa_b         =   numpy.zeros((nao,nao), dtype=float)
    if select_CO_a is not None:
        kappa_a[select_CO_a[0]:select_CO_a[1], select_CO_a[0]:select_CO_a[1]] = 1 * curvature[0]
    if select_CO_b is not None:
        kappa_b[select_CO_b[0]:select_CO_b[1], select_CO_b[0]:select_CO_b[1]] = 1 * curvature[1]
    delta_e_ia_a    =   kappa_a[:nocca, nocca:] # in shape(nocca,nvira)
    delta_e_ia_b    =   kappa_b[:noccb, noccb:] # in shape(noccb,nvirb)

    if wfnsym is not None and mol.symmetry:
        if isinstance(wfnsym, str):
            wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
        orbsyma, orbsymb = uhf_symm.get_orbsym(mol, mo_coeff)
        wfnsym = wfnsym % 10  # convert to D2h subgroup
        orbsyma_in_d2h = numpy.asarray(orbsyma) % 10
        orbsymb_in_d2h = numpy.asarray(orbsymb) % 10
        sym_forbida = (orbsyma_in_d2h[occidxa,None] ^ orbsyma_in_d2h[viridxa]) != wfnsym
        sym_forbidb = (orbsymb_in_d2h[occidxb,None] ^ orbsymb_in_d2h[viridxb]) != wfnsym
        sym_forbid = numpy.hstack((sym_forbida.ravel(), sym_forbidb.ravel()))

    e_ia_a = mo_energy[0][viridxa] - mo_energy[0][occidxa,None]
    e_ia_b = mo_energy[1][viridxb] - mo_energy[1][occidxb,None]
    # adding the curvature correction
    e_ia_a -= delta_e_ia_a
    e_ia_b -= delta_e_ia_b

    e_ia = numpy.hstack((e_ia_a.reshape(-1), e_ia_b.reshape(-1)))
    hdiag = e_ia
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.8-mem_now)
    vresp = mf.gen_response(hermi=0, max_memory=max_memory)

    def vind(zs):
        nz = len(zs)
        zs = numpy.asarray(zs)
        if wfnsym is not None and mol.symmetry:
            zs = numpy.copy(zs)
            zs[:,sym_forbid] = 0

        za = zs[:,:nocca*nvira].reshape(nz,nocca,nvira)
        zb = zs[:,nocca*nvira:].reshape(nz,noccb,nvirb)
        dmova = lib.einsum('xov,qv,po->xpq', za, orbva.conj(), orboa)
        dmovb = lib.einsum('xov,qv,po->xpq', zb, orbvb.conj(), orbob)

        v1ao = vresp(numpy.asarray((dmova,dmovb)))

        v1a = lib.einsum('xpq,po,qv->xov', v1ao[0], orboa.conj(), orbva)
        v1b = lib.einsum('xpq,po,qv->xov', v1ao[1], orbob.conj(), orbvb)
        v1a += numpy.einsum('xia,ia->xia', za, e_ia_a)
        v1b += numpy.einsum('xia,ia->xia', zb, e_ia_b)

        hx = numpy.hstack((v1a.reshape(nz,-1), v1b.reshape(nz,-1)))
        if wfnsym is not None and mol.symmetry:
            hx[:,sym_forbid] = 0
        return hx

    return vind, hdiag


class TDA_postGSC(uhf.TDA):

    def __init__(self, mf, losc_data):
        super().__init__(mf)
        self.losc_data = losc_data
        pass

    def gen_vind(self, mf=None, losc_data=None):
        if mf is None: 
            mf = self._scf
        if losc_data is None:
            losc_data = self.losc_data
        return gen_tda_operation_postGSC(mf, losc_data, wfnsym=self.wfnsym)


# region TDA-LOSC-Restricted

def gen_tda_operation_postLOSC_R(mf, losc_data, fock_ao=None, singlet=True, wfnsym=None):
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    # assert (mo_coeff.dtype == numpy.double)
    mo_energy = losc_data['Orblosc'][0]
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    occidx = numpy.where(mo_occ==2)[0]
    viridx = numpy.where(mo_occ==0)[0]
    nocc = len(occidx)
    nvir = len(viridx)
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]


    curvature       =   losc_data['curvature']
    select_CO_idx   =   losc_data['select_CO_idx']
    U               =   losc_data['U']
    select_CO     =   select_CO_idx[0]
    kappa         =   numpy.zeros((nao,nao), dtype=float)
    Umat             =   numpy.eye(nao, dtype=float) # in shape(nao,nao)
    if select_CO is not None:
        kappa[select_CO[0]:select_CO[1], select_CO[0]:select_CO[1]] = 1 * curvature[0]
        Umat[select_CO[0]:select_CO[1], select_CO[0]:select_CO[1]] = 1 * U[0]
    delta_e_ia    =   kappa[:nocc, nocc:] # in shape(nocc,nvir)
    Upo        =   Umat[:,:nocc] # in shape(nao,nocca)
    Upv        =   Umat[:,nocc:] # in shape(nao,nvira)


    if wfnsym is not None and mol.symmetry:
        if isinstance(wfnsym, str):
            wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
        wfnsym = wfnsym % 10  # convert to D2h subgroup
        x_sym = _get_x_sym_table(mf)
        sym_forbid = x_sym != wfnsym

    if fock_ao is None:
        e_ia = mo_energy[viridx] - mo_energy[occidx,None]
        hdiag = e_ia - delta_e_ia
    else:
        fock = reduce(numpy.dot, (mo_coeff.conj().T, fock_ao, mo_coeff))
        foo = fock[occidx[:,None],occidx]
        fvv = fock[viridx[:,None],viridx]
        hdiag = fvv.diagonal() - foo.diagonal()[:,None]

    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = hdiag.ravel()

    mo_coeff = numpy.asarray(numpy.hstack((orbo,orbv)), order='F')
    vresp = mf.gen_response(singlet=singlet, hermi=0)

    def vind(zs):
        zs = numpy.asarray(zs).reshape(-1,nocc,nvir)
        if wfnsym is not None and mol.symmetry:
            zs = numpy.copy(zs)
            zs[:,sym_forbid] = 0

        # *2 for double occupancy
        dmov = lib.einsum('xov,qv,po->xpq', zs*2, orbv.conj(), orbo)
        v1ao = vresp(dmov)
        v1ov = lib.einsum('xpq,po,qv->xov', v1ao, orbo.conj(), orbv)
        if fock_ao is None:
            v1ov += numpy.einsum('xia,ia->xia', zs, e_ia)
        else:
            v1ov += lib.einsum('xqs,sp->xqp', zs, fvv)
            v1ov -= lib.einsum('xpr,sp->xsr', zs, foo)

        # adding the LOSC xc kernel correction
        # TODO: extending to complex value of U
        # TODO: check if we need to times 2 for the double occupancy
        W_a = lib.einsum('pq,pa,qi,xia->xpq', kappa, Upv, Upo, zs) # in shape(nz,nao,nao)
        delta_AX = - lib.einsum('pi,xqp,qa->xia', Upo, W_a, Upv) # in shape(nz,nocc,nvir)
        v1ov += delta_AX

        if wfnsym is not None and mol.symmetry:
            v1ov[:,sym_forbid] = 0
        return v1ov.reshape(v1ov.shape[0],-1)
    return vind, hdiag

class TDA_postLOSC_R(rhf.TDA):

    def __init__(self, mf, losc_data):
        super().__init__(mf)
        self.losc_data = losc_data
        pass

    def gen_vind(self, mf=None, losc_data=None):
        if mf is None: 
            mf = self._scf
        if losc_data is None:
            losc_data = self.losc_data
        return gen_tda_operation_postLOSC_R(mf, losc_data, singlet=self.singlet, wfnsym=self.wfnsym)




# region TDA-LOSC 

def gen_tda_operation_postLOSC(loscmf, losc_data, fock_ao=None, wfnsym=None):
    mol = loscmf.mol
    mo_coeff = loscmf.mo_coeff
    assert (mo_coeff[0].dtype == numpy.double)
    mo_energy = losc_data['Orblosc']
    mo_occ = loscmf.mo_occ
    nao, nmo = mo_coeff[0].shape
    occidxa = numpy.where(mo_occ[0]>0)[0]
    occidxb = numpy.where(mo_occ[1]>0)[0]
    viridxa = numpy.where(mo_occ[0]==0)[0]
    viridxb = numpy.where(mo_occ[1]==0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:,occidxa]
    orbob = mo_coeff[1][:,occidxb]
    orbva = mo_coeff[0][:,viridxa]
    orbvb = mo_coeff[1][:,viridxb]

    curvature       =   losc_data['curvature']
    select_CO_idx   =   losc_data['select_CO_idx']
    U               =   losc_data['U']
    select_CO_a     =   select_CO_idx[0]
    select_CO_b     =   select_CO_idx[1]
    kappa_a         =   numpy.zeros((nao,nao), dtype=float) # in shape(nao,nao)
    kappa_b         =   numpy.zeros((nao,nao), dtype=float) # in shape(nao,nao)
    U_a             =   numpy.eye(nao, dtype=float) # in shape(nao,nao)
    U_b             =   numpy.eye(nao, dtype=float) # in shape(nao,nao)
    if select_CO_a is not None:
        kappa_a[select_CO_a[0]:select_CO_a[1], select_CO_a[0]:select_CO_a[1]] = 1 * curvature[0]
        U_a[select_CO_a[0]:select_CO_a[1], select_CO_a[0]:select_CO_a[1]] = 1 * U[0]
    if select_CO_b is not None:
        kappa_b[select_CO_b[0]:select_CO_b[1], select_CO_b[0]:select_CO_b[1]] = 1 * curvature[1]
        U_b[select_CO_b[0]:select_CO_b[1], select_CO_b[0]:select_CO_b[1]] = 1 * U[1]
    delta_e_ia_a    =   kappa_a[:nocca, nocca:] # in shape(nocca,nvira)
    delta_e_ia_b    =   kappa_b[:noccb, noccb:] # in shape(noccb,nvirb)
    Upo_a        =   U_a[:,:nocca] # in shape(nao,nocca)
    Upv_a        =   U_a[:,nocca:] # in shape(nao,nvira)
    Upo_b        =   U_b[:,:noccb] # in shape(nao,noccb)
    Upv_b        =   U_b[:,noccb:] # in shape(nao,nvirb)

    if wfnsym is not None and mol.symmetry:
        if isinstance(wfnsym, str):
            wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
        orbsyma, orbsymb = uhf_symm.get_orbsym(mol, mo_coeff)
        wfnsym = wfnsym % 10  # convert to D2h subgroup
        orbsyma_in_d2h = numpy.asarray(orbsyma) % 10
        orbsymb_in_d2h = numpy.asarray(orbsymb) % 10
        sym_forbida = (orbsyma_in_d2h[occidxa,None] ^ orbsyma_in_d2h[viridxa]) != wfnsym
        sym_forbidb = (orbsymb_in_d2h[occidxb,None] ^ orbsymb_in_d2h[viridxb]) != wfnsym
        sym_forbid = numpy.hstack((sym_forbida.ravel(), sym_forbidb.ravel()))

    e_ia_a = mo_energy[0][viridxa] - mo_energy[0][occidxa,None]
    e_ia_b = mo_energy[1][viridxb] - mo_energy[1][occidxb,None]
    hdiag_a = e_ia_a - delta_e_ia_a
    hdiag_b = e_ia_b - delta_e_ia_b

    # e_ia = numpy.hstack((e_ia_a.reshape(-1), e_ia_b.reshape(-1)))
    hdiag = numpy.hstack((hdiag_a.reshape(-1), hdiag_b.reshape(-1)))
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, loscmf.max_memory*.8-mem_now)
    vresp = loscmf.gen_response(hermi=0, max_memory=max_memory)

    def vind(zs):
        nz = len(zs)
        zs = numpy.asarray(zs)
        if wfnsym is not None and mol.symmetry:
            zs = numpy.copy(zs)
            zs[:,sym_forbid] = 0

        za = zs[:,:nocca*nvira].reshape(nz,nocca,nvira)
        zb = zs[:,nocca*nvira:].reshape(nz,noccb,nvirb)
        dmova = lib.einsum('xov,qv,po->xpq', za, orbva.conj(), orboa)
        dmovb = lib.einsum('xov,qv,po->xpq', zb, orbvb.conj(), orbob)

        v1ao = vresp(numpy.asarray((dmova,dmovb)))

        v1a = lib.einsum('xpq,po,qv->xov', v1ao[0], orboa.conj(), orbva)
        v1b = lib.einsum('xpq,po,qv->xov', v1ao[1], orbob.conj(), orbvb)
        v1a += numpy.einsum('xia,ia->xia', za, e_ia_a)
        v1b += numpy.einsum('xia,ia->xia', zb, e_ia_b)

        # adding the LOSC xc kernel correction
        # TODO: extending to complex value of U
        W_a = lib.einsum('pq,pa,qi,xia->xpq', kappa_a, Upv_a, Upo_a, za) # in shape(nz,nao,nao)
        W_b = lib.einsum('pq,pa,qi,xia->xpq', kappa_b, Upv_b, Upo_b, zb) # in shape(nz,nao,nao)

        delta_AX_a = - lib.einsum('pi,xqp,qa->xia', Upo_a, W_a, Upv_a) # in shape(nz,nocca,nvira)
        delta_AX_b = - lib.einsum('pi,xqp,qa->xia', Upo_b, W_b, Upv_b) # in shape(nz,noccb,nvirb)
        v1a += delta_AX_a
        v1b += delta_AX_b

        hx = numpy.hstack((v1a.reshape(nz,-1), v1b.reshape(nz,-1)))
        if wfnsym is not None and mol.symmetry:
            hx[:,sym_forbid] = 0
        return hx

    return vind, hdiag

class TDA_postLOSC(uhf.TDA):

    def __init__(self, loscmf, losc_data):
        super().__init__(loscmf)
        self.losc_data = losc_data
        pass

    def gen_vind(self, loscmf=None, losc_data=None):
        if loscmf is None: 
            loscmf = self._scf
        if losc_data is None:
            losc_data = self.losc_data
        return gen_tda_operation_postLOSC(loscmf, losc_data, wfnsym=self.wfnsym)

# region TDDFT-GSC-Res
def gen_tdhf_operation_postGSC_R(mf, losc_data, fock_ao=None, singlet=True, wfnsym=None):
    '''Generate function to compute

    [ A   B ][X]
    [-B* -A*][Y]
    '''
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    # assert (mo_coeff.dtype == numpy.double)
    mo_energy = losc_data['Orblosc'][0]
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    occidx = numpy.where(mo_occ==2)[0]
    viridx = numpy.where(mo_occ==0)[0]
    nocc = len(occidx)
    nvir = len(viridx)
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]

    curvature       =   losc_data['curvature']
    select_CO_idx   =   losc_data['select_CO_idx']
    select_CO     =   select_CO_idx[0]
    kappa         =   numpy.zeros((nao,nao), dtype=float)
    if select_CO is not None:
        kappa[select_CO[0]:select_CO[1], select_CO[0]:select_CO[1]] = 1 * curvature[0]
    delta_e_ia    =   kappa[:nocc, nocc:] # in shape(nocc,nvir)

    if wfnsym is not None and mol.symmetry:
        if isinstance(wfnsym, str):
            wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
        wfnsym = wfnsym % 10  # convert to D2h subgroup
        sym_forbid = _get_x_sym_table(mf) != wfnsym

    assert fock_ao is None

    e_ia    =   mo_energy[viridx] - mo_energy[occidx,None]
    hdiag   =   mo_energy[viridx] - mo_energy[occidx,None]
    e_ia    -=  delta_e_ia
    hdiag   -=  delta_e_ia
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = numpy.hstack((hdiag.ravel(), -hdiag.ravel()))

    mo_coeff = numpy.asarray(numpy.hstack((orbo,orbv)), order='F')
    vresp = mf.gen_response(singlet=singlet, hermi=0)

    def vind(xys):
        xys = numpy.asarray(xys).reshape(-1,2,nocc,nvir)
        if wfnsym is not None and mol.symmetry:
            # shape(nz,2,nocc,nvir): 2 ~ X,Y
            xys = numpy.copy(xys)
            xys[:,:,sym_forbid] = 0

        xs, ys = xys.transpose(1,0,2,3)
        # *2 for double occupancy
        dms  = lib.einsum('xov,qv,po->xpq', xs*2, orbv.conj(), orbo)
        dms += lib.einsum('xov,pv,qo->xpq', ys*2, orbv, orbo.conj())
        v1ao = vresp(dms) # = <mb||nj> Xjb + <mj||nb> Yjb
        # A ~= <ib||aj>, B = <ij||ab>
        # AX + BY
        # = <ib||aj> Xjb + <ij||ab> Yjb
        # = (<mb||nj> Xjb + <mj||nb> Yjb) Cmi* Cna
        v1ov = lib.einsum('xpq,po,qv->xov', v1ao, orbo.conj(), orbv)
        # (B*)X + (A*)Y
        # = <ab||ij> Xjb + <aj||ib> Yjb
        # = (<mb||nj> Xjb + <mj||nb> Yjb) Cma* Cni
        v1vo = lib.einsum('xpq,qo,pv->xov', v1ao, orbo, orbv.conj())
        v1ov += numpy.einsum('xia,ia->xia', xs, e_ia)  # AX
        v1vo += numpy.einsum('xia,ia->xia', ys, e_ia.conj())  # (A*)Y

        if wfnsym is not None and mol.symmetry:
            v1ov[:,sym_forbid] = 0
            v1vo[:,sym_forbid] = 0

        # (AX, -AY)
        nz = xys.shape[0]
        hx = numpy.hstack((v1ov.reshape(nz,-1), -v1vo.reshape(nz,-1)))
        return hx

    return vind, hdiag


class TDHF_postGSC_R(rhf.TDHF):

    def __init__(self, loscmf, losc_data):
        super().__init__(loscmf)
        self.losc_data = losc_data
        pass

    def gen_vind(self, loscmf=None, losc_data=None):
        if loscmf is None: 
            loscmf = self._scf
        if losc_data is None:
            losc_data = self.losc_data
        return gen_tdhf_operation_postGSC_R(loscmf, losc_data, singlet=self.singlet, wfnsym=self.wfnsym)
    



# region TDDFT-GSC
def gen_tdhf_operation_postGSC(loscmf, losc_data, fock_ao=None, singlet=True, wfnsym=None):
    '''Generate function to compute

    [ A  B][X]
    [-B -A][Y]

    with GSC correction.
    '''
    mol = loscmf.mol
    mo_coeff = loscmf.mo_coeff
    assert (mo_coeff[0].dtype == numpy.double)
    mo_energy = losc_data['Orblosc']
    mo_occ = loscmf.mo_occ
    nao, nmo = mo_coeff[0].shape
    occidxa = numpy.where(mo_occ[0]>0)[0]
    occidxb = numpy.where(mo_occ[1]>0)[0]
    viridxa = numpy.where(mo_occ[0]==0)[0]
    viridxb = numpy.where(mo_occ[1]==0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:,occidxa] # in shape (nao,nocca)
    orbob = mo_coeff[1][:,occidxb] # in shape (nao,noccb)
    orbva = mo_coeff[0][:,viridxa] # in shape (nao,nvira)
    orbvb = mo_coeff[1][:,viridxb] # in shape (nao,nvirb)

    curvature       =   losc_data['curvature']
    select_CO_idx   =   losc_data['select_CO_idx']
    select_CO_a     =   select_CO_idx[0]
    select_CO_b     =   select_CO_idx[1]
    kappa_a         =   numpy.zeros((nao,nao), dtype=float)
    kappa_b         =   numpy.zeros((nao,nao), dtype=float)
    if select_CO_a is not None:
        kappa_a[select_CO_a[0]:select_CO_a[1], select_CO_a[0]:select_CO_a[1]] = 1 * curvature[0]
    if select_CO_b is not None:
        kappa_b[select_CO_b[0]:select_CO_b[1], select_CO_b[0]:select_CO_b[1]] = 1 * curvature[1]
    delta_e_ia_a    =   kappa_a[:nocca, nocca:] # in shape(nocca,nvira)
    delta_e_ia_b    =   kappa_b[:noccb, noccb:] # in shape(noccb,nvirb)


    if wfnsym is not None and mol.symmetry:
        if isinstance(wfnsym, str):
            wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
        orbsyma, orbsymb = uhf_symm.get_orbsym(mol, mo_coeff)
        wfnsym = wfnsym % 10  # convert to D2h subgroup
        orbsyma_in_d2h = numpy.asarray(orbsyma) % 10
        orbsymb_in_d2h = numpy.asarray(orbsymb) % 10
        sym_forbida = (orbsyma_in_d2h[occidxa,None] ^ orbsyma_in_d2h[viridxa]) != wfnsym
        sym_forbidb = (orbsymb_in_d2h[occidxb,None] ^ orbsymb_in_d2h[viridxb]) != wfnsym
        sym_forbid = numpy.hstack((sym_forbida.ravel(), sym_forbidb.ravel()))

    e_ia_a = mo_energy[0][viridxa] - mo_energy[0][occidxa,None] # in shape(nocc_a,nvir_a)
    e_ia_b = mo_energy[1][viridxb] - mo_energy[1][occidxb,None] # in shape(nocc_b,nvir_b)
    # adding the curvature correction
    e_ia_a -= delta_e_ia_a
    e_ia_b -= delta_e_ia_b
    # in shape(nocca*nvira+noccb*nvirb,)
    e_ia = hdiag = numpy.hstack((e_ia_a.ravel(), e_ia_b.ravel())) # in shape(nocca*nvira+noccb*nvirb,)
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = numpy.hstack((hdiag, -hdiag)) # in shape(2*(nocca*nvira+noccb*nvirb),)


    mem_now = lib.current_memory()[0]
    max_memory = max(2000, loscmf.max_memory*.8-mem_now)
    vresp = loscmf.gen_response(hermi=0, max_memory=max_memory)

    def vind(xys):
        nz = len(xys)
        xys = numpy.asarray(xys).reshape(nz,2,-1)
        if wfnsym is not None and mol.symmetry:
            # shape(nz,2,-1): 2 ~ X,Y
            xys = numpy.copy(xys)
            xys[:,:,sym_forbid] = 0

        xs, ys = xys.transpose(1,0,2)
        # xs ~ [Xa, Xb], in shape(nz,nocca*nvira+noccb*nvirb)
        # ys ~ [Ya, Yb], in shape(nz,nocca*nvira+noccb*nvirb)
        xa = xs[:,:nocca*nvira].reshape(nz,nocca,nvira) # in shape(nz,nocca,nvira)
        xb = xs[:,nocca*nvira:].reshape(nz,noccb,nvirb) # in shape(nz,noccb,nvirb)
        ya = ys[:,:nocca*nvira].reshape(nz,nocca,nvira) # in shape(nz,nocca,nvira)
        yb = ys[:,nocca*nvira:].reshape(nz,noccb,nvirb) # in shape(nz,noccb,nvirb)
        # dms = AX + BY
        dmsa  = lib.einsum('xov,qv,po->xpq', xa, orbva.conj(), orboa) # in shape(nz,nao,nao)
        dmsb  = lib.einsum('xov,qv,po->xpq', xb, orbvb.conj(), orbob) # in shape(nz,nao,nao)
        dmsa += lib.einsum('xov,pv,qo->xpq', ya, orbva, orboa.conj())
        dmsb += lib.einsum('xov,pv,qo->xpq', yb, orbvb, orbob.conj())

        v1ao = vresp(numpy.asarray((dmsa,dmsb))) # in shape(2,nz,nao,nao)

        v1aov = lib.einsum('xpq,po,qv->xov', v1ao[0], orboa.conj(), orbva)
        # v1aov in shape(nz, nocca, nvira)
        v1avo = lib.einsum('xpq,qo,pv->xov', v1ao[0], orboa, orbva.conj())
        # v1avo in shape(nz, nocca, nvira)
        v1bov = lib.einsum('xpq,po,qv->xov', v1ao[1], orbob.conj(), orbvb)
        # v1bov in shape(nz, noccb, nvirb)
        v1bvo = lib.einsum('xpq,qo,pv->xov', v1ao[1], orbob, orbvb.conj())
        # v1bvo in shape(nz, noccb, nvirb)

        v1ov = xs * e_ia  # AX, here we added the orbital energy difference
        v1vo = ys * e_ia  # AY
        v1ov[:,:nocca*nvira] += v1aov.reshape(nz,-1) # here we added the coupling elements
        v1vo[:,:nocca*nvira] += v1avo.reshape(nz,-1)
        v1ov[:,nocca*nvira:] += v1bov.reshape(nz,-1)
        v1vo[:,nocca*nvira:] += v1bvo.reshape(nz,-1)
        if wfnsym is not None and mol.symmetry:
            v1ov[:,sym_forbid] = 0
            v1vo[:,sym_forbid] = 0
        hx = numpy.hstack((v1ov, -v1vo)) # in shape(nz, 2*(nocca*nvira+noccb*nvirb))
        return hx

    return vind, hdiag


class TDHF_postGSC(uhf.TDHF):

    def __init__(self, loscmf, losc_data):
        super().__init__(loscmf)
        self.losc_data = losc_data
        pass

    def gen_vind(self, loscmf=None, losc_data=None):
        if loscmf is None: 
            loscmf = self._scf
        if losc_data is None:
            losc_data = self.losc_data
        return gen_tdhf_operation_postGSC(loscmf, losc_data, singlet=self.singlet, wfnsym=self.wfnsym)
    
# region TDDFT-LOSC-Res
def gen_tdhf_operation_postLOSC_R(mf, losc_data, fock_ao=None, singlet=True, wfnsym=None):
    '''Generate function to compute

    [ A   B ][X]
    [-B* -A*][Y]
    '''
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    # assert (mo_coeff.dtype == numpy.double)
    mo_energy = losc_data['Orblosc'][0]
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    occidx = numpy.where(mo_occ==2)[0]
    viridx = numpy.where(mo_occ==0)[0]
    nocc = len(occidx)
    nvir = len(viridx)
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]

    curvature       =   losc_data['curvature']
    select_CO_idx   =   losc_data['select_CO_idx']
    U             =   losc_data['U']
    select_CO     =   select_CO_idx[0]
    kappa         =   numpy.zeros((nao,nao), dtype=float)
    Umat             =   numpy.eye(nao, dtype=float) # in shape(nao,nao)
    if select_CO is not None:
        kappa[select_CO[0]:select_CO[1], select_CO[0]:select_CO[1]] = 1 * curvature[0]
        Umat[select_CO[0]:select_CO[1], select_CO[0]:select_CO[1]] = 1 * U[0]
    Upo        =   Umat[:,:nocc] # in shape(nao,nocca)
    Upv        =   Umat[:,nocc:] # in shape(nao,nvira)
    delta_e_ia    =   kappa[:nocc, nocc:] # in shape(nocc,nvir)

    if wfnsym is not None and mol.symmetry:
        if isinstance(wfnsym, str):
            wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
        wfnsym = wfnsym % 10  # convert to D2h subgroup
        sym_forbid = _get_x_sym_table(mf) != wfnsym

    assert fock_ao is None

    e_ia   =   mo_energy[viridx] - mo_energy[occidx,None]
    hdiag  =   e_ia - delta_e_ia 
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = numpy.hstack((hdiag.ravel(), -hdiag.ravel()))

    mo_coeff = numpy.asarray(numpy.hstack((orbo,orbv)), order='F')
    vresp = mf.gen_response(singlet=singlet, hermi=0)

    def vind(xys):
        xys = numpy.asarray(xys).reshape(-1,2,nocc,nvir)
        if wfnsym is not None and mol.symmetry:
            # shape(nz,2,nocc,nvir): 2 ~ X,Y
            xys = numpy.copy(xys)
            xys[:,:,sym_forbid] = 0

        xs, ys = xys.transpose(1,0,2,3)
        # *2 for double occupancy
        dms  = lib.einsum('xov,qv,po->xpq', xs*2, orbv.conj(), orbo)
        dms += lib.einsum('xov,pv,qo->xpq', ys*2, orbv, orbo.conj())
        v1ao = vresp(dms) # = <mb||nj> Xjb + <mj||nb> Yjb
        # A ~= <ib||aj>, B = <ij||ab>
        # AX + BY
        # = <ib||aj> Xjb + <ij||ab> Yjb
        # = (<mb||nj> Xjb + <mj||nb> Yjb) Cmi* Cna
        v1ov = lib.einsum('xpq,po,qv->xov', v1ao, orbo.conj(), orbv)
        # (B*)X + (A*)Y
        # = <ab||ij> Xjb + <aj||ib> Yjb
        # = (<mb||nj> Xjb + <mj||nb> Yjb) Cma* Cni
        v1vo = lib.einsum('xpq,qo,pv->xov', v1ao, orbo, orbv.conj())
        v1ov += numpy.einsum('xia,ia->xia', xs, e_ia)  # AX
        v1vo += numpy.einsum('xia,ia->xia', ys, e_ia.conj())  # (A*)Y

        # adding the LOSC xc kernel correction
        # TODO: extending to complex value of U
        # TODO: check if we need to times 2 for the double occupancy
        X_W = lib.einsum('pq,pa,qi,xia->xpq', kappa, Upv, Upo, xs) # in shape(nz,nao,nao)
        Y_W = lib.einsum('pq,pa,qi,xia->xpq', kappa, Upv, Upo, ys) # in shape(nz,nao,nao)

        delta_AX = - lib.einsum('pi,xqp,qa->xia', Upo, X_W, Upv) # in shape(nz,nocc,nvir)
        delta_AY = - lib.einsum('pi,xqp,qa->xia', Upo, Y_W, Upv) # same as above
        delta_BX = - lib.einsum('pi,xpq,qa->xia', Upo, X_W, Upv) # same as above
        delta_BY = - lib.einsum('pi,xpq,qa->xia', Upo, Y_W, Upv) # same as above

        # excitation part
        v1ov += delta_AX # here we added the DeltaA X term
        v1ov += delta_BY # here we added the DeltaB Y term

        # de-excitation part
        v1vo += delta_BX # here we added the DeltaB X term
        v1vo += delta_AY # here we added the DeltaA Y term


        if wfnsym is not None and mol.symmetry:
            v1ov[:,sym_forbid] = 0
            v1vo[:,sym_forbid] = 0

        # (AX, -AY)
        nz = xys.shape[0]
        hx = numpy.hstack((v1ov.reshape(nz,-1), -v1vo.reshape(nz,-1)))
        return hx

    return vind, hdiag

class TDHF_postLOSC_R(rhf.TDHF):

    def __init__(self, loscmf, losc_data):
        super().__init__(loscmf)
        self.losc_data = losc_data
        pass

    def gen_vind(self, loscmf=None, losc_data=None):
        if loscmf is None: 
            loscmf = self._scf
        if losc_data is None:
            losc_data = self.losc_data
        return gen_tdhf_operation_postLOSC_R(loscmf, losc_data, singlet=self.singlet, wfnsym=self.wfnsym)
    

# region TDDFT-LOSC

def gen_tdhf_operation_postLOSC(loscmf, losc_data, fock_ao=None, singlet=True, wfnsym=None):
    '''Generate function to compute

    [ A  B][X]
    [-B -A][Y]

    with LOSC correction to both the orbital energy and the fxc kernel.
    '''
    mol = loscmf.mol
    mo_coeff = loscmf.mo_coeff
    assert (mo_coeff[0].dtype == numpy.double)
    mo_energy = losc_data['Orblosc']
    mo_occ = loscmf.mo_occ
    nao, nmo = mo_coeff[0].shape
    occidxa = numpy.where(mo_occ[0]>0)[0]
    occidxb = numpy.where(mo_occ[1]>0)[0]
    viridxa = numpy.where(mo_occ[0]==0)[0]
    viridxb = numpy.where(mo_occ[1]==0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:,occidxa] # in shape (nao,nocca)
    orbob = mo_coeff[1][:,occidxb] # in shape (nao,noccb)
    orbva = mo_coeff[0][:,viridxa] # in shape (nao,nvira)
    orbvb = mo_coeff[1][:,viridxb] # in shape (nao,nvirb)

    curvature       =   losc_data['curvature']
    select_CO_idx   =   losc_data['select_CO_idx']
    U               =   losc_data['U']
    select_CO_a     =   select_CO_idx[0]
    select_CO_b     =   select_CO_idx[1]
    kappa_a         =   numpy.zeros((nao,nao), dtype=float) # in shape(nao,nao)
    kappa_b         =   numpy.zeros((nao,nao), dtype=float) # in shape(nao,nao)
    U_a             =   numpy.eye(nao, dtype=float) # in shape(nao,nao)
    U_b             =   numpy.eye(nao, dtype=float) # in shape(nao,nao)
    if select_CO_a is not None:
        kappa_a[select_CO_a[0]:select_CO_a[1], select_CO_a[0]:select_CO_a[1]] = 1 * curvature[0]
        U_a[select_CO_a[0]:select_CO_a[1], select_CO_a[0]:select_CO_a[1]] = 1 * U[0]
    if select_CO_b is not None:
        kappa_b[select_CO_b[0]:select_CO_b[1], select_CO_b[0]:select_CO_b[1]] = 1 * curvature[1]
        U_b[select_CO_b[0]:select_CO_b[1], select_CO_b[0]:select_CO_b[1]] = 1 * U[1]
    delta_e_ia_a    =   kappa_a[:nocca, nocca:] # in shape(nocca,nvira)
    delta_e_ia_b    =   kappa_b[:noccb, noccb:] # in shape(noccb,nvirb)
    Upo_a        =   U_a[:,:nocca] # in shape(nao,nocca)
    Upv_a        =   U_a[:,nocca:] # in shape(nao,nvira)
    Upo_b        =   U_b[:,:noccb] # in shape(nao,noccb)
    Upv_b        =   U_b[:,noccb:] # in shape(nao,nvirb)

    if wfnsym is not None and mol.symmetry:
        if isinstance(wfnsym, str):
            wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
        orbsyma, orbsymb = uhf_symm.get_orbsym(mol, mo_coeff)
        wfnsym = wfnsym % 10  # convert to D2h subgroup
        orbsyma_in_d2h = numpy.asarray(orbsyma) % 10
        orbsymb_in_d2h = numpy.asarray(orbsymb) % 10
        sym_forbida = (orbsyma_in_d2h[occidxa,None] ^ orbsyma_in_d2h[viridxa]) != wfnsym
        sym_forbidb = (orbsymb_in_d2h[occidxb,None] ^ orbsymb_in_d2h[viridxb]) != wfnsym
        sym_forbid = numpy.hstack((sym_forbida.ravel(), sym_forbidb.ravel()))

    e_ia_a = mo_energy[0][viridxa] - mo_energy[0][occidxa,None] # in shape(nocc_a,nvir_a)
    e_ia_b = mo_energy[1][viridxb] - mo_energy[1][occidxb,None] # in shape(nocc_b,nvir_b)
    hdiag_a = e_ia_a - delta_e_ia_a
    hdiag_b = e_ia_b - delta_e_ia_b

    e_ia  = numpy.hstack((e_ia_a.ravel(), e_ia_b.ravel())) # in shape(nocca*nvira+noccb*nvirb,)
    hdiag = numpy.hstack((hdiag_a.ravel(), hdiag_b.ravel())) # in shape(nocca*nvira+noccb*nvirb,)
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = numpy.hstack((hdiag, -hdiag)) # in shape(2*(nocca*nvira+noccb*nvirb),)


    mem_now = lib.current_memory()[0]
    max_memory = max(2000, loscmf.max_memory*.8-mem_now)
    vresp = loscmf.gen_response(hermi=0, max_memory=max_memory)

    def vind(xys):
        nz = len(xys)
        xys = numpy.asarray(xys).reshape(nz,2,-1)
        if wfnsym is not None and mol.symmetry:
            # shape(nz,2,-1): 2 ~ X,Y
            xys = numpy.copy(xys)
            xys[:,:,sym_forbid] = 0

        xs, ys = xys.transpose(1,0,2)
        # xs ~ [Xa, Xb], in shape(nz,nocca*nvira+noccb*nvirb)
        # ys ~ [Ya, Yb], in shape(nz,nocca*nvira+noccb*nvirb)
        xa = xs[:,:nocca*nvira].reshape(nz,nocca,nvira) # in shape(nz,nocca,nvira)
        xb = xs[:,nocca*nvira:].reshape(nz,noccb,nvirb) # in shape(nz,noccb,nvirb)
        ya = ys[:,:nocca*nvira].reshape(nz,nocca,nvira) # in shape(nz,nocca,nvira)
        yb = ys[:,nocca*nvira:].reshape(nz,noccb,nvirb) # in shape(nz,noccb,nvirb)
        # dms = AX + BY
        dmsa  = lib.einsum('xov,qv,po->xpq', xa, orbva.conj(), orboa) # in shape(nz,nao,nao)
        dmsb  = lib.einsum('xov,qv,po->xpq', xb, orbvb.conj(), orbob) # in shape(nz,nao,nao)
        dmsa += lib.einsum('xov,pv,qo->xpq', ya, orbva, orboa.conj())
        dmsb += lib.einsum('xov,pv,qo->xpq', yb, orbvb, orbob.conj())

        v1ao = vresp(numpy.asarray((dmsa,dmsb))) # in shape(2,nz,nao,nao)

        v1aov = lib.einsum('xpq,po,qv->xov', v1ao[0], orboa.conj(), orbva)
        # v1aov in shape(nz, nocca, nvira)
        v1avo = lib.einsum('xpq,qo,pv->xov', v1ao[0], orboa, orbva.conj())
        # v1avo in shape(nz, nocca, nvira)
        v1bov = lib.einsum('xpq,po,qv->xov', v1ao[1], orbob.conj(), orbvb)
        # v1bov in shape(nz, noccb, nvirb)
        v1bvo = lib.einsum('xpq,qo,pv->xov', v1ao[1], orbob, orbvb.conj())
        # v1bvo in shape(nz, noccb, nvirb)

        v1ov = xs * e_ia  # AX, here we added the orbital energy difference
        v1vo = ys * e_ia  # AY
        v1ov[:,:nocca*nvira] += v1aov.reshape(nz,-1) # here we added the coupling elements
        v1vo[:,:nocca*nvira] += v1avo.reshape(nz,-1)
        v1ov[:,nocca*nvira:] += v1bov.reshape(nz,-1)
        v1vo[:,nocca*nvira:] += v1bvo.reshape(nz,-1)

        # adding the LOSC xc kernel correction
        # TODO: extending to complex value of U
        X_W_a = lib.einsum('pq,pa,qi,xia->xpq', kappa_a, Upv_a, Upo_a, xa) # in shape(nz,nao,nao)
        Y_W_a = lib.einsum('pq,pa,qi,xia->xpq', kappa_a, Upv_a, Upo_a, ya) # in shape(nz,nao,nao)
        X_W_b = lib.einsum('pq,pa,qi,xia->xpq', kappa_b, Upv_b, Upo_b, xb) # in shape(nz,nao,nao)
        Y_W_b = lib.einsum('pq,pa,qi,xia->xpq', kappa_b, Upv_b, Upo_b, yb) # in shape(nz,nao,nao)

        delta_AX_a = - lib.einsum('pi,xqp,qa->xia', Upo_a, X_W_a, Upv_a) # in shape(nz,nocca,nvira)
        delta_AY_a = - lib.einsum('pi,xqp,qa->xia', Upo_a, Y_W_a, Upv_a) # same as above
        delta_BX_a = - lib.einsum('pi,xpq,qa->xia', Upo_a, X_W_a, Upv_a) # same as above
        delta_BY_a = - lib.einsum('pi,xpq,qa->xia', Upo_a, Y_W_a, Upv_a) # same as above
        delta_AX_b = - lib.einsum('pi,xqp,qa->xia', Upo_b, X_W_b, Upv_b) # in shape(nz,noccb,nvirb)
        delta_AY_b = - lib.einsum('pi,xqp,qa->xia', Upo_b, Y_W_b, Upv_b) # same as above
        delta_BX_b = - lib.einsum('pi,xpq,qa->xia', Upo_b, X_W_b, Upv_b) # same as above
        delta_BY_b = - lib.einsum('pi,xpq,qa->xia', Upo_b, Y_W_b, Upv_b) # same as above

        # excitation part
        v1ov[:,:nocca*nvira] += delta_AX_a.reshape(nz,-1) # here we added the DeltaA X term
        v1ov[:,nocca*nvira:] += delta_AX_b.reshape(nz,-1)
        v1ov[:,:nocca*nvira] += delta_BY_a.reshape(nz,-1) # here we added the DeltaB Y term
        v1ov[:,nocca*nvira:] += delta_BY_b.reshape(nz,-1)

        # de-excitation part
        v1vo[:,:nocca*nvira] += delta_BX_a.reshape(nz,-1) # here we added the DeltaB X term
        v1vo[:,nocca*nvira:] += delta_BX_b.reshape(nz,-1)
        v1vo[:,:nocca*nvira] += delta_AY_a.reshape(nz,-1) # here we added the DeltaA Y term
        v1vo[:,nocca*nvira:] += delta_AY_b.reshape(nz,-1)


        if wfnsym is not None and mol.symmetry:
            v1ov[:,sym_forbid] = 0
            v1vo[:,sym_forbid] = 0
        hx = numpy.hstack((v1ov, -v1vo)) # in shape(nz, 2*(nocca*nvira+noccb*nvirb))
        return hx

    return vind, hdiag


class TDHF_postLOSC(uhf.TDHF):

    def __init__(self, loscmf, losc_data):
        super().__init__(loscmf)
        self.losc_data = losc_data
        pass

    def gen_vind(self, loscmf=None, losc_data=None):
        if loscmf is None: 
            loscmf = self._scf
        if losc_data is None:
            losc_data = self.losc_data
        return gen_tdhf_operation_postLOSC(loscmf, losc_data, 
                                           singlet=self.singlet, wfnsym=self.wfnsym)

TDDFT_GSC = TDHF_GSC = TDHF_postGSC
TDDFT_GSC_R = TDHF_GSC_R = TDHF_postGSC_R
TDDFT_LOSC = TDHF_LOSC = TDHF_postLOSC
TDDFT_LOSC_R = TDHF_LOSC_R = TDHF_postLOSC_R
TDA_GSC = TDA_postGSC
TDA_GSC_R = TDA_postGSC_R
TDA_LOSC = TDA_postLOSC
TDA_LOSC_R = TDA_postLOSC_R


# region For SCF-LOSC
def scflosc_analyze_R(tdobj, losc_data=None, verbose=None, 
                      parent_rep=False, corresponding_orbs=False):
    if losc_data is None:
        losc_data = tdobj.losc_data
    V = losc_data['CO_rotation'][0]
    _, inv_corresp = pyscf_losc.orbital_correspondence(V)
    log = logger.new_logger(tdobj, verbose)
    mol = tdobj.mol
    mo_coeff = tdobj._scf.mo_coeff
    mo_occ = tdobj._scf.mo_occ
    nocc = numpy.count_nonzero(mo_occ == 2)

    e_ev = numpy.asarray(tdobj.e) * nist.HARTREE2EV
    e_wn = numpy.asarray(tdobj.e) * nist.HARTREE2WAVENUMBER
    wave_length = 1e7/e_wn

    if tdobj.singlet:
        log.note('\n** Singlet excitation energies and oscillator strengths **')
    else:
        log.note('\n** Triplet excitation energies and oscillator strengths **')

    if mol.symmetry:
        orbsym = hf_symm.get_orbsym(mol, mo_coeff)
        x_sym = symm.direct_prod(orbsym[mo_occ==2], orbsym[mo_occ==0], mol.groupname)
    else:
        x_sym = None

    f_oscillator = tdobj.oscillator_strength()
    for i, ei in enumerate(tdobj.e):
        x, y = tdobj.xy[i]
        if x_sym is None:
            log.note('Excited State %3d: %12.5f eV %9.2f nm  f=%.4f',
                     i+1, e_ev[i], wave_length[i], f_oscillator[i])
        else:
            print('Symmetry NOT implemented for LOSC.')
            log.note('Excited State %3d: %12.5f eV %9.2f nm  f=%.4f',
                     i+1, e_ev[i], wave_length[i], f_oscillator[i])
        
        if log.verbose >= 2: 
            o_idx, v_idx = numpy.where(abs(x) > 0.2)
            o_list = []
            v_list = []
            for o, v in zip(o_idx, v_idx):
                log.note('  %4d -> %-4d %12.5f',
                         o+MO_BASE, v+MO_BASE+nocc, x[o,v])
                if o not in o_list:
                    o_list.append(o)
                if v not in v_list:
                    v_list.append(v)
            
            if parent_rep:
                log.note('Old CO representation:')
                x_old = V[:nocc,:nocc].T @ x @ V[nocc:,nocc:]
                o_idx_old, v_idx_old = numpy.where(abs(x_old) > 0.2)
                for o, v in zip(o_idx_old, v_idx_old):
                    log.note('  %4d -> %-4d %12.5f',
                            o+MO_BASE, v+MO_BASE+nocc, x_old[o,v])
                log.note('')

            if corresponding_orbs:
                log.note('Related orbital rotation:')
                log.note('New CO -> Old CO, amplitude')
                for o in o_list:
                    key = o+MO_BASE
                    log.note(f'Orbital {key}:')
                    for idx, val in inv_corresp[key]:
                        if abs(val) > 0.2:
                            log.note(f'{key} -> {idx}, {val}')
                for v in v_list:
                    key = v+MO_BASE+nocc
                    log.note(f'Orbital {key}:')
                    for idx, val in inv_corresp[key]:
                        if abs(val) > 0.2:
                            log.note(f'{key} -> {idx}, {val}')
                log.note('')


    if log.verbose >= logger.INFO:
        log.info('\n** Transition electric dipole moments (AU) **')
        log.info('state          X           Y           Z        Dip. S.      Osc.')
        trans_dip = tdobj.transition_dipole()
        for i, ei in enumerate(tdobj.e):
            dip = trans_dip[i]
            log.info('%3d    %11.4f %11.4f %11.4f %11.4f %11.4f',
                     i+1, dip[0], dip[1], dip[2], numpy.dot(dip, dip),
                     f_oscillator[i])

        log.info('\n** Transition velocity dipole moments (imaginary part, AU) **')
        log.info('state          X           Y           Z        Dip. S.      Osc.')
        trans_v = tdobj.transition_velocity_dipole()
        f_v = tdobj.oscillator_strength(gauge='velocity', order=0)
        for i, ei in enumerate(tdobj.e):
            v = trans_v[i]
            log.info('%3d    %11.4f %11.4f %11.4f %11.4f %11.4f',
                     i+1, v[0], v[1], v[2], numpy.dot(v, v), f_v[i])

        log.info('\n** Transition magnetic dipole moments (imaginary part, AU) **')
        log.info('state          X           Y           Z')
        trans_m = tdobj.transition_magnetic_dipole()
        for i, ei in enumerate(tdobj.e):
            m = trans_m[i]
            log.info('%3d    %11.4f %11.4f %11.4f',
                     i+1, m[0], m[1], m[2])
    return tdobj


def scflosc_analyze(tdobj, losc_data=None, verbose=None,
                    parent_rep=False, corresponding_orbs=False):
    if losc_data is None:
        losc_data = tdobj.losc_data
    V_a = losc_data['CO_rotation'][0]
    V_b = losc_data['CO_rotation'][1]
    _, inv_corresp_a = pyscf_losc.orbital_correspondence(V_a)
    _, inv_corresp_b = pyscf_losc.orbital_correspondence(V_b)
    log = logger.new_logger(tdobj, verbose)
    mol = tdobj.mol
    mo_coeff = tdobj._scf.mo_coeff
    mo_occ = tdobj._scf.mo_occ
    nocc_a = numpy.count_nonzero(mo_occ[0] == 1)
    nocc_b = numpy.count_nonzero(mo_occ[1] == 1)

    e_ev = numpy.asarray(tdobj.e) * nist.HARTREE2EV
    e_wn = numpy.asarray(tdobj.e) * nist.HARTREE2WAVENUMBER
    wave_length = 1e7/e_wn

    log.note('\n** Excitation energies and oscillator strengths **')

    if mol.symmetry:
        orbsyma, orbsymb = uhf_symm.get_orbsym(mol, mo_coeff)
        x_syma = symm.direct_prod(orbsyma[mo_occ[0]==1], orbsyma[mo_occ[0]==0], mol.groupname)
        x_symb = symm.direct_prod(orbsymb[mo_occ[1]==1], orbsymb[mo_occ[1]==0], mol.groupname)
    else:
        x_syma = None

    f_oscillator = tdobj.oscillator_strength()
    for i, ei in enumerate(tdobj.e):
        x, y = tdobj.xy[i]
        if x_syma is None:
            log.note('Excited State %3d: %12.5f eV %9.2f nm  f=%.4f',
                     i+1, e_ev[i], wave_length[i], f_oscillator[i])
        else:
            wfnsyma = rhf._analyze_wfnsym(tdobj, x_syma, x[0])
            wfnsymb = rhf._analyze_wfnsym(tdobj, x_symb, x[1])
            if wfnsyma == wfnsymb:
                wfnsym = wfnsyma
            else:
                wfnsym = '???'
            log.note('Excited State %3d: %4s %12.5f eV %9.2f nm  f=%.4f',
                     i+1, wfnsym, e_ev[i], wave_length[i], f_oscillator[i])

        if log.verbose >= 2: 
            oa_list = []
            va_list = []
            ob_list = []
            vb_list = []
            for o, v in zip(* numpy.where(abs(x[0]) > 0.2)):
                log.note('    %4da -> %4da %12.5f',
                         o+MO_BASE, v+MO_BASE+nocc_a, x[0][o,v])
                if o not in oa_list:
                    oa_list.append(o)
                if v not in va_list:
                    va_list.append(v)
            for o, v in zip(* numpy.where(abs(x[1]) > 0.2)):
                log.note('    %4db -> %4db %12.5f',
                         o+MO_BASE, v+MO_BASE+nocc_b, x[1][o,v])
                if o not in ob_list:
                    ob_list.append(o)
                if v not in vb_list:
                    vb_list.append(v)

            if parent_rep:
                log.note('Old CO representation:')
                x_old_a = V_a[:nocc_a,:nocc_a].T @ x[0] @ V_a[nocc_a:,nocc_a:]
                x_old_b = V_b[:nocc_b,:nocc_b].T @ x[1] @ V_b[nocc_b:,nocc_b:]
                o_idx_old_a, v_idx_old_a = numpy.where(abs(x_old_a) > 0.2)
                o_idx_old_b, v_idx_old_b = numpy.where(abs(x_old_b) > 0.2)
                for o, v in zip(o_idx_old_a, v_idx_old_a):
                    log.note('    %4da -> %4da %12.5f',
                            o+MO_BASE, v+MO_BASE+nocc_a, x_old_a[o,v])
                for o, v in zip(o_idx_old_b, v_idx_old_b):
                    log.note('    %4db -> %4db %12.5f',
                            o+MO_BASE, v+MO_BASE+nocc_b, x_old_b[o,v])
                log.note('')

            if corresponding_orbs:    
                log.note('Related orbital rotation:')
                log.note('New CO -> Old CO, amplitude')
                for o in oa_list:
                    key = o+MO_BASE
                    log.note(f'Orbital {key}a:')
                    for idx, val in inv_corresp_a[key]:
                        if abs(val) > 0.2:
                            log.note(f'{key}a -> {idx}a, {val}')
                for v in va_list:
                    key = v+MO_BASE+nocc_a
                    log.note(f'Orbital {key}a:')
                    for idx, val in inv_corresp_a[key]:
                        if abs(val) > 0.2:
                            log.note(f'{key}a -> {idx}a, {val}')
                for o in ob_list:
                    key = o+MO_BASE
                    log.note(f'Orbital {key}b:')
                    for idx, val in inv_corresp_b[key]:
                        if abs(val) > 0.2:
                            log.note(f'{key}b -> {idx}b, {val}')
                for v in vb_list:
                    key = v+MO_BASE+nocc_b
                    log.note(f'Orbital {key}b:')
                    for idx, val in inv_corresp_b[key]:
                        if abs(val) > 0.2:
                            log.note(f'{key}b -> {idx}b, {val}')
                log.note('')
            


    if log.verbose >= logger.INFO:
        log.info('\n** Transition electric dipole moments (AU) **')
        log.info('state          X           Y           Z        Dip. S.      Osc.')
        trans_dip = tdobj.transition_dipole()
        for i, ei in enumerate(tdobj.e):
            dip = trans_dip[i]
            log.info('%3d    %11.4f %11.4f %11.4f %11.4f %11.4f',
                     i+1, dip[0], dip[1], dip[2], numpy.dot(dip, dip),
                     f_oscillator[i])

        log.info('\n** Transition velocity dipole moments (imaginary part, AU) **')
        log.info('state          X           Y           Z        Dip. S.      Osc.')
        trans_v = tdobj.transition_velocity_dipole()
        f_v = tdobj.oscillator_strength(gauge='velocity', order=0)
        for i, ei in enumerate(tdobj.e):
            v = trans_v[i]
            log.info('%3d    %11.4f %11.4f %11.4f %11.4f %11.4f',
                     i+1, v[0], v[1], v[2], numpy.dot(v, v), f_v[i])

        log.info('\n** Transition magnetic dipole moments (imaginary part, AU) **')
        log.info('state          X           Y           Z')
        trans_m = tdobj.transition_magnetic_dipole()
        for i, ei in enumerate(tdobj.e):
            m = trans_m[i]
            log.info('%3d    %11.4f %11.4f %11.4f',
                     i+1, m[0], m[1], m[2])
    return tdobj

# region SCF-LOSC-TDA-Res

def gen_tda_operation_scfLOSC_R(loscmf, losc_data, fock_ao=None, 
                                singlet=True, wfnsym=None):
    mol = loscmf.mol
    mo_coeff = loscmf.mo_coeff
    # assert (mo_coeff.dtype == numpy.double)
    mo_energy = loscmf.mo_energy
    mo_occ = loscmf.mo_occ
    nao, nmo = mo_coeff.shape
    occidx = numpy.where(mo_occ==2)[0]
    viridx = numpy.where(mo_occ==0)[0]
    nocc = len(occidx)
    nvir = len(viridx)
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]


    curvature       =   losc_data['curvature']
    select_CO_idx   =   losc_data['select_CO_idx']
    full_U          =   losc_data['full_U']
    select_CO     =   select_CO_idx[0]
    kappa         =   numpy.zeros((nao,nao), dtype=float)
    Umat          =   full_U[0] # in shape(nao,nao)
    if select_CO is not None:
        kappa[select_CO[0]:select_CO[1], select_CO[0]:select_CO[1]] = 1 * curvature[0]
    delta_e_ia    =   kappa[:nocc, nocc:] # in shape(nocc,nvir)
    Upo        =   Umat[:,:nocc] # in shape(nao,nocca)
    Upv        =   Umat[:,nocc:] # in shape(nao,nvira)


    if wfnsym is not None and mol.symmetry:
        if isinstance(wfnsym, str):
            wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
        wfnsym = wfnsym % 10  # convert to D2h subgroup
        x_sym = _get_x_sym_table(loscmf)
        sym_forbid = x_sym != wfnsym

    if fock_ao is None:
        e_ia = mo_energy[viridx] - mo_energy[occidx,None]
        hdiag = e_ia - delta_e_ia
    else:
        fock = reduce(numpy.dot, (mo_coeff.conj().T, fock_ao, mo_coeff))
        foo = fock[occidx[:,None],occidx]
        fvv = fock[viridx[:,None],viridx]
        hdiag = fvv.diagonal() - foo.diagonal()[:,None]

    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = hdiag.ravel()

    mo_coeff = numpy.asarray(numpy.hstack((orbo,orbv)), order='F')
    vresp = loscmf.gen_response(singlet=singlet, hermi=0)

    def vind(zs):
        zs = numpy.asarray(zs).reshape(-1,nocc,nvir)
        if wfnsym is not None and mol.symmetry:
            zs = numpy.copy(zs)
            zs[:,sym_forbid] = 0

        # *2 for double occupancy
        dmov = lib.einsum('xov,qv,po->xpq', zs*2, orbv.conj(), orbo)
        v1ao = vresp(dmov)
        v1ov = lib.einsum('xpq,po,qv->xov', v1ao, orbo.conj(), orbv)
        if fock_ao is None:
            v1ov += numpy.einsum('xia,ia->xia', zs, e_ia)
        else:
            v1ov += lib.einsum('xqs,sp->xqp', zs, fvv)
            v1ov -= lib.einsum('xpr,sp->xsr', zs, foo)

        # adding the LOSC xc kernel correction
        # TODO: extending to complex value of U
        # TODO: check if we need to times 2 for the double occupancy
        W_a = lib.einsum('pq,pa,qi,xia->xpq', kappa, Upv, Upo, zs) # in shape(nz,nao,nao)
        delta_AX = - lib.einsum('pi,xqp,qa->xia', Upo, W_a, Upv) # in shape(nz,nocc,nvir)
        v1ov += delta_AX

        if wfnsym is not None and mol.symmetry:
            v1ov[:,sym_forbid] = 0
        return v1ov.reshape(v1ov.shape[0],-1)
    return vind, hdiag

class TDA_scfLOSC_R(rhf.TDA):

    def __init__(self, loscmf, losc_data):
        super().__init__(loscmf)
        self.losc_data = losc_data
        pass

    def gen_vind(self, loscmf=None, losc_data=None):
        if loscmf is None: 
            loscmf = self._scf
        if losc_data is None:
            losc_data = self.losc_data
        return gen_tda_operation_scfLOSC_R(loscmf, losc_data, 
                                           singlet=self.singlet, wfnsym=self.wfnsym)

    analyze = scflosc_analyze_R


# region SCF-LOSC-TDA

def gen_tda_operation_scfLOSC(loscmf, losc_data, fock_ao=None, wfnsym=None):
    mol = loscmf.mol
    mo_coeff = loscmf.mo_coeff
    assert (mo_coeff[0].dtype == numpy.double)
    mo_energy = loscmf.mo_energy
    mo_occ = loscmf.mo_occ
    nao, nmo = mo_coeff[0].shape
    occidxa = numpy.where(mo_occ[0]>0)[0]
    occidxb = numpy.where(mo_occ[1]>0)[0]
    viridxa = numpy.where(mo_occ[0]==0)[0]
    viridxb = numpy.where(mo_occ[1]==0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:,occidxa]
    orbob = mo_coeff[1][:,occidxb]
    orbva = mo_coeff[0][:,viridxa]
    orbvb = mo_coeff[1][:,viridxb]

    curvature       =   losc_data['curvature']
    select_CO_idx   =   losc_data['select_CO_idx']
    full_U          =   losc_data['full_U']
    select_CO_a     =   select_CO_idx[0]
    select_CO_b     =   select_CO_idx[1]
    kappa_a         =   numpy.zeros((nao,nao), dtype=float) # in shape(nao,nao)
    kappa_b         =   numpy.zeros((nao,nao), dtype=float) # in shape(nao,nao)
    U_a             =   full_U[0] # in shape(nao,nao)
    U_b             =   full_U[1] # in shape(nao,nao)
    if select_CO_a is not None:
        kappa_a[select_CO_a[0]:select_CO_a[1], select_CO_a[0]:select_CO_a[1]] = 1 * curvature[0]
    if select_CO_b is not None:
        kappa_b[select_CO_b[0]:select_CO_b[1], select_CO_b[0]:select_CO_b[1]] = 1 * curvature[1]
    delta_e_ia_a    =   kappa_a[:nocca, nocca:] # in shape(nocca,nvira)
    delta_e_ia_b    =   kappa_b[:noccb, noccb:] # in shape(noccb,nvirb)
    Upo_a        =   U_a[:,:nocca] # in shape(nao,nocca)
    Upv_a        =   U_a[:,nocca:] # in shape(nao,nvira)
    Upo_b        =   U_b[:,:noccb] # in shape(nao,noccb)
    Upv_b        =   U_b[:,noccb:] # in shape(nao,nvirb)

    if wfnsym is not None and mol.symmetry:
        if isinstance(wfnsym, str):
            wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
        orbsyma, orbsymb = uhf_symm.get_orbsym(mol, mo_coeff)
        wfnsym = wfnsym % 10  # convert to D2h subgroup
        orbsyma_in_d2h = numpy.asarray(orbsyma) % 10
        orbsymb_in_d2h = numpy.asarray(orbsymb) % 10
        sym_forbida = (orbsyma_in_d2h[occidxa,None] ^ orbsyma_in_d2h[viridxa]) != wfnsym
        sym_forbidb = (orbsymb_in_d2h[occidxb,None] ^ orbsymb_in_d2h[viridxb]) != wfnsym
        sym_forbid = numpy.hstack((sym_forbida.ravel(), sym_forbidb.ravel()))

    e_ia_a = mo_energy[0][viridxa] - mo_energy[0][occidxa,None]
    e_ia_b = mo_energy[1][viridxb] - mo_energy[1][occidxb,None]
    hdiag_a = e_ia_a - delta_e_ia_a
    hdiag_b = e_ia_b - delta_e_ia_b

    # e_ia = numpy.hstack((e_ia_a.reshape(-1), e_ia_b.reshape(-1)))
    hdiag = numpy.hstack((hdiag_a.reshape(-1), hdiag_b.reshape(-1)))
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, loscmf.max_memory*.8-mem_now)
    vresp = loscmf.gen_response(hermi=0, max_memory=max_memory)

    def vind(zs):
        nz = len(zs)
        zs = numpy.asarray(zs)
        if wfnsym is not None and mol.symmetry:
            zs = numpy.copy(zs)
            zs[:,sym_forbid] = 0

        za = zs[:,:nocca*nvira].reshape(nz,nocca,nvira)
        zb = zs[:,nocca*nvira:].reshape(nz,noccb,nvirb)
        dmova = lib.einsum('xov,qv,po->xpq', za, orbva.conj(), orboa)
        dmovb = lib.einsum('xov,qv,po->xpq', zb, orbvb.conj(), orbob)

        v1ao = vresp(numpy.asarray((dmova,dmovb)))

        v1a = lib.einsum('xpq,po,qv->xov', v1ao[0], orboa.conj(), orbva)
        v1b = lib.einsum('xpq,po,qv->xov', v1ao[1], orbob.conj(), orbvb)
        v1a += numpy.einsum('xia,ia->xia', za, e_ia_a)
        v1b += numpy.einsum('xia,ia->xia', zb, e_ia_b)

        # adding the LOSC xc kernel correction
        # TODO: extending to complex value of U
        W_a = lib.einsum('pq,pa,qi,xia->xpq', kappa_a, Upv_a, Upo_a, za) # in shape(nz,nao,nao)
        W_b = lib.einsum('pq,pa,qi,xia->xpq', kappa_b, Upv_b, Upo_b, zb) # in shape(nz,nao,nao)

        delta_AX_a = - lib.einsum('pi,xqp,qa->xia', Upo_a, W_a, Upv_a) # in shape(nz,nocca,nvira)
        delta_AX_b = - lib.einsum('pi,xqp,qa->xia', Upo_b, W_b, Upv_b) # in shape(nz,noccb,nvirb)
        v1a += delta_AX_a
        v1b += delta_AX_b

        hx = numpy.hstack((v1a.reshape(nz,-1), v1b.reshape(nz,-1)))
        if wfnsym is not None and mol.symmetry:
            hx[:,sym_forbid] = 0
        return hx

    return vind, hdiag

class TDA_scfLOSC(uhf.TDA):

    def __init__(self, loscmf, losc_data):
        super().__init__(loscmf)
        self.losc_data = losc_data
        pass

    def gen_vind(self, loscmf=None, losc_data=None):
        if loscmf is None: 
            loscmf = self._scf
        if losc_data is None:
            losc_data = self.losc_data
        return gen_tda_operation_scfLOSC(loscmf, losc_data, wfnsym=self.wfnsym)
    
    analyze = scflosc_analyze

# region SCF-LOSC-TDDFT-Res

def gen_tdhf_operation_scfLOSC_R(loscmf, losc_data, fock_ao=None, singlet=True, wfnsym=None):
    '''Generate function to compute

    [ A   B ][X]
    [-B* -A*][Y]
    '''
    mol = loscmf.mol
    mo_coeff = loscmf.mo_coeff
    # assert (mo_coeff.dtype == numpy.double)
    mo_energy = loscmf.mo_energy
    mo_occ = loscmf.mo_occ
    nao, nmo = mo_coeff.shape
    occidx = numpy.where(mo_occ==2)[0]
    viridx = numpy.where(mo_occ==0)[0]
    nocc = len(occidx)
    nvir = len(viridx)
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]

    curvature       =   losc_data['curvature']
    select_CO_idx   =   losc_data['select_CO_idx']
    full_U          =   losc_data['full_U']
    select_CO     =   select_CO_idx[0]
    kappa         =   numpy.zeros((nao,nao), dtype=float)
    Umat            =   full_U[0] # in shape(nao,nao)
    if select_CO is not None:
        kappa[select_CO[0]:select_CO[1], select_CO[0]:select_CO[1]] = 1 * curvature[0]
    Upo        =   Umat[:,:nocc] # in shape(nao,nocca)
    Upv        =   Umat[:,nocc:] # in shape(nao,nvira)
    delta_e_ia    =   kappa[:nocc, nocc:] # in shape(nocc,nvir)

    if wfnsym is not None and mol.symmetry:
        if isinstance(wfnsym, str):
            wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
        wfnsym = wfnsym % 10  # convert to D2h subgroup
        sym_forbid = _get_x_sym_table(loscmf) != wfnsym

    assert fock_ao is None

    e_ia   =   mo_energy[viridx] - mo_energy[occidx,None]
    hdiag  =   e_ia - delta_e_ia 
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = numpy.hstack((hdiag.ravel(), -hdiag.ravel()))

    mo_coeff = numpy.asarray(numpy.hstack((orbo,orbv)), order='F')
    vresp = loscmf.gen_response(singlet=singlet, hermi=0)

    def vind(xys):
        xys = numpy.asarray(xys).reshape(-1,2,nocc,nvir)
        if wfnsym is not None and mol.symmetry:
            # shape(nz,2,nocc,nvir): 2 ~ X,Y
            xys = numpy.copy(xys)
            xys[:,:,sym_forbid] = 0

        xs, ys = xys.transpose(1,0,2,3)
        # *2 for double occupancy
        dms  = lib.einsum('xov,qv,po->xpq', xs*2, orbv.conj(), orbo)
        dms += lib.einsum('xov,pv,qo->xpq', ys*2, orbv, orbo.conj())
        v1ao = vresp(dms) # = <mb||nj> Xjb + <mj||nb> Yjb
        # A ~= <ib||aj>, B = <ij||ab>
        # AX + BY
        # = <ib||aj> Xjb + <ij||ab> Yjb
        # = (<mb||nj> Xjb + <mj||nb> Yjb) Cmi* Cna
        v1ov = lib.einsum('xpq,po,qv->xov', v1ao, orbo.conj(), orbv)
        # (B*)X + (A*)Y
        # = <ab||ij> Xjb + <aj||ib> Yjb
        # = (<mb||nj> Xjb + <mj||nb> Yjb) Cma* Cni
        v1vo = lib.einsum('xpq,qo,pv->xov', v1ao, orbo, orbv.conj())
        v1ov += numpy.einsum('xia,ia->xia', xs, e_ia)  # AX
        v1vo += numpy.einsum('xia,ia->xia', ys, e_ia.conj())  # (A*)Y

        # adding the LOSC xc kernel correction
        # TODO: extending to complex value of U
        # TODO: check if we need to times 2 for the double occupancy
        X_W = lib.einsum('pq,pa,qi,xia->xpq', kappa, Upv, Upo, xs) # in shape(nz,nao,nao)
        Y_W = lib.einsum('pq,pa,qi,xia->xpq', kappa, Upv, Upo, ys) # in shape(nz,nao,nao)

        delta_AX = - lib.einsum('pi,xqp,qa->xia', Upo, X_W, Upv) # in shape(nz,nocc,nvir)
        delta_AY = - lib.einsum('pi,xqp,qa->xia', Upo, Y_W, Upv) # same as above
        delta_BX = - lib.einsum('pi,xpq,qa->xia', Upo, X_W, Upv) # same as above
        delta_BY = - lib.einsum('pi,xpq,qa->xia', Upo, Y_W, Upv) # same as above

        # excitation part
        v1ov += delta_AX # here we added the DeltaA X term
        v1ov += delta_BY # here we added the DeltaB Y term

        # de-excitation part
        v1vo += delta_BX # here we added the DeltaB X term
        v1vo += delta_AY # here we added the DeltaA Y term


        if wfnsym is not None and mol.symmetry:
            v1ov[:,sym_forbid] = 0
            v1vo[:,sym_forbid] = 0

        # (AX, -AY)
        nz = xys.shape[0]
        hx = numpy.hstack((v1ov.reshape(nz,-1), -v1vo.reshape(nz,-1)))
        return hx

    return vind, hdiag

class TDHF_scfLOSC_R(rhf.TDHF):

    def __init__(self, loscmf, losc_data):
        super().__init__(loscmf)
        self.losc_data = losc_data
        pass

    def gen_vind(self, loscmf=None, losc_data=None):
        if loscmf is None: 
            loscmf = self._scf
        if losc_data is None:
            losc_data = self.losc_data
        return gen_tdhf_operation_scfLOSC_R(loscmf, losc_data, 
                                            singlet=self.singlet, wfnsym=self.wfnsym)
    
    analyze = scflosc_analyze_R

# endregion
# region SCF-LOSC-TDDFT
def gen_tdhf_operation_scfLOSC(loscmf, losc_data, fock_ao=None, singlet=True, wfnsym=None):
    '''Generate function to compute

    [ A  B][X]
    [-B -A][Y]

    with LOSC correction to both the orbital energy and the fxc kernel.
    '''
    mol = loscmf.mol
    mo_coeff = loscmf.mo_coeff
    assert (mo_coeff[0].dtype == numpy.double)
    mo_energy = loscmf.mo_energy
    mo_occ = loscmf.mo_occ
    nao, nmo = mo_coeff[0].shape
    occidxa = numpy.where(mo_occ[0]>0)[0]
    occidxb = numpy.where(mo_occ[1]>0)[0]
    viridxa = numpy.where(mo_occ[0]==0)[0]
    viridxb = numpy.where(mo_occ[1]==0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:,occidxa] # in shape (nao,nocca)
    orbob = mo_coeff[1][:,occidxb] # in shape (nao,noccb)
    orbva = mo_coeff[0][:,viridxa] # in shape (nao,nvira)
    orbvb = mo_coeff[1][:,viridxb] # in shape (nao,nvirb)

    curvature       =   losc_data['curvature']
    select_CO_idx   =   losc_data['select_CO_idx']
    full_U          =   losc_data['full_U']
    select_CO_a     =   select_CO_idx[0]
    select_CO_b     =   select_CO_idx[1]
    kappa_a         =   numpy.zeros((nao,nao), dtype=float) # in shape(nao,nao)
    kappa_b         =   numpy.zeros((nao,nao), dtype=float) # in shape(nao,nao)
    U_a             =   full_U[0] # in shape(nao,nao)
    U_b             =   full_U[1] # in shape(nao,nao)
    if select_CO_a is not None:
        kappa_a[select_CO_a[0]:select_CO_a[1], select_CO_a[0]:select_CO_a[1]] = 1 * curvature[0]
    if select_CO_b is not None:
        kappa_b[select_CO_b[0]:select_CO_b[1], select_CO_b[0]:select_CO_b[1]] = 1 * curvature[1]
    delta_e_ia_a    =   kappa_a[:nocca, nocca:] # in shape(nocca,nvira)
    delta_e_ia_b    =   kappa_b[:noccb, noccb:] # in shape(noccb,nvirb)
    Upo_a        =   U_a[:,:nocca] # in shape(nao,nocca)
    Upv_a        =   U_a[:,nocca:] # in shape(nao,nvira)
    Upo_b        =   U_b[:,:noccb] # in shape(nao,noccb)
    Upv_b        =   U_b[:,noccb:] # in shape(nao,nvirb)

    if wfnsym is not None and mol.symmetry:
        if isinstance(wfnsym, str):
            wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
        orbsyma, orbsymb = uhf_symm.get_orbsym(mol, mo_coeff)
        wfnsym = wfnsym % 10  # convert to D2h subgroup
        orbsyma_in_d2h = numpy.asarray(orbsyma) % 10
        orbsymb_in_d2h = numpy.asarray(orbsymb) % 10
        sym_forbida = (orbsyma_in_d2h[occidxa,None] ^ orbsyma_in_d2h[viridxa]) != wfnsym
        sym_forbidb = (orbsymb_in_d2h[occidxb,None] ^ orbsymb_in_d2h[viridxb]) != wfnsym
        sym_forbid = numpy.hstack((sym_forbida.ravel(), sym_forbidb.ravel()))

    e_ia_a = mo_energy[0][viridxa] - mo_energy[0][occidxa,None] # in shape(nocc_a,nvir_a)
    e_ia_b = mo_energy[1][viridxb] - mo_energy[1][occidxb,None] # in shape(nocc_b,nvir_b)
    hdiag_a = e_ia_a - delta_e_ia_a
    hdiag_b = e_ia_b - delta_e_ia_b

    e_ia  = numpy.hstack((e_ia_a.ravel(), e_ia_b.ravel())) # in shape(nocca*nvira+noccb*nvirb,)
    hdiag = numpy.hstack((hdiag_a.ravel(), hdiag_b.ravel())) # in shape(nocca*nvira+noccb*nvirb,)
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = numpy.hstack((hdiag, -hdiag)) # in shape(2*(nocca*nvira+noccb*nvirb),)


    mem_now = lib.current_memory()[0]
    max_memory = max(2000, loscmf.max_memory*.8-mem_now)
    vresp = loscmf.gen_response(hermi=0, max_memory=max_memory)

    def vind(xys):
        nz = len(xys)
        xys = numpy.asarray(xys).reshape(nz,2,-1)
        if wfnsym is not None and mol.symmetry:
            # shape(nz,2,-1): 2 ~ X,Y
            xys = numpy.copy(xys)
            xys[:,:,sym_forbid] = 0

        xs, ys = xys.transpose(1,0,2)
        # xs ~ [Xa, Xb], in shape(nz,nocca*nvira+noccb*nvirb)
        # ys ~ [Ya, Yb], in shape(nz,nocca*nvira+noccb*nvirb)
        xa = xs[:,:nocca*nvira].reshape(nz,nocca,nvira) # in shape(nz,nocca,nvira)
        xb = xs[:,nocca*nvira:].reshape(nz,noccb,nvirb) # in shape(nz,noccb,nvirb)
        ya = ys[:,:nocca*nvira].reshape(nz,nocca,nvira) # in shape(nz,nocca,nvira)
        yb = ys[:,nocca*nvira:].reshape(nz,noccb,nvirb) # in shape(nz,noccb,nvirb)
        # dms = AX + BY
        dmsa  = lib.einsum('xov,qv,po->xpq', xa, orbva.conj(), orboa) # in shape(nz,nao,nao)
        dmsb  = lib.einsum('xov,qv,po->xpq', xb, orbvb.conj(), orbob) # in shape(nz,nao,nao)
        dmsa += lib.einsum('xov,pv,qo->xpq', ya, orbva, orboa.conj())
        dmsb += lib.einsum('xov,pv,qo->xpq', yb, orbvb, orbob.conj())

        v1ao = vresp(numpy.asarray((dmsa,dmsb))) # in shape(2,nz,nao,nao)

        v1aov = lib.einsum('xpq,po,qv->xov', v1ao[0], orboa.conj(), orbva)
        # v1aov in shape(nz, nocca, nvira)
        v1avo = lib.einsum('xpq,qo,pv->xov', v1ao[0], orboa, orbva.conj())
        # v1avo in shape(nz, nocca, nvira)
        v1bov = lib.einsum('xpq,po,qv->xov', v1ao[1], orbob.conj(), orbvb)
        # v1bov in shape(nz, noccb, nvirb)
        v1bvo = lib.einsum('xpq,qo,pv->xov', v1ao[1], orbob, orbvb.conj())
        # v1bvo in shape(nz, noccb, nvirb)

        v1ov = xs * e_ia  # AX, here we added the orbital energy difference
        v1vo = ys * e_ia  # AY
        v1ov[:,:nocca*nvira] += v1aov.reshape(nz,-1) # here we added the coupling elements
        v1vo[:,:nocca*nvira] += v1avo.reshape(nz,-1)
        v1ov[:,nocca*nvira:] += v1bov.reshape(nz,-1)
        v1vo[:,nocca*nvira:] += v1bvo.reshape(nz,-1)

        # adding the LOSC xc kernel correction
        # TODO: extending to complex value of U
        X_W_a = lib.einsum('pq,pa,qi,xia->xpq', kappa_a, Upv_a, Upo_a, xa) # in shape(nz,nao,nao)
        Y_W_a = lib.einsum('pq,pa,qi,xia->xpq', kappa_a, Upv_a, Upo_a, ya) # in shape(nz,nao,nao)
        X_W_b = lib.einsum('pq,pa,qi,xia->xpq', kappa_b, Upv_b, Upo_b, xb) # in shape(nz,nao,nao)
        Y_W_b = lib.einsum('pq,pa,qi,xia->xpq', kappa_b, Upv_b, Upo_b, yb) # in shape(nz,nao,nao)

        delta_AX_a = - lib.einsum('pi,xqp,qa->xia', Upo_a, X_W_a, Upv_a) # in shape(nz,nocca,nvira)
        delta_AY_a = - lib.einsum('pi,xqp,qa->xia', Upo_a, Y_W_a, Upv_a) # same as above
        delta_BX_a = - lib.einsum('pi,xpq,qa->xia', Upo_a, X_W_a, Upv_a) # same as above
        delta_BY_a = - lib.einsum('pi,xpq,qa->xia', Upo_a, Y_W_a, Upv_a) # same as above
        delta_AX_b = - lib.einsum('pi,xqp,qa->xia', Upo_b, X_W_b, Upv_b) # in shape(nz,noccb,nvirb)
        delta_AY_b = - lib.einsum('pi,xqp,qa->xia', Upo_b, Y_W_b, Upv_b) # same as above
        delta_BX_b = - lib.einsum('pi,xpq,qa->xia', Upo_b, X_W_b, Upv_b) # same as above
        delta_BY_b = - lib.einsum('pi,xpq,qa->xia', Upo_b, Y_W_b, Upv_b) # same as above

        # excitation part
        v1ov[:,:nocca*nvira] += delta_AX_a.reshape(nz,-1) # here we added the DeltaA X term
        v1ov[:,nocca*nvira:] += delta_AX_b.reshape(nz,-1)
        v1ov[:,:nocca*nvira] += delta_BY_a.reshape(nz,-1) # here we added the DeltaB Y term
        v1ov[:,nocca*nvira:] += delta_BY_b.reshape(nz,-1)

        # de-excitation part
        v1vo[:,:nocca*nvira] += delta_BX_a.reshape(nz,-1) # here we added the DeltaB X term
        v1vo[:,nocca*nvira:] += delta_BX_b.reshape(nz,-1)
        v1vo[:,:nocca*nvira] += delta_AY_a.reshape(nz,-1) # here we added the DeltaA Y term
        v1vo[:,nocca*nvira:] += delta_AY_b.reshape(nz,-1)


        if wfnsym is not None and mol.symmetry:
            v1ov[:,sym_forbid] = 0
            v1vo[:,sym_forbid] = 0
        hx = numpy.hstack((v1ov, -v1vo)) # in shape(nz, 2*(nocca*nvira+noccb*nvirb))
        return hx

    return vind, hdiag


class TDHF_scfLOSC(uhf.TDHF):

    def __init__(self, loscmf, losc_data):
        super().__init__(loscmf)
        self.losc_data = losc_data
        pass

    def gen_vind(self, loscmf=None, losc_data=None):
        if loscmf is None: 
            loscmf = self._scf
        if losc_data is None:
            losc_data = self.losc_data
        return gen_tdhf_operation_scfLOSC(loscmf, losc_data, 
                                           singlet=self.singlet, wfnsym=self.wfnsym)
    
    analyze = scflosc_analyze

del (OUTPUT_THRESHOLD)