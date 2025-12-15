#
# Author: Ye Li, <pkupkupku@pku.edu.cn> (this email address is valid till June 2027)
#
# In this module, we implement the TD-LOSC method in a purely POST manner.
# Notes:
# (1) The input contains:
#     - tdobj: the TD object, which is a pyscf.tdscf.TDDFT or pyscf.tdscf.TDA object.
#     - losc_data: the POST-LOSC data, which is a dictionary containing the LOSC data.
# (2) The TDA and TDDFT methods are implemented for both restricted and unrestricted cases.
#


import numpy
from pyscf import scf, tdscf, lib
from pyscf.lib import logger
from pyscf.data import nist
from pyscf.scf import hf_symm
from pyscf.scf import uhf_symm
from pyscf import symm
from pyscf.tdscf import uhf
from pyscf.tdscf import rhf
from pyscf import __config__

import pyscf_losc
from pyscf_losc import pyscf_losc_tddft

MO_BASE = getattr(__config__, 'MO_BASE', 1)

# region TDBase
def U_analyze(postTDobj, verbose=None):
    tdobj = postTDobj.tdobj
    dE = postTDobj.dE
    if dE is None:
        raise RuntimeError('postTDobj.dE is None. Please run postTDobj.kernel() first!')

    log = logger.new_logger(tdobj, verbose)
    mol = tdobj.mol
    mo_coeff = tdobj._scf.mo_coeff
    mo_occ = tdobj._scf.mo_occ
    nocc_a = numpy.count_nonzero(mo_occ[0] == 1)
    nocc_b = numpy.count_nonzero(mo_occ[1] == 1)

    e_ev = numpy.asarray(tdobj.e) * nist.HARTREE2EV
    e_wn = numpy.asarray(tdobj.e) * nist.HARTREE2WAVENUMBER
    wave_length = 1e7/e_wn

    post_e_ev = e_ev + dE * nist.HARTREE2EV
    post_e_wn = e_wn + dE * nist.HARTREE2WAVENUMBER
    post_wave_length = 1e7/post_e_wn

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
            log.note('Parent Excited State %3d: %12.5f eV %9.2f nm  f=%.4f',
                     i+1, e_ev[i], wave_length[i], f_oscillator[i])
            log.note('Post-LOSC Excited State %3d: %12.5f eV %9.2f nm',
                     i+1, post_e_ev[i], post_wave_length[i])
        else:
            wfnsyma = rhf._analyze_wfnsym(tdobj, x_syma, x[0])
            wfnsymb = rhf._analyze_wfnsym(tdobj, x_symb, x[1])
            if wfnsyma == wfnsymb:
                wfnsym = wfnsyma
            else:
                wfnsym = '???'
            log.note('Post Excited State %3d: %4s %12.5f eV %9.2f nm  f=%.4f',
                     i+1, wfnsym, e_ev[i], wave_length[i], f_oscillator[i])
            log.note('Post-LOSC Excited State %3d: %4s %12.5f eV %9.2f nm',
                     i+1, wfnsym, post_e_ev[i], post_wave_length[i])

        if log.verbose >= 2: # Added by YeLi
            for o, v in zip(* numpy.where(abs(x[0]) > 0.1)):
                log.note('    %4da -> %4da %12.5f',
                         o+MO_BASE, v+MO_BASE+nocc_a, x[0][o,v])
            for o, v in zip(* numpy.where(abs(x[1]) > 0.1)):
                log.note('    %4db -> %4db %12.5f',
                         o+MO_BASE, v+MO_BASE+nocc_b, x[1][o,v])

        # if log.verbose >= logger.INFO:
        #     for o, v in zip(* numpy.where(abs(x[0]) > 0.1)):
        #         log.info('    %4da -> %4da %12.5f',
        #                  o+MO_BASE, v+MO_BASE+nocc_a, x[0][o,v])
        #     for o, v in zip(* numpy.where(abs(x[1]) > 0.1)):
        #         log.info('    %4db -> %4db %12.5f',
        #                  o+MO_BASE, v+MO_BASE+nocc_b, x[1][o,v])

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


def R_analyze(postTDobj, verbose=None):
    tdobj = postTDobj.tdobj
    dE = postTDobj.dE
    if dE is None:
        raise RuntimeError('postTDobj.dE is None. Please run postTDobj.kernel() first!')
    
    log = logger.new_logger(tdobj, verbose)
    mol = tdobj.mol
    mo_coeff = tdobj._scf.mo_coeff
    mo_occ = tdobj._scf.mo_occ
    nocc = numpy.count_nonzero(mo_occ == 2)

    e_ev = numpy.asarray(tdobj.e) * nist.HARTREE2EV
    e_wn = numpy.asarray(tdobj.e) * nist.HARTREE2WAVENUMBER
    wave_length = 1e7/e_wn

    post_e_ev = e_ev + dE * nist.HARTREE2EV
    post_e_wn = e_wn + dE * nist.HARTREE2WAVENUMBER
    post_wave_length = 1e7/post_e_wn

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
            log.note('Parent Excited State %3d: %12.5f eV %9.2f nm  f=%.4f',
                     i+1, e_ev[i], wave_length[i], f_oscillator[i])
            log.note('Post-LOSC Excited State %3d: %12.5f eV %9.2f nm',
                     i+1, post_e_ev[i], post_wave_length[i])
        else:
            wfnsym = rhf._analyze_wfnsym(tdobj, x_sym, x)
            log.note('Parent Excited State %3d: %4s %12.5f eV %9.2f nm  f=%.4f',
                     i+1, wfnsym, e_ev[i], wave_length[i], f_oscillator[i])
            log.note('Post-LOSC Excited State %3d: %4s %12.5f eV %9.2f nm',
                     i+1, wfnsym, post_e_ev[i], post_wave_length[i])
        
        if log.verbose >= 2: # Added by Ye Li
            o_idx, v_idx = numpy.where(abs(x) > 0.1)
            for o, v in zip(o_idx, v_idx):
                log.note('  %4d -> %-4d %12.5f',
                         o+MO_BASE, v+MO_BASE+nocc, x[o,v])

        # if log.verbose >= logger.INFO:
        #     o_idx, v_idx = numpy.where(abs(x) > 0.1)
        #     for o, v in zip(o_idx, v_idx):
        #         log.info('    %4d -> %-4d %12.5f',
        #                  o+MO_BASE, v+MO_BASE+nocc, x[o,v])

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


class postLOSC_TDBase():
    def __init__(self, tdobj, losc_data):
        self.tdobj = tdobj
        self._scf  = tdobj._scf
        self.losc_data = losc_data
        self.dE = None
        pass

    def __getattr__(self, attr):
        if attr in self.tdobj.__dict__:
            return getattr(self.tdobj, attr)
        else:
            raise AttributeError(f"'post_UTDA' object has no attribute '{attr}'")


# region UTDA
def UTDA_correction(mf, losc_data, xy):
    mo_coeff = mf.mo_coeff
    assert (mo_coeff[0].dtype == numpy.double)
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

    curvature       =   losc_data['curvature']
    select_CO_idx   =   losc_data['select_CO_idx']
    U               =   losc_data['U']
    select_CO_a     =   select_CO_idx[0]
    select_CO_b     =   select_CO_idx[1]

    # (1) Prepare the U matrices and kappa matrices
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
    Upo_a        =   U_a[:,:nocca] # in shape(nao,nocca)
    Upv_a        =   U_a[:,nocca:] # in shape(nao,nvira)
    Upo_b        =   U_b[:,:noccb] # in shape(nao,noccb)
    Upv_b        =   U_b[:,noccb:] # in shape(nao,nvirb)

    # (2) Prepare the orbital energy corrections
    d_eigs = (losc_data['losc_dfa_orbital_energy']
                - losc_data['dfa_orbital_energy']) / 27.21138602 # in a.u.
    d_eigs_oa = d_eigs[0][:nocca]  # in shape(nocca,)
    d_eigs_va = d_eigs[0][nocca:]  # in shape(nvira,)
    d_eigs_ob = d_eigs[1][:noccb]  # in shape(noccb,)
    d_eigs_vb = d_eigs[1][noccb:]  # in shape(nvirb,)

    deig_mat_a = d_eigs_va[None,:] - d_eigs_oa[:,None] # in shape(nocca,nvira)
    deig_mat_b = d_eigs_vb[None,:] - d_eigs_ob[:,None] # in shape(noccb,nvirb)

    # (3) Prepare the transition amplitudes X (Y=0 for TDA)
        # xy = self.tdobj.xy, which is a list
        # xy in len nz, namely, the number of excited states
        # each xy[i] is a tuple of ((Xa,Xb), (Ya,Yb))
    X_a = numpy.asarray([xi[0][0] for xi in xy]) # in shape(nz,nocca,nvira)
    X_b = numpy.asarray([xi[0][1] for xi in xy]) # in shape(nz,noccb,nvirb)

    # (4) Calculate orbitalet transition amplitudes
    Z_a = lib.einsum('pi,zia,qa->zpq', Upo_a, X_a, Upv_a) # in shape(nz,nao,nao)
    Z_b = lib.einsum('pi,zia,qa->zpq', Upo_b, X_b, Upv_b) # in shape(nz,nao,nao)

    # (5) Calculate the TDA energy correction
    dE = (lib.einsum('ia,zia->z', deig_mat_a, numpy.abs(X_a)**2)
            + lib.einsum('ia,zia->z', deig_mat_b, numpy.abs(X_b)**2)
            - lib.einsum('pq,zpq->z', kappa_a, numpy.abs(Z_a)**2)
            - lib.einsum('pq,zpq->z', kappa_b, numpy.abs(Z_b)**2)
        )
    return numpy.real(dE)

class postLOSC_UTDA(postLOSC_TDBase):

    analyze = U_analyze

    def kernel(self, verbose=None):
        self.dE = UTDA_correction(self._scf, self.losc_data, self.tdobj.xy)

        if verbose == None:
            verbose = self.tdobj.verbose
        log = logger.new_logger(self.tdobj, verbose)
        log.note('UTDA energy corrections (in eV):\n %s', self.dE * nist.HARTREE2EV)
        return self.dE

        

# region RTDA

def RTDA_correction(mf, losc_data, xy):
    mo_coeff = mf.mo_coeff
    assert (mo_coeff.dtype == numpy.double)
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    occidx = numpy.where(mo_occ==2)[0]
    viridx = numpy.where(mo_occ==0)[0]
    nocc = len(occidx)
    nvir = len(viridx)

    curvature       =   losc_data['curvature']
    select_CO_idx   =   losc_data['select_CO_idx']
    U               =   losc_data['U']
    select_CO       =   select_CO_idx[0]

    # (1) Prepare the U matrices and kappa matrices
    kappa           =   numpy.zeros((nao,nao), dtype=float) # in shape(nao,nao)
    Umat             =   numpy.eye(nao, dtype=float) # in shape(nao,nao)
    if select_CO is not None:
        kappa[select_CO[0]:select_CO[1], select_CO[0]:select_CO[1]] = 1 * curvature[0]
        Umat[select_CO[0]:select_CO[1], select_CO[0]:select_CO[1]] = 1 * U[0]

    Upo        =   Umat[:,:nocc] # in shape(nao,nocc)
    Upv        =   Umat[:,nocc:] # in shape(nao,nvir)

    # (2) Prepare the orbital energy corrections
    d_eigs = (losc_data['losc_dfa_orbital_energy'][0]
                - losc_data['dfa_orbital_energy'][0]) / 27.21138602 # in a.u.
    d_eigs_o = d_eigs[:nocc]  # in shape(nocc,)
    d_eigs_v = d_eigs[nocc:]  # in shape(nvir,)

    deig_mat = d_eigs_v[None,:] - d_eigs_o[:,None] # in shape(nocc,nvir)

    # (3) Prepare the transition amplitudes X (Y=0 for TDA)
        # xy = self.tdobj.xy, which is a list
        # xy in len nz, namely, the number of excited states
        # each xy[i] is a tuple of (X,Y)
    X = numpy.asarray([xi[0] for xi in xy]) # in shape(nz,nocca,nvira)

    # (4) Calculate orbitalet transition amplitudes
    Z = lib.einsum('pi,zia,qa->zpq', Upo, X, Upv) # in shape(nz,nao,nao)

    # (5) Calculate the TDA energy correction
    dE = (lib.einsum('ia,zia->z', deig_mat, numpy.abs(X)**2)
            - lib.einsum('pq,zpq->z', kappa, numpy.abs(Z)**2)
        ) * 2 # multiply by 2 because the normalization of X is different
    return numpy.real(dE)

class postLOSC_RTDA(postLOSC_TDBase):

    analyze = R_analyze

    def kernel(self, verbose=None):
        self.dE = RTDA_correction(self._scf, self.losc_data, self.tdobj.xy)

        if verbose == None:
            verbose = self.tdobj.verbose
        log = logger.new_logger(self.tdobj, verbose)
        log.note('RTDA energy corrections (in eV):\n %s', self.dE * nist.HARTREE2EV)
        return self.dE

# region UTDDFT
def UTDDFT_correction(mf, losc_data, xy):
    mo_coeff = mf.mo_coeff
    assert (mo_coeff[0].dtype == numpy.double)
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

    curvature       =   losc_data['curvature']
    select_CO_idx   =   losc_data['select_CO_idx']
    U               =   losc_data['U']
    select_CO_a     =   select_CO_idx[0]
    select_CO_b     =   select_CO_idx[1]

    # (1) Prepare the U matrices and kappa matrices
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
    Upo_a        =   U_a[:,:nocca] # in shape(nao,nocca)
    Upv_a        =   U_a[:,nocca:] # in shape(nao,nvira)
    Upo_b        =   U_b[:,:noccb] # in shape(nao,noccb)
    Upv_b        =   U_b[:,noccb:] # in shape(nao,nvirb)

    # (2) Prepare the orbital energy corrections
    d_eigs = (losc_data['losc_dfa_orbital_energy']
                - losc_data['dfa_orbital_energy']) / 27.21138602 # in a.u.
    d_eigs_oa = d_eigs[0][:nocca]  # in shape(nocca,)
    d_eigs_va = d_eigs[0][nocca:]  # in shape(nvira,)
    d_eigs_ob = d_eigs[1][:noccb]  # in shape(noccb,)
    d_eigs_vb = d_eigs[1][noccb:]  # in shape(nvirb,)

    deig_mat_a = d_eigs_va[None,:] - d_eigs_oa[:,None] # in shape(nocca,nvira)
    deig_mat_b = d_eigs_vb[None,:] - d_eigs_ob[:,None] # in shape(noccb,nvirb)

    # (3) Prepare the transition amplitudes X (Y=0 for TDA)
        # xy = self.tdobj.xy, which is a list
        # xy in len nz, namely, the number of excited states
        # each xy[i] is a tuple of ((Xa,Xb), (Ya,Yb))
    X_a = numpy.asarray([xi[0][0] for xi in xy]) # in shape(nz,nocca,nvira)
    X_b = numpy.asarray([xi[0][1] for xi in xy]) # in shape(nz,noccb,nvirb)
    Y_a = numpy.asarray([xi[1][0] for xi in xy]) # in shape(nz,nocca,nvira)
    Y_b = numpy.asarray([xi[1][1] for xi in xy]) # in shape(nz,noccb,nvirb)

    # (4) Calculate orbitalet transition amplitudes
    Z_a = lib.einsum('pi,zia,qa->zpq', Upo_a, X_a, Upv_a) # in shape(nz,nao,nao)
    Z_b = lib.einsum('pi,zia,qa->zpq', Upo_b, X_b, Upv_b) # in shape(nz,nao,nao)
    Q_a = lib.einsum('pi,zia,qa->zpq', Upo_a, Y_a, Upv_a) # in shape(nz,nao,nao)
    Q_b = lib.einsum('pi,zia,qa->zpq', Upo_b, Y_b, Upv_b) # in shape(nz,nao,nao)

    # (5) Calculate the TDA energy correction
    dXAX = (lib.einsum('ia,zia->z', deig_mat_a, numpy.abs(X_a)**2)
            + lib.einsum('ia,zia->z', deig_mat_b, numpy.abs(X_b)**2)
            - lib.einsum('pq,zpq->z', kappa_a, numpy.abs(Z_a)**2)
            - lib.einsum('pq,zpq->z', kappa_b, numpy.abs(Z_b)**2)
        )
    dYAY = (lib.einsum('ia,zia->z', deig_mat_a, numpy.abs(Y_a)**2)
            + lib.einsum('ia,zia->z', deig_mat_b, numpy.abs(Y_b)**2)
            - lib.einsum('pq,zpq->z', kappa_a, numpy.abs(Q_a)**2)
            - lib.einsum('pq,zpq->z', kappa_b, numpy.abs(Q_b)**2)
        )
    dXBY = - (lib.einsum('pq,zpq,zqp->z', kappa_a, Z_a, Q_a)
              + lib.einsum('pq,zpq,zqp->z', kappa_b, Z_b, Q_b)
            )
    dYBX = dXBY # for finite system and real orbitals
    dE = dXAX + dYAY + dXBY + dYBX
    return numpy.real(dE)

class postLOSC_UTDDFT(postLOSC_TDBase):
    analyze = U_analyze

    def kernel(self, verbose=None):
        self.dE = UTDDFT_correction(self._scf, self.losc_data, self.tdobj.xy)

        if verbose == None:
            verbose = self.tdobj.verbose
        log = logger.new_logger(self.tdobj, verbose)
        log.note('UTDDFT energy corrections (in eV):\n %s', self.dE * nist.HARTREE2EV)
        return self.dE
    
# region RTDDFT
def RTDDFT_correction(mf, losc_data, xy):
    mo_coeff = mf.mo_coeff
    assert (mo_coeff.dtype == numpy.double)
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    occidx = numpy.where(mo_occ==2)[0]
    viridx = numpy.where(mo_occ==0)[0]
    nocc = len(occidx)
    nvir = len(viridx)

    curvature       =   losc_data['curvature']
    select_CO_idx   =   losc_data['select_CO_idx']
    U               =   losc_data['U']
    select_CO       =   select_CO_idx[0]

    # (1) Prepare the U matrices and kappa matrices
    kappa           =   numpy.zeros((nao,nao), dtype=float) # in shape(nao,nao)
    Umat             =   numpy.eye(nao, dtype=float) # in shape(nao,nao)
    if select_CO is not None:
        kappa[select_CO[0]:select_CO[1], select_CO[0]:select_CO[1]] = 1 * curvature[0]
        Umat[select_CO[0]:select_CO[1], select_CO[0]:select_CO[1]] = 1 * U[0]

    Upo        =   Umat[:,:nocc] # in shape(nao,nocc)
    Upv        =   Umat[:,nocc:] # in shape(nao,nvir)

    # (2) Prepare the orbital energy corrections
    d_eigs = (losc_data['losc_dfa_orbital_energy'][0]
                - losc_data['dfa_orbital_energy'][0]) / 27.21138602 # in a.u.
    d_eigs_o = d_eigs[:nocc]  # in shape(nocc,)
    d_eigs_v = d_eigs[nocc:]  # in shape(nvir,)

    deig_mat = d_eigs_v[None,:] - d_eigs_o[:,None] # in shape(nocc,nvir)

    # (3) Prepare the transition amplitudes X (Y=0 for TDA)
        # xy = self.tdobj.xy, which is a list
        # xy in len nz, namely, the number of excited states
        # each xy[i] is a tuple of (X,Y)
    X = numpy.asarray([xi[0] for xi in xy]) # in shape(nz,nocca,nvira)
    Y = numpy.asarray([xi[1] for xi in xy]) # in shape(nz,nocca,nvira)

    # (4) Calculate orbitalet transition amplitudes
    Z = lib.einsum('pi,zia,qa->zpq', Upo, X, Upv) # in shape(nz,nao,nao)
    Q = lib.einsum('pi,zia,qa->zpq', Upo, Y, Upv) # in shape(nz,nao,nao)

    # (5) Calculate the TDA energy correction
    dXAX = (lib.einsum('ia,zia->z', deig_mat, numpy.abs(X)**2)
            - lib.einsum('pq,zpq->z', kappa, numpy.abs(Z)**2)
        ) * 2 # multiply by 2 because the normalization of X is different
    dYAY = (lib.einsum('ia,zia->z', deig_mat, numpy.abs(Y)**2)
            - lib.einsum('pq,zpq->z', kappa, numpy.abs(Q)**2)
        ) * 2 # multiply by 2 because the normalization of Y is different
    dXBY = - lib.einsum('pq,zpq,zqp->z', kappa, Z, Q) * 2 
              # multiply by 2 because the normalization of X and Y are different
    dYBX = dXBY # for finite system and real orbitals
    dE = dXAX + dYAY + dXBY + dYBX
    return numpy.real(dE)

class postLOSC_RTDDFT(postLOSC_TDBase):
    analyze = R_analyze

    def kernel(self, verbose=None):
        self.dE = RTDDFT_correction(self._scf, self.losc_data, self.tdobj.xy)

        if verbose == None:
            verbose = self.tdobj.verbose
        log = logger.new_logger(self.tdobj, verbose)
        log.note('RTDDFT energy corrections (in eV):\n %s', self.dE * nist.HARTREE2EV)
        return self.dE

# region Namespace

UTDA = postLOSC_UTDA
RTDA = postLOSC_RTDA
UTDDFT = postLOSC_UTDDFT
RTDDFT = postLOSC_RTDDFT




    

    





