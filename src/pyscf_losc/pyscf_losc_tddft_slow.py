import numpy
from pyscf import scf, tdscf, lib
from pyscf.lib import logger
from pyscf.data import nist

import pyscf_losc
from pyscf_losc import pyscf_losc_tddft

def energy_filter(e, x1, energy_window=None):
    """
    Filter eigenvalues and eigenvectors by energy window, transpose eigenvectors,
    and sort by energy in ascending order.
    """
    e_ev = numpy.real(e * 27.2114)  # convert to eV and take real part

    # find the mask for the energy window
    if energy_window is None:
        mask = slice(None)  # select all
    elif energy_window[0] is None and energy_window[1] is None:
        mask = slice(None)
    elif energy_window[0] is None:
        mask = (e_ev <= energy_window[1])
    elif energy_window[1] is None:
        mask = (e_ev >= energy_window[0])
    else:
        if energy_window[0] >= energy_window[1]:
            raise ValueError("Energy window is invalid: %s" % str(energy_window))
        mask = (e_ev >= energy_window[0]) & (e_ev <= energy_window[1])
    
    # if there's no eigenvalue in the energy window, raise an error
    if not mask.any():
        raise ValueError("No eigenvalue in the energy window: %s" % str(energy_window))

    # find the indices of the eigenvalues that are within the energy window
    e_filtered = numpy.real(e[mask])
    x1_filtered = x1.T[mask]  # transpose the eigenvectors

    # sort the eigenvalues and eigenvectors by energy
    sorted_indices = numpy.argsort(e_filtered)
    e_sorted = e_filtered[sorted_indices]
    x1_sorted = x1_filtered[sorted_indices]
    return e_sorted, x1_sorted



def losc_correction_R(mf, losc_data):
    # mol = mf.mol
    mo_coeff = mf.mo_coeff
    # assert (mo_coeff.dtype == numpy.double)
    # mo_energy = losc_data['Orblosc'][0]
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    occidx = numpy.where(mo_occ==2)[0]
    viridx = numpy.where(mo_occ==0)[0]
    nocc = len(occidx)
    nvir = len(viridx)

    curvature       =   losc_data['curvature']
    select_CO_idx   =   losc_data['select_CO_idx']
    U               =   losc_data['U']
    select_CO     =   select_CO_idx[0]
    kappa         =   numpy.zeros((nao,nao), dtype=float)
    Umat             =   numpy.eye(nao, dtype=float) # in shape(nao,nao)
    if select_CO is not None:
        kappa[select_CO[0]:select_CO[1], select_CO[0]:select_CO[1]] = 1 * curvature[0]
        Umat[select_CO[0]:select_CO[1], select_CO[0]:select_CO[1]] = 1 * U[0]
    Upo        =   Umat[:,:nocc] # in shape(nao,nocca)
    Upv        =   Umat[:,nocc:] # in shape(nao,nvira)

    dA = -1.0 * lib.einsum('pq,pi,qa,pj,qb->iajb', kappa, Upo, Upv, Upo, Upv)
    dB = -1.0 * lib.einsum('pq,pi,qa,pb,qj->iajb', kappa, Upo, Upv, Upv, Upo)
    dA = dA.reshape(nocc*nvir, nocc*nvir)
    dB = dB.reshape(nocc*nvir, nocc*nvir)
    return dA, dB


def losc_correction_U(mf, losc_data):
    mo_coeff = mf.mo_coeff
    assert (mo_coeff[0].dtype == numpy.double)
    # mo_energy = mf.mo_energy
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
    Upo_a        =   U_a[:,:nocca] # in shape(nao,nocca)
    Upv_a        =   U_a[:,nocca:] # in shape(nao,nvira)
    Upo_b        =   U_b[:,:noccb] # in shape(nao,noccb)
    Upv_b        =   U_b[:,noccb:] # in shape(nao,nvirb)

    dA_aaaa = -1.0 * lib.einsum('pq,pi,qa,pj,qb->iajb', 
                                kappa_a, Upo_a, Upv_a, Upo_a, Upv_a)
    dB_aaaa = -1.0 * lib.einsum('pq,pi,qa,pb,qj->iajb', 
                                kappa_a, Upo_a, Upv_a, Upv_a, Upo_a)
    dA_bbbb = -1.0 * lib.einsum('pq,pi,qa,pj,qb->iajb',
                                kappa_b, Upo_b, Upv_b, Upo_b, Upv_b)
    dB_bbbb = -1.0 * lib.einsum('pq,pi,qa,pb,qj->iajb', 
                                kappa_b, Upo_b, Upv_b, Upv_b, Upo_b)
    # There is no correction to A_aabb and B_aabb
    dA_aaaa = dA_aaaa.reshape(nocca*nvira, nocca*nvira)
    dB_aaaa = dB_aaaa.reshape(nocca*nvira, nocca*nvira)
    dA_bbbb = dA_bbbb.reshape(noccb*nvirb, noccb*nvirb)
    dB_bbbb = dB_bbbb.reshape(noccb*nvirb, noccb*nvirb)
    return dA_aaaa, dA_bbbb, dB_aaaa, dB_bbbb


# region Slow TDA R
class slow_TDA_R(pyscf_losc_tddft.TDA_scfLOSC_R):
    """
    Class for sloving RTDA equations by directly diagonalizing the orbital rotation Hessian.
    """
    energy_window = [None, None]

    def get_orhess(self):
        """
        Get the orbital rotation Hessian.
        """
        if self.singlet:
            A, _ = tdscf.rhf.get_ab(self._scf)
            Nocc, Nvir = A.shape[0], A.shape[1]
            dA, _ = losc_correction_R(self._scf, self.losc_data)
            return A.reshape(Nocc * Nvir, Nocc * Nvir) + dA
        else:
            raise ValueError("Triplet excitation not implemented for slow RTDA.")
        
    def kernel(self):
        self.e, x1 = numpy.linalg.eigh(self.get_orhess())
        self.e, x1 = energy_filter(self.e, x1, self.energy_window)
        # After the filter, x1 is transposed
        self.converged = [True for e in self.e]

        nocc = (self._scf.mo_occ>0).sum()
        nmo = self._scf.mo_occ.size
        nvir = nmo - nocc
        # 1/sqrt(2) because self.x is for alpha excitation and 2(X^+*X) = 1
        self.xy = [(xi.reshape(nocc,nvir)*numpy.sqrt(.5),0) for xi in x1]
        self._finalize()
        return self.e, self.xy, x1
    
# region Slow TDA U
class slow_TDA_U(pyscf_losc_tddft.TDA_scfLOSC):
    """
    Class for sloving UTDA equations by directly diagonalizing the orbital rotation Hessian.
    """
    energy_window = [None, None]

    def get_orhess(self):
        """
        Get the orbital rotation Hessian.
        """
        A, _ = tdscf.uhf.get_ab(self._scf)
        A_aaaa, A_aabb, A_bbbb = A
        A_bbaa = A_aabb.transpose(2,3,0,1)
        Nocc_a, Nvir_a = A_aaaa.shape[0], A_aaaa.shape[1]
        Nocc_b, Nvir_b = A_bbbb.shape[0], A_bbbb.shape[1]
        A_aaaa = A_aaaa.reshape(Nocc_a * Nvir_a, Nocc_a * Nvir_a)
        A_bbbb = A_bbbb.reshape(Nocc_b * Nvir_b, Nocc_b * Nvir_b)
        A_aabb = A_aabb.reshape(Nocc_a * Nvir_a, Nocc_b * Nvir_b)
        A_bbaa = A_bbaa.reshape(Nocc_b * Nvir_b, Nocc_a * Nvir_a)
        dA_aaaa, dA_bbbb, _, _ = losc_correction_U(self._scf, self.losc_data)
        return numpy.block([[A_aaaa + dA_aaaa, A_aabb],
                             [A_bbaa, A_bbbb + dA_bbbb]])
    

    def kernel(self):
        self.e, x1 = numpy.linalg.eigh(self.get_orhess())
        self.e, x1 = energy_filter(self.e, x1, self.energy_window)
        self.converged = [True for e in self.e]
        nmo = self._scf.mo_occ[0].size
        nocca = (self._scf.mo_occ[0]>0).sum()
        noccb = (self._scf.mo_occ[1]>0).sum()
        nvira = nmo - nocca
        nvirb = nmo - noccb
        self.xy = [((xi[:nocca*nvira].reshape(nocca,nvira),  # X_alpha
                    xi[nocca*nvira:].reshape(noccb,nvirb)), # X_beta
                    (0, 0))  # (Y_alpha, Y_beta)
                    for xi in x1]
        self._finalize()
        return self.e, self.xy, x1
    
# region Slow TDDFT R
class slow_TDDFT_R(pyscf_losc_tddft.TDHF_scfLOSC_R):
    """
    Class for sloving Restricted TDDFT equations 
    by directly diagonalizing the orbital rotation Hessian.
    """
    energy_window = [0, None]

    def get_orhess(self):
        """
        Get the orbital rotation Hessian.
        """
        if self.singlet:
            A, B = tdscf.rhf.get_ab(self._scf)
            Nocc, Nvir = A.shape[0], A.shape[1]
            A = A.reshape(Nocc * Nvir, Nocc * Nvir)
            B = B.reshape(Nocc * Nvir, Nocc * Nvir)
            dA, dB = losc_correction_R(self._scf, self.losc_data)
            # Currently, only real-valued orbitals are supported
            #  [ A ,  B ]
            #  [-B*, -A*] in principle
            return numpy.block([[A+dA, B+dB], [-B-dB, -A-dA]])
        else:
            raise ValueError("Triplet excitation not implemented for slow RTDA.")
        
    def kernel(self):
        self.e, x1 = numpy.linalg.eig(self.get_orhess())
        self.e, x1 = energy_filter(self.e, x1, self.energy_window)
        # After the filter, x1 is transposed
        self.converged = [True for e in self.e]

        nocc = (self._scf.mo_occ>0).sum()
        nmo = self._scf.mo_occ.size
        nvir = nmo - nocc
        def norm_xy(z):
            x, y = z.reshape(2,nocc,nvir)
            norm = lib.norm(x)**2 - lib.norm(y)**2
            norm = numpy.sqrt(.5/norm)  # normalize to 0.5 for alpha spin
            return x*norm, y*norm
        self.xy = [norm_xy(z) for z in x1]
        self._finalize()
        return self.e, self.xy
    

# region Slow TDDFT U
class slow_TDDFT_U(pyscf_losc_tddft.TDHF_scfLOSC):
    """
    Class for sloving Unrestricted TDDFT equations 
    by directly diagonalizing the orbital rotation Hessian.
    """
    energy_window = [0, None]   

    def get_orhess(self):
        """
        Get the orbital rotation Hessian.
        """
        A, B = tdscf.uhf.get_ab(self._scf)
        A_aaaa, A_aabb, A_bbbb = A
        B_aaaa, B_aabb, B_bbbb = B
        del A, B

        A_bbaa = A_aabb.transpose(2,3,0,1)
        B_bbaa = B_aabb.transpose(2,3,0,1)
        Nocc_a, Nvir_a = A_aaaa.shape[0], A_aaaa.shape[1]
        Nocc_b, Nvir_b = A_bbbb.shape[0], A_bbbb.shape[1]
        A_aaaa = A_aaaa.reshape(Nocc_a * Nvir_a, Nocc_a * Nvir_a)
        A_bbbb = A_bbbb.reshape(Nocc_b * Nvir_b, Nocc_b * Nvir_b)
        A_aabb = A_aabb.reshape(Nocc_a * Nvir_a, Nocc_b * Nvir_b)
        A_bbaa = A_bbaa.reshape(Nocc_b * Nvir_b, Nocc_a * Nvir_a)
        B_aaaa = B_aaaa.reshape(Nocc_a * Nvir_a, Nocc_a * Nvir_a)
        B_bbbb = B_bbbb.reshape(Nocc_b * Nvir_b, Nocc_b * Nvir_b)
        B_aabb = B_aabb.reshape(Nocc_a * Nvir_a, Nocc_b * Nvir_b)
        B_bbaa = B_bbaa.reshape(Nocc_b * Nvir_b, Nocc_a * Nvir_a)

        dA_aaaa, dA_bbbb, dB_aaaa, dB_bbbb = losc_correction_U(self._scf, self.losc_data)
        # Currently, only real-valued orbitals are supported
        #  [ A ,  B ]
        #  [-B*, -A*] in principle
        A = numpy.block([[A_aaaa + dA_aaaa, A_aabb], [A_bbaa, A_bbbb + dA_bbbb]])
        B = numpy.block([[B_aaaa + dB_aaaa, B_aabb], [B_bbaa, B_bbbb + dB_bbbb]])
        return numpy.block([[A, B], [-B, -A]])
    
    def kernel(self):
        self.e, x1 = numpy.linalg.eig(self.get_orhess())
        self.e, x1 = energy_filter(self.e, x1, self.energy_window)
        self.converged = [True for e in self.e]
        nmo = self._scf.mo_occ[0].size
        nocca = (self._scf.mo_occ[0]>0).sum()
        noccb = (self._scf.mo_occ[1]>0).sum()
        nvira = nmo - nocca
        nvirb = nmo - noccb
        xy = []
        for i, z in enumerate(x1):
            x, y = z.reshape(2,-1)
            norm = lib.norm(x)**2 - lib.norm(y)**2
            if norm > 0:
                norm = 1/numpy.sqrt(norm)
                xy.append(((x[:nocca*nvira].reshape(nocca,nvira) * norm,  # X_alpha
                            x[nocca*nvira:].reshape(noccb,nvirb) * norm), # X_beta
                           (y[:nocca*nvira].reshape(nocca,nvira) * norm,  # Y_alpha
                            y[nocca*nvira:].reshape(noccb,nvirb) * norm)))# Y_beta
        self.xy = xy
        self._finalize()
        return self.e, self.xy
    


