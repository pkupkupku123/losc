import pyscf
import py_losc
import numpy as np
from numpy import reshape
from pyscf import gto, scf
from pyscf_losc import utils
from pyscf_losc import options

#############################################################################
# For the starting PySCF SCF calculation, save the .chk file. Initial guess #
# for the SCF-LOSC calculation will be generated based on the file.         #
#############################################################################

#: `py_losc.DFAInfo` object for B3LYP functional.
B3LYP = py_losc.DFAInfo(0.8, 0.2, 'B3LYP', 0, 0)

#: `py_losc.DFAInfo` object for SVWN functional.
SVWN = py_losc.DFAInfo(1.0, 0, 'SVWN', 0, 0)

#: `py_losc.DFAInfo` object for BLYP functional.
BLYP = py_losc.DFAInfo(1.0, 0, 'BLYP', 0, 0)

#: `py_losc.DFAInfo` object for PBE functional.
PBE = py_losc.DFAInfo(1.0, 0, 'PBE', 0, 0)

#: `py_losc.DFAInfo` object for pure GGA type functional.
GGA = py_losc.DFAInfo(1.0, 0, 'Pure GGA functional', 0, 0)

#: `py_losc.DFAInfo` object for PBE0 functional.
PBE0 = py_losc.DFAInfo(0.75, 0.25, 'PBE0', 0, 0)

#: `py_losc.DFAInfo` object for CAM-B3LYP functional.
CAM_B3LYP = py_losc.DFAInfo(0.81, 0.19, 'CAM-B3LYP', 0.460, 0.33)

#: `py_losc.DFAInfo` object for WB97 functional.
WB97 = py_losc.DFAInfo(1.0, 0.0, 'WB97', 1.0, 0.4)
WB97_TUNED = py_losc.DFAInfo(0.785, 0.0, 'WB97', 1.0, 0.4)

#: `py_losc.DFAInfo` object for WB97X functional.
WB97X = py_losc.DFAInfo(0.842, 0.158, 'WB97X', 0.842, 0.3)
WB97X_TUNED = py_losc.DFAInfo(0.743, 0.158, 'WB97X', 0.842, 0.3)

#: `py_losc.DFAInfo` object for WB97X-V functional.
WB97X_V = py_losc.DFAInfo(0.833, 0.167, 'WB97X-V', 0.833, 0.3)

#: `py_losc.DFAInfo` object for WB97M-V functional.
WB97M_V = py_losc.DFAInfo(0.85, 0.15, 'WB97M-V', 0.85, 0.3)
WB97M_V_TUNED = py_losc.DFAInfo(0.64, 0.15, 'WB97M-V', 0.85, 0.3)


def _validate_inp(mf):
    """Validate the input molecular structure, global option, and DFA 
    wavefunction for the LOSC calculation.

    Input
    -----
    mf : pyscf.dft.rks.RKS or pyscf.dft.uks.UKS
        Restricted/Unrestricted KS SCF object.
    """

    # chek type
    if not isinstance(mf, pyscf.dft.rks.RKS):
        if not isinstance(mf, pyscf.dft.uks.UKS):
            raise Exception(
                f"Invalid input: {mf}."
            )

    # check symmetry
    if mf.mol.symmetry != False:
        raise Exception("Sorry, LOSC only supports C1 symmetry.")

    # check functional
    # if mf.omega != None:
    #     raise Exception("Sorry, LOSC does not support range-separated \
    #         functional.") 
    

def post_scf_losc(dfa_info, mf, orbital_energy_unit='eV', verbose=1, occ=None,
                  return_losc_data=False, window=None, fitbasis='def2-qzvpp-ri'):
    """Perform the post-SCF-LOSC calculation.

    Parameters
    ----------
    dfa_info : ?????????? py_losc.DFAInfo
        The information of the parent DFA, including the weights of exchanges.
    mf : pyscf.dft.rks.RKS or pyscf.dft.uks.UKS
        The converged starting wavefunction from PySCF calculation.
    orbital_energy_unit : {'eV', 'au'}, default to 'eV'.
        The unit of orbital energies in the output.

        - 'eV' : electronvolt.
        - 'au' : atomic unit, hartree.
    verbose : int, default=1
        Print level. 0 means print nothing. 1 means normal print level.
    occ : dict, default=None
        A dictionary that specifies the customized occupation number. All the
        str in this dict is case-insensitive.

        The structure of the dict:
        {
            'alpha': # for alpha spin
            {
                int or str: float,
            }
            'beta': # for beta spin
            {
                int or str: float,
            }
        }

        - The key-value pair (int: float): int refers to orbital index 
        (0-based); float refers to the customized occupation number (in the
        range of [0, 1]).
        - The key-value pair (str: float): str can be 'homo' or 'lumo', which
        refers to HOMO and LUMO orbital respectively; floats refers to the
        customized occupation number (in the range of [0, 1]).

        Example:
        >> H2 system: charge=0, mult=1
        The aufbau occupation numbers of the system are:
        alpha occ: 1, 0, 0, ...
        bdta occ: 1, 0, 0, ...
        >> Customized occ dictionary:
        occ={
            'alpha': {'homo': 0.5}
            'beta': {'3': 0.7}
        }
        >> Resulted occupation numbers:
        alpha occ: 0.5, 0, 0, ...
        beta occ: 1, 0, 0, 0.7, ...

        Note that using `homo` and the orbital index of the HOMO as the key
        simultaneously will cause ambiguity to specify the HOMO occupation 
        number, so it is not allowed. The same rule applies to the LUMO.
    return_losc_data : bool, default=False
        Return LOSC data or not.
    window : [float, float], optional
        This variable is used to specify the orbital energy window in eV. 
        All the orbitals whose energies are in the window will be selected to
        do the LOSC localization. If no window is specified, all the orbitals
        will be used.

    Returns
    -------
    total_energy : float
        The total energy from the post-SCF-LOSC calculation.
    orbital_energies : [np.array, ...]
        Orbital energies from the post-SCF-LOSC calculation. If the input mf
        is a pyscf.dft.rks.RKS object, `otbital_energies` only includes the 
        alpha orbital energies. If the input mf is pyscf.dft.uks.UKS object,
        `orbital_energies` includes both alpha and beta energies in order.
    losc_data : dict
        losc_data will be returned if `return_losc_data` is True. `losc_data`
        contains the data of LOSC calculations.
    """

    #########################################################################
    # step 1: sanity-check of input dfa wfn and other parameters            #
    #########################################################################
    _validate_inp(mf)

    if orbital_energy_unit not in ['eV', 'au']:
        raise Exception(
            'Input for "orbital_energy_unit" has to be "eV" or "au".'
        )
    local_print = utils.init_local_print(verbose, mf.mol)

    # print header
    out_file = mf.mol.output
    local_print(1, "#######################################################")
    local_print(1, "#                                                     #")
    local_print(1, "#                    post-SCF-LOSC                    #")
    local_print(1, "#                                                     #")
    local_print(1, "#######################################################")
    # set nspin value based on mf
    #     nspin = 1 if isinstance(mf, pyscf.dft.rks.RKS)
    #     nspin = 2 if isinstance(mf, pyscf.dft.uks.UKS)
    if isinstance(mf, pyscf.dft.rks.RKS):
        nspin = 1
    else:
        nspin = 2
    
    #########################################################################
    # step 2: build occupation matrix                                       #
    #########################################################################
    #         see func `form_occ()` in utils.py
    if occ == None:
        occ = {}
    _, _, occ_val = utils.form_occ(mf, occ)
    nelec = [sum(x) for x in occ_val]
    
    #########################################################################
    # step 3: map needed matrices                                           #
    #########################################################################

    # step 3.1: CO coefficients
    if nspin == 1:
        C_co = [np.asarray(mf.mo_coeff), np.asarray(mf.mo_coeff)] 
    else:
        C_co = [np.asarray(mf.mo_coeff[0]), np.asarray(mf.mo_coeff[1])] 

    # step 3.2: Fock matrix
    if nspin == 1:
        H_ao = [np.asarray(mf.get_fock()), np.asarray(mf.get_fock())]
    else:
        H_ao = [np.asarray(mf.get_fock()[0]), np.asarray(mf.get_fock()[1])]

    # step 3.3: vector ao dipole integrals
    mf.mol.set_common_origin([0, 0, 0])
    D_ao = np.asarray(mf.mol.intor('int1e_r'))
    nao = mf.mol.nao_nr
    # step 3.4: overlap matrix
    S = np.asarray(mf.mol.intor('int1e_ovlp'))
    # step 3.5: density matrix
    #           mf.make_rdm1(mf.mocoeff, mo_occ)
    if nspin == 1:
        total_D = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
        alpha_D = [0.5 * val for val in total_D]
        D = [np.asarray(alpha_D), np.asarray(alpha_D)] 
    else:
        D = [
            np.asarray(mf.make_rdm1(mf.mo_coeff, mf.mo_occ)[0]),
            np.asarray(mf.make_rdm1(mf.mo_coeff, mf.mo_occ)[1]),
        ]

    #########################################################################
    # step 4: LOSC localization                                             #
    #########################################################################
    #local_print(1, f'{options}')
    # step 4.1: select COs
    def select_CO(mf, spin, window):
        """Select COs to do the LOSC localization"""
        if isinstance(mf, pyscf.dft.rks.RKS):
            nspin = 1
        else:
            nspin = 2
        if nspin == 1:
            eig = [np.asarray(mf.mo_energy), np.asarray(mf.mo_energy)][spin]
        else:
            eig = [
                np.asarray(mf.mo_energy)[0],
                np.asarray(mf.mo_energy)[1]
            ][spin]
        nao = mf.mol.nao_nr()
        if not window or nelec[spin] == 0:
            return None
        if len(window) != 2:
            raise Exception(
                'Invalid window: wrong size of window.'
            )
        if window[0] >= window[1]:
            raise Exception(
                'Invalid window: left bound >= right bound.'
            )
        idx_start = next(filter(
            lambda x: x[1] * 27.21132 >= window[0], enumerate(eig)
        ), [nao])[0]
        idx_end = next(filter(
            lambda x: x[1] * 27.21132 >= window[1], enumerate(eig)
        ), [nao])[0]
        if idx_end - idx_start <= 0:
            raise Exception(
                'LOSC window is too tight. No COs in the window.'
            )
        return (idx_start, idx_end)
    local_print(1, '##################################')
    local_print(1, '#    LOSC Localization Window    #')
    local_print(1, '##################################')
    select_CO_idx = [None] * nspin
    for s in range(nspin):
        idx = select_CO(mf, s, window)
        if not idx:
            local_print(1, f'localization COs index (spin={s}): All COs')
        else:
            local_print(
                1, f'localization COs index (spin={s}): [{idx[0]}, {idx[1]})'
            )
        select_CO_idx[s] = idx

    # step 4.2: create losc localizer object
    C_lo = [None] * nspin
    U = [None] * nspin
    for s in range(nspin):
        if select_CO_idx[s]:
            idx_start, idx_end = select_CO_idx[s]
            C_co_chosen = C_co[s][:, list(range(idx_start, idx_end))]
        else:
            C_co_chosen = C_co[s]
        localizer_version = options.get_param('localizer', 'version')
        if localizer_version == 2:
            localizer = py_losc.LocalizerV2(C_co_chosen, H_ao[s], D_ao)
            localizer.set_c(options.get_param('localizer', 'v2_parameter_c'))
            localizer.set_gamma(
                options.get_param('localizer', 'v2_parameter_gamma')
            )
        else:
            raise Exception(
                f'Non-supporting localization version: {localizer_version}.'
            )
        localizer.set_max_iter(options.get_param('localizer', 'max_iter'))
        localizer.set_convergence(
            options.get_param('localizer', 'convergence')
        )
        localizer.set_random_permutation(
            options.get_param('localizer', 'random_permutation')
        )
    # step 4.3: compute LOs
        C_lo[s], U[s] = localizer.lo_U()
        # print localization status
        local_print(1, '##################################')
        local_print(1, f'#    Localization for spin: {s}    #')
        local_print(1, '##################################')
        local_print(1, f' ==> iteration steps:      {localizer.steps()}')
        local_print(
            1, f' ==> cost function value: {localizer.cost_func(C_lo[s])}' 
        )
        if localizer.is_converged():
            local_print(
                1, ' ==> convergence:          True'
            )
        else:
            local_print(
                1, ' ==> convergence:          False, WARNING!!!'
            )
    
    #########################################################################
    # step 5: compute LOSC curvature matrix                                 #
    #########################################################################
    # step 5.1: density fitting
    #           see func form_df_matrix(mf, C_lo) in utils.py
    df_pii, df_Vpq_inv = utils.form_df_matrix(mf, C_lo, dfa_info, df_basis=fitbasis)

    # step 5.2: build weights of grid points
    grid_w = mf.grids.weights

    # step 5.3: compute values of LOs on grid points
    #           see func form_grid_lo() in utils.py
    grid_lo = [utils.form_grid_lo(mf, C_lo_) for C_lo_ in C_lo]

    # step 5.4: build LOSC curvature matrices
    j_x_separation    = options.get_param('curvature', 'j_x_separation')
    curvature_version = options.get_param('curvature', 'version')
    if j_x_separation:
        curvature_J = [None] * nspin
        curvature_X = [None] * nspin
        for s in range(nspin):
            if curvature_version == 2:
                curvature_helper = py_losc.CurvatureV2(dfa_info, 
                                                       df_pii[s],
                                                       df_Vpq_inv, 
                                                       grid_lo[s],
                                                       grid_w)
                curvature_helper.set_tau(options.get_param(
                    'curvature', 'v2_parameter_tau'
                ))  
                curvature_helper.set_zeta(options.get_param(
                    'curvature', 'v2_parameter_zeta'
                ))
            elif curvature_version == 1:
                curvature_helper = py_losc.CurvatureV1(dfa_info, 
                                                       df_pii[s],
                                                       df_Vpq_inv, 
                                                       grid_lo[s],
                                                       grid_w)
                curvature_helper.set_tau(options.get_param(
                    'curvature', 'v1_parameter_tau'
                ))
            else:
                raise Exception(
                    f'Unsupported curvature version: {curvature_version}.'
                )
            curvature_J[s] = curvature_helper.kappa_J()
            curvature_X[s] = curvature_helper.kappa_LDAX()
    else:
        curvature = [None] * nspin
        for s in range(nspin):
            if curvature_version == 2:
                curvature_helper = py_losc.CurvatureV2(dfa_info, 
                                                       df_pii[s],
                                                       df_Vpq_inv, 
                                                       grid_lo[s],
                                                       grid_w)
                curvature_helper.set_tau(options.get_param(
                    'curvature', 'v2_parameter_tau'
                ))  
                curvature_helper.set_zeta(options.get_param(
                    'curvature', 'v2_parameter_zeta'
                ))
            elif curvature_version == 1:
                curvature_helper = py_losc.CurvatureV1(dfa_info, 
                                                       df_pii[s],
                                                       df_Vpq_inv,
                                                       grid_lo[s],
                                                       grid_w)
                curvature_helper.set_tau(options.get_param(
                    'curvature', 'v1_parameter_tau'
                ))
            else:
                raise Exception(
                    f'Unsupported curvature version: {curvature_version}.'
                )
            curvature[s] = curvature_helper.kappa()

    #########################################################################
    # step 6: compute LOSC local occupation matrix                          #
    #########################################################################
    local_occ = [None] * nspin
    for s in range(nspin):
        local_occ[s] = py_losc.local_occupation(C_lo[s], S, D[s])

    # print matrix into PySCF output file
    local_print(3, '##################################')
    local_print(3, '#   Details of Matrices in LOSC  #')
    local_print(3, '##################################')
    for s in range(nspin):
        local_print(3, '')
        local_print(3, f'CO coefficient matrix.T: spin={s}')
        utils.print_full_matrix(C_co[s].T, mf.mol)
    for s in range(nspin):
        local_print(3, '')
        local_print(3, f'LO coefficient matrix.T: spin={s}')
        utils.print_full_matrix(C_lo[s].T, mf.mol)
    for s in range(nspin):
        local_print(3, '')
        local_print(3, f'LO U matrix: spin={s}')
        utils.print_full_matrix(U[s], mf.mol)
    for s in range(nspin):
        local_print(3, '')
        if j_x_separation:
            local_print(3, f'Curvature: J matrix: spin={s}')
            utils.print_sym_matrix(curvature_J[s], mf.mol)
            local_print(3, f'Curvature: X matrix: spin={s}')
            utils.print_sym_matrix(curvature_X[s], mf.mol)
        else:
            local_print(3, f'Curvature: spin={s}')
            utils.print_sym_matrix(curvature[s], mf.mol)
    for s in range(nspin):
        local_print(3, '')
        local_print(3, f'Local Occupation matrix: spin={s}')
        utils.print_sym_matrix(local_occ[s], mf.mol)

    #########################################################################
    # step 7: calculate LOSC corrections                                    #
    #########################################################################
    H_losc = [None] * nspin
    E_losc = [None] * nspin
    losc_eigs = [None] * nspin
    if orbital_energy_unit == 'au':
        eig_factor = 1.0
    else:
        eig_factor = 27.21138602
    
    if j_x_separation:
        for s in range(nspin):
            # build losc effective Hamiltonian
            H_losc[s] = py_losc.ao_hamiltonian_correction_LDA(
                    S, C_lo[s], curvature_J[s], curvature_X[s], local_occ[s]
                    )
            # compute losc energy correction
            E_losc[s] = py_losc.energy_correction_LDA(
                    curvature_J[s], curvature_X[s], local_occ[s]
                    )
            # compute corrected orbital energies
            losc_eigs[s] = np.array(py_losc.orbital_energy_post_scf(
                    H_ao[s], H_losc[s], C_co[s]
                    )) * eig_factor
        E_losc_tot = 2 * E_losc[0] if nspin == 1 else sum(E_losc)
        E_losc_dfa_tot = mf.e_tot + E_losc_tot
    else:
        for s in range(nspin):
            # build losc effective Hamiltonian
            H_losc[s] = py_losc.ao_hamiltonian_correction(
                S, C_lo[s], curvature[s], local_occ[s]
            )
            # compute losc energy correction
            E_losc[s] = py_losc.energy_correction(curvature[s], local_occ[s])
            # compute corrected orbital energies
            losc_eigs[s] = np.array(py_losc.orbital_energy_post_scf(
                H_ao[s], H_losc[s], C_co[s]
            )) * eig_factor
        E_losc_tot = 2 * E_losc[0] if nspin == 1 else sum(E_losc)
        E_losc_dfa_tot = mf.e_tot + E_losc_tot


    #########################################################################
    # step 8: pack LOSC data into dict & print energies to output           #
    #########################################################################
    if nspin == 1:
        dfa_eigs = [
            np.asarray(mf.mo_energy * eig_factor),
            np.asarray(mf.mo_energy * eig_factor)
        ]
    if nspin == 2:
        dfa_eigs = [
            np.asarray(mf.mo_energy[0] * eig_factor),
            np.asarray(mf.mo_energy[1] * eig_factor)
        ]
    if occ == None:
        occ = {}
    else:
        occ = occ
    losc_data = {}
    losc_data['occ'] = occ
    losc_data['losc_type'] = 'post-SCF-LOSC'
    losc_data['orbital_energy_unit'] = orbital_energy_unit
    losc_data['nspin'] = nspin
    if j_x_separation:
        losc_data['curvature_J'] = curvature_J
        losc_data['curvature_X'] = curvature_X
    else:
        losc_data['curvature'] = curvature
    losc_data['C_lo'] = C_lo
    losc_data['dfa_energy'] = mf.e_tot
    if nspin == 1:
        losc_data['dfa_orbital_energy'] = [mf.mo_energy * eig_factor]
    else:
        losc_data['dfa_orbital_energy'] = mf.mo_energy * eig_factor
    losc_data['losc_energy']        = E_losc_tot
    losc_data['losc_dfa_energy']    = E_losc_tot + mf.e_tot
    losc_data['losc_dfa_orbital_energy'] = losc_eigs
    losc_data['local_occ']          = local_occ # added by Ye Li
    losc_data['S']                  = S # added by Ye Li
    losc_data['U']                  = U # added by Ye Li
    losc_data['select_CO_idx']      = select_CO_idx # added by Ye Li
    losc_data['Orblosc']            = [losc_eigs[s] / eig_factor for s in range(nspin)] 
    # in a.u., added by Ye Li

    # print energies to output
    local_print(1, '##################################')
    local_print(1, '#  post-SCF-LOSC Energy Summary  #')
    local_print(1, '##################################')
    utils.print_total_energies(1, mf.mol, losc_data)
    utils.print_orbital_energies(1, mf, losc_data, window=window)


    if return_losc_data:
        return E_losc_dfa_tot, losc_eigs, losc_data
    else:
        return E_losc_dfa_tot, losc_eigs

def scf_losc(dfa_info, mf, losc_data=None, occ=None,
             orbital_energy_unit='eV', newton=False, verbose=5, window=None,
             return_losc_data=False, fitbasis='def2-qzvpp-ri'):
    """Perform the SCF-LOSC (frozen-LO) calculation based on a DFA 
    wavefunction

    Parameters
    ----------
    dfa_info : py_losc.DFAInfo
        The information of the parent DFA.
    mf : pyscf.dft.rks.RKS or pyscf.dft.uks.UKS
        Wavefunction from the parent DFA.
    orbital_energy_unit : {'eV', 'au'}, default='eV'
        The unit of orbital energies in the output.

        - 'eV': electronvolt.
        - 'au': atomic unit, hartree.

    newton : bool
        Whether turn on the PySCF buiild-in newton method to do the SCF 
        calculation or not.
    verbose : int, default=5
        The print level to control `post_scf_losc` and PySCF SCF calculation.
    window : [float, float], optional
        The orbital energy window in eV to select COs for LOSC localization.
        See `post_scf_losc`.

    Returns
    -------
    mlosc : pyscf.dft.rks.RKS or pyscf.dft.uks.UKS
        A PySCF wavefunction object that has LOSC contribution included.
    """
    #########################################################################
    # step 1: sanity-check (& cuszomize the occupation number if 'occ' in   #
    # mf.losc_data                                                          #
    #########################################################################
    _validate_inp(mf)
    if orbital_energy_unit not in ['eV', 'au']:
        raise Exception(
            'Invalid input for "orbital_energy_unit".'
        )
    if occ == None:
        occ = {}

    #########################################################################
    # step 2: do post-scf-losc to build curvature and LOs.                  #
    #########################################################################
    original_C_co = 1 * np.asarray(mf.mo_coeff) 
    # deep copy of the original canonical orbitals
    # (Nbasis, Nocc+Nvir) array for RKS, (2, Nbasis, Nocc+Nvir) for UKS.

    _, _, losc_data = post_scf_losc(
        dfa_info, mf, orbital_energy_unit=orbital_energy_unit, 
        verbose=verbose, occ=occ, return_losc_data=True, window=window,
        fitbasis=fitbasis
    )

    #########################################################################
    # step 3: perform SCF-LOSC.                                             #
    #########################################################################
    # see utils.generate_loscmf()
    j_x_separation = options.get_param('curvature', 'j_x_separation')
    loscmf = utils.generate_loscmf(mf, losc_data=losc_data, j_x_separation=j_x_separation)
    loscmf.init_guess = mf.chkfile
    loscmf.converged = False
    if newton == True:
        loscmf = loscmf.newton() 
    print('begin SCF-LOSC calculation.')
    loscmf.kernel()

    #########################################################################
    # step 4: pack LOSC data & print energies                               #
    #########################################################################
    eig_factor = 1.0 if orbital_energy_unit == 'au' else 27.21138602
    losc_data['losc_type'] = 'SCF-LOSC'
    losc_data['losc_energy'] = loscmf.e_tot - losc_data['dfa_energy']
    losc_data['losc_dfa_energy'] = loscmf.e_tot
    nspin = losc_data['nspin']
    if nspin == 1:
        losc_data['losc_dfa_orbital_energy'] = [loscmf.mo_energy * eig_factor]
    else: 
        losc_data['losc_dfa_orbital_energy'] = loscmf.mo_energy * eig_factor

    utils.print_total_energies(verbose, loscmf.mol, losc_data)
    utils.print_orbital_energies(verbose, loscmf, losc_data) 

    #########################################################################
    # step 5: update the U matrix & KS canonical orbital rotation           #
    # added by Ye Li                                                        #
    #########################################################################

    # step 5.1: update the U matrix
    ### The U matrix is the transformation matrix from the full canonical
    ##### orbitals to the full localized orbitals. 
    ##### U = C_lo.T @ S @ C_co
    ### This matrix will be used in LOSC-TD-DFT to construct corrections
    ##### to the Casida equation.
    nspin       =   losc_data['nspin']
    S           =   losc_data['S']      # overlap matrix, 
                                        # (Nbasis, Nbasis) array.
    new_C_co    =   np.asarray(loscmf.mo_coeff)   # new canonical orbitals, 
                                                  # (Nbasis, Nocc+Nvir) array for RKS, 
                                                  # (2, Nbasis, Nocc+Nvir) for UKS.
    C_lo        =   losc_data['C_lo']   # localized orbitals, 
                                        # [(Nbasis, N_selected) array] * nspin.
    selected_CO =   losc_data['select_CO_idx']  # selected COs for localization,
                                                # [(idx_start, idx_end)] * nspin.
    full_U      =   [None] * nspin      # list to store the full U matrix.
    if nspin == 1:
        if selected_CO[0] == None:
            full_U[0]    =   np.dot(np.dot(C_lo[0].T, S), new_C_co)
        else:
            Amat    =   1 * original_C_co   # deep copy of the original canonical orbitals,
                                            # now RKS, (Nbasis, Nocc+Nvir) array.
            idx_start, idx_end      =   selected_CO[0]
            ### update Amat as the full localized orbitals,
            ### Amat is a (Nbasis, N_LO) = (Nbasis, Nocc+Nvir) array.
            Amat[:,idx_start:idx_end]    =   1 * C_lo[0]   
            ### compute the new full U matrix,
            ### U is a (N_LO, N_CO) = (Nocc+Nvir, Nocc+Nvir) array.
            full_U[0]    =   np.dot(np.dot(Amat.T, S), new_C_co) 

    else:
        Amat_beta   =   1 * original_C_co[1]
        if selected_CO[0] == None:
            full_U[0]    =   np.dot(np.dot(C_lo[0].T, S), new_C_co[0])
        else:
            idx_start_alpha, idx_end_alpha    =   selected_CO[0]
            ### update Amat_alpha as the full localized orbitals,
            ### Amat_alpha is a (Nbasis, N_LO) = (Nbasis, Nocc+Nvir) array.
            Amat_alpha  =   1 * original_C_co[0]
            Amat_alpha[:,idx_start_alpha:idx_end_alpha]    =   1 * C_lo[0]
            ### compute the new full U matrix,
            ### U_alpha is a (N_LO, N_CO) = (Nocc+Nvir, Nocc+Nvir) array.
            full_U[0]    =   np.dot(np.dot(Amat_alpha.T, S), new_C_co[0])
        if selected_CO[1] == None:
            full_U[1]    =   np.dot(np.dot(C_lo[1].T, S), new_C_co[1])
        else:
            idx_start_beta, idx_end_beta      =   selected_CO[1]
            ### update Amat_beta as the full localized orbitals,
            ### Amat_beta is a (Nbasis, N_LO) = (Nbasis, Nocc+Nvir) array.
            Amat_beta[:,idx_start_beta:idx_end_beta]    =   1 * C_lo[1]
            ### compute the new full U matrix,
            ### U_beta is a (N_LO, N_CO) = (Nocc+Nvir, Nocc+Nvir) array.
            full_U[1]    =   np.dot(np.dot(Amat_beta.T, S), new_C_co[1])

    ### store the full U matrix in losc_data
    losc_data['full_U']  =   full_U

    # step 5.2: compute the canonical orbital rotation matrix V
    ### The sort of energy level could change during the SCF-LOSC calculation,
    ##### thus we need to find the correspondence between the old and new orbitals,
    ##### and this would be useful for matching the excitations in QE-DFT or TD-DFT.
    ### We will find the correspondence based on the orbital rotation from the
    ##### original canonical orbitals to the losc-corrected canonical orbitals.
    ##### V = new_C_co.T @ S @ original_C_co
    ### The orbital rotation matrix will be stored in losc_data. And it can be processed
    ##### in the following function orbital_correspondence().
    V           =   [None] * nspin
    if nspin == 1:
        V[0]    =   np.dot(np.dot(new_C_co.T, S), original_C_co)
    else:
        V[0]    =   np.dot(np.dot(new_C_co[0].T, S), original_C_co[0])
        V[1]    =   np.dot(np.dot(new_C_co[1].T, S), original_C_co[1])

    ### store the orbital rotation matrix in losc_data
    losc_data['CO_rotation']    =   V

    if return_losc_data:
        return loscmf, losc_data
    else:
        return loscmf 

def orbital_correspondence(Vmat, if_print=False):
    """Find the correspondence between the old and new orbitals based on the
    orbital rotation matrix V.
    """
    ### The orbital rotation matrix V is a (N_CO_new, N_CO_old) unitary array
    N_CO    =   Vmat.shape[0]
    correspondence  =   {}
    for i in range(N_CO):
        ### find the corresponding orbital index of the new orbitals
        idxes_new = np.argsort(np.abs(Vmat[:,i]))[::-1]
        ### store the correspondence in a dictionary
        correspondence[i+1]  =   []
        for k in range(3):
            idx_new = idxes_new[k]
            correspondence[i+1].append((idx_new+1, Vmat[idx_new, i]))
    if if_print:
        print('The correspondence between the old and new orbitals:')
        print('Old index -> new index, V value')
        for key in correspondence:
            print(f'Orbital {key}:')
            for idx, val in correspondence[key]:
                if abs(val) > 0.2:
                    print(f'{key} -> {idx}, {val}')

    inverse_correspondence  =   {}
    for i in range(N_CO):
        ### find the corresponding orbital index of the old orbitals
        idxes_old = np.argsort(np.abs(Vmat[i,:]))[::-1]
        ### store the correspondence in a dictionary
        inverse_correspondence[i+1]  =   []
        for k in range(3):
            idx_old = idxes_old[k]
            inverse_correspondence[i+1].append((idx_old+1, Vmat[i, idx_old]))
    return correspondence, inverse_correspondence


