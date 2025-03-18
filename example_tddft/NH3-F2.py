import pyscf
import pyscf_losc
import pyscf_losc.pyscf_losc_tddft as losc_tddft
from pyscf import dft, tdscf
import os
import sys

def get_NH3F2_complex(R, Runit='Bohr', basis='aug-ccpvtz', symmetry=False):
    if Runit == 'Bohr':
        R_perA = R * 0.529177
    elif Runit == 'A':
        R_perA = R
    else:
        print("Invalid unit of R")
        return None
    mol = pyscf.gto.M(
            unit = 'A',
            atom = f"""
                    N 0.000000 0.000000 0.000000
                    H 0.000000 0.939052 -0.378892
                    H -0.813243 -0.469526 -0.378892
                    H 0.813243 -0.469526 -0.378892
                    F 0.000000 0.000000 {R_perA}
                    F 0.000000 0.000000 {R_perA + 1.396376}
                    """,
            charge = 0,
            spin = 0,
            basis = basis,
            symmetry = symmetry,
            verbose = 3
            )
    return mol


def TDLOSC(mol, functional, nstates=20):

    mf = dft.RKS(mol)
    mf.xc = functional
    mf.level_shift = 0.1
    mf.kernel()

    td = tdscf.TDDFT(mf)
    td.nstates = nstates
    td.level_shift = 0.1
    print('#'*50)
    print(f'TD-{functional}')
    td.singlet = True
    td.kernel()
    td.analyze()
    print('#'*50)

    tda = tdscf.TDA(mf)
    tda.nstates = nstates
    print('#'*50)
    print(f'TDA-{functional}')
    tda.singlet = True
    tda.kernel()
    tda.analyze()
    print('#'*50)

    pyscf_losc.options.set_param('localizer', 'max_iter', max_iter)
    pyscf_losc.options.set_param('curvature', 'version', cversion)
    loscmf, losc_data  =   pyscf_losc.scf_losc(
                            getattr(pyscf_losc, 
                                    losc_correction),
                            mf,
                            window=energy_window,
                            return_losc_data=True,
                            fitbasis=fitbasis
                            )


    td = losc_tddft.TDHF_scfLOSC_R(loscmf, losc_data)
    td.nstates = nstates
    td.level_shift = 0.1
    print('#'*50)
    print(f'TD-LOSC-{functional}')
    td.singlet = True
    td.kernel()
    td.analyze(parent_rep=True)
    print('#'*50)


    tda = losc_tddft.TDA_scfLOSC_R(loscmf, losc_data)
    tda.nstates = nstates
    print('#'*50)
    print(f'TDA-LOSC-{functional}')
    tda.singlet = True
    tda.kernel()
    tda.analyze(parent_rep=True)
    print('#'*50)
    return None


if __name__ == "__main__":
    path            =   os.path.dirname(os.path.abspath(__file__))
    output_name     =   os.path.join(path, "NH3-F2.out")

    basis           =   'aug-ccpvtz'
    fitbasis        =   'aug-ccpvtz-ri'
    functional      =   'B3LYP'
    losc_correction =   'B3LYP'
    max_iter        =   2000
    cversion        =   2
    energy_window   =   [-30,10]

    with open(output_name, 'w') as f:
        sys.stdout = f

        R_list  =  [5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 20.0, 
                    25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0]
        
        print('The NH3-F2 complex')

        for R in R_list:
            for i in range(6):
                print("")
            print(f"R = {R} Bohr")

            mol = get_NH3F2_complex(R, 
                                    Runit='Bohr', 
                                    basis=basis, 
                                    symmetry=False)
            TDLOSC(mol, functional, nstates=16)

            for i in range(6):
                print("")