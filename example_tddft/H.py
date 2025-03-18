import pyscf
import pyscf_losc
import pyscf_losc.pyscf_losc_tddft as losc_tddft
from pyscf import dft, tdscf

# Create an H atom
mol = pyscf.gto.M(
            unit = 'A',
            atom =  """
                    H 0.000000 0.000000 0.000000
                    """,
            charge = 0,
            spin = 1,
            basis = 'aug-ccpv5z',
            symmetry = False,
            verbose = 3
            )

# Run a ground-state B3LYP calculation
mf = dft.UKS(mol)
mf.xc = 'B3LYP'
mf.kernel()


# Run a TD-B3LYP calculation
td = tdscf.TDDFT(mf)
td.nstates = 5
td.kernel()
td.analyze()


# Run a ground-state LOSC-B3LYP calculation, 
# Note that the we need to perform a LOSC calculation before a TD-LOSC calculation
loscmf, losc_data = pyscf_losc.scf_losc(
                        pyscf_losc.B3LYP,
                        mf,
                        window=[-30,10],
                        return_losc_data=True,
                        fitbasis='aug-ccpv5z-ri'
                        )


# Run a TD-LOSC-B3LYP calculation
# Unlike the ground-state case, 
## we do not need calculate TD-B3LYP before TD-LOSC-B3LYP.
losc_td = losc_tddft.TDHF_scfLOSC(loscmf, losc_data)
losc_td.nstates = 5
losc_td.kernel()
losc_td.analyze(parent_rep=True) 
# parent_rep=True will print the transition amplitude 
## in the parent CO representation.
# This is useful for comparing the excitation energies
## to the TD-B3LYP calculation.
