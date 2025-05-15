Fork Updates
------------

This is a fork of the repository `Yang Laboratory/losc <https://github.com/Yang-Laboratory/losc>`_, maintained by Ye Li, a graduate student in `Chen Li's Research Group <https://www.chem.pku.edu.cn/chenli/>`_ at Peking University.
Email me (pkupkupku@pku.edu.cn) if you have any question.

Updates in This Fork:
=====================
2025-05-10:
===========

- Now one can solve losc-tddft/tda equations by direct diagonalization.

2025-04-23:
===========

- Now one can directly apply solvent model to LOSC (TD)SCF objects in PySCF.

2025-03-18:
===========

- Implemented LOSC for linear-response TDDFT calculations.
- Added LOSC1 and LOSC2 curvatures for range-separated functionals (further improvement is required for enhanced accuracy).
- Resolved several issues, including:
  
  - Incompatibility of SCF-LOSC with PySCF version 2.5.0 or higher.
  - The :math:`\tau` parameter modification for Curvature2 not being effective.

Notes:
=======

- Currently, all the above changes apply only to the PySCF module; corresponding updates for the Psi4 module are still pending.
- This fork does not include the FMOL analysis released on July 4, 2024, in the original repository.
- Please refer to the ``example_tddft`` folder for examples of using LOSC in LR-TDDFT calculations.

References for Updates in This Fork:
====================================

- Li, Y.; Li, C. Submitted to *J. Chem. Theory Comput.*.



For installation and usage of ground-state LOSC, refer to the original README below.

Localized Orbital Scaling Correction (LOSC)
-------------------------------------------

LOSC is a newly developed method in `Weitao Yang's Research group
<https://yanglab.chem.duke.edu>`_ to solve the delocalization error
for the density functional theory (DFT) in quantum chemistry.

What does this Project do?
==========================

- This project provides libraries with interfaces for popular programming
  languages in quantum chemistry, including C, C++ and Python,
  for the developers who would be interested to implement the LOSC method
  in their favorite quantum chemistry packages. These libraries
  provide the functionalities to perform the essential calculations of
  the LOSC method, such as constructing the LOs, LOSC curvature matrix,
  and LOSC corrections.

- This project provides the implementation of LOSC method in an open-source
  quantum chemistry package, `psi4 <https://psicode.org>`_, with its
  Python interface. If you are a user of psi4, you can directly use this
  library with psi4 to perform LOSC calculations.

Manual and Documentation
========================

The manual and documentation are available at https://yang-laboratory.github.io/losc/.
Refer to the website for the instructions of installation and usage.


References of LOSC
==================

- Li, C.; Zheng, X.; Su, N. Q.; Yang, W. Localized Orbital Scaling
  Correction for System- atic Elimination of Delocalization Error in
  Density Functional Approximations.
  `Natl. Sci. Rev. 2018, 5, 203−215. 203-215.
  <https://doi.org/10.1093/nsr/nwx111>`_

- Su, N. Q.; Mahler, A.; Yang, W. Preserving Symmetry and
  Degeneracy in the Localized Orbital Scaling Correction Approach. J.
  `Phys. Chem. Lett. 2020, 11, 1528−1535.
  <https://doi.org/10.1021/acs.jpclett.9b03888>`_

- Mei, Y.; Chen, Z.; Yang, W.
  Self-Consistent Calculation of the Localized Orbital Scaling Correction
  for Correct Electron Densities and Energy-Level Alignments in Density
  Functional Theory.
  `J. Phys. Chem. Lett. 2020, 11, 23, 10269–10277.
  <https://doi.org/10.1021/acs.jpclett.0c03133>`_
