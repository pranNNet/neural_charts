'''
Training preprocessing: raw FV file -> training file. See README.md.

The only code used from the solver tree (../euler1d-data) is infrastructure.solver_config
(cons_to_prim, prim_to_cons) and infrastructure.results (ExactRiemannSolver); that folder is put
on the import path here so this package runs without installing anything.
'''

# import os
# import sys

# SOLVER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'euler1d_data')
# if SOLVER_DIR not in sys.path:
#     sys.path.append(SOLVER_DIR)
