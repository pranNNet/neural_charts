'''
Applies boundary conditions for the 1D Euler equation solver.

- Boundary conditions enforced via ghost cells
'''

import numpy as np

def apply_BC(Q, 
             BC = 'Zero-Gradient', 
             ghost_N = 3):
    '''
    :param ndarray Q: (3, N_cells) Conservative State Matrix
    :param int ghost_N: Number of ghost cells to add at each end
    :return ndarray Q_num: (3, N_cells+2*ghost_N) Extended Conservative State Matrix

    Extends the domain with boundary condition ghost cells

    Three BC Types:
    - Zero-Gradient: Q_ghost = Q_boundary
    - Reflective: rho_ghost = rho_boundary, mom_ghost = neg mom_boundary, E_ghost = E_boundary
    - Periodical: Q_ghost_left = Q_boundary_right, Q_ghost_right = Q_boundary_left
    '''

    Q_num = np.empty((3, Q.shape[1] + 2*ghost_N))
    Q_num[:, ghost_N:-ghost_N] = Q

    if BC == 'Zero-Gradient':
        Q_num[:, :ghost_N] = Q[:, 0][:, np.newaxis]
        Q_num[:, -ghost_N:] = Q[:, -1][:, np.newaxis]
    elif BC == 'Reflective':
        Q_num[:, :ghost_N] = Q[:, :ghost_N][:, ::-1]
        Q_num[1, :ghost_N] *= -1.0
        Q_num[:, -ghost_N:] = Q[:, -ghost_N:][:, ::-1]
        Q_num[1, -ghost_N:] *= -1.0
    elif BC == 'Periodical':
        Q_num[:, :ghost_N] = Q[:, -ghost_N:]
        Q_num[:, -ghost_N:] = Q[:, :ghost_N]
    else:
        raise ValueError('Invalid BC')
        
    return Q_num