'''
1D Euler Equation Solver

- Two Flux Reconstruction Methods
    - Kareef: AUSM + & WENO 5
    - Pranet: HLLC & WENO 5Z

Specify simulation parameters via ...
Specify solver parameters via...
'''

import numpy as np
from euler1d.boundary import apply_BC
from infrastructure.solver_config import EulerConfig
from infrastructure.results import EulerResults
from datetime import datetime

from euler1d.schemes.flux.hllc import HLLC
from euler1d.schemes.flux.ausm_plus import AUSMp

from euler1d.schemes.reconstruction.weno5Z import WENO5Z
from euler1d.schemes.reconstruction.weno5 import WENO5

 
def EulerSolver(config: EulerConfig,
                flux = 'HLLC',
                reconstruction = 'WENO5Z'
                ):
    '''
    :param EulerConfig config: Configuration for the problem
    :param str flux: Flux Scheme (WENO5Z, WENO5)
    :param str reconstruction: Reconstruction Method (AUSM+, HLLC)

    :return EulerResults: Results of the simulation
    

    Main function to run the 1D Euler simulation
        - integrates RK4 for time stepping
        - uses specified flux and reconstruction schemes

    '''
    #unpack config 
    domain_size = config.domain_size
    N_cells = config.N_cells
    dx = config.dx
    t_max = config.t_max
    dt = config.dt
    gamma = config.gamma

    Q_init = config.IC
    BC = config.BC
    N_ghost = config.N_ghost

    #Select schemes based on user input
    #TODO: Modify with name of kareef's schemes
    if flux == 'HLLC':
        F_scheme = HLLC
    elif flux == 'AUSM+' or flux == 'AUSMp':
        F_scheme = AUSMp
    else:
        raise ValueError("Invalid Flux Scheme")

    if reconstruction == 'WENO5Z':
        R_scheme = WENO5Z
    elif reconstruction == 'WENO5':
        R_scheme = WENO5
    else:
        raise ValueError("Invalid Reconstruction Scheme")


    def calc_Flux(Q_curr):
        Q_num = apply_BC(Q = Q_curr,
                         BC = BC,
                         ghost_N = N_ghost)

        Q_L, Q_R = R_scheme(Q_num, dx = dx)
        F = F_scheme(Q_L, Q_R, dx = dx, gamma = gamma)
        return F

    # RHS of discretized 1D Euler
    def calc_discrete_Q_dot(Q_set, dx = dx):
        F = calc_Flux(Q_set)
        return -1/dx * (F[:, 1:] - F[:, :-1])
    
    t_hist = [0]
    Q_hist = [Q_init]
    F_hist = [calc_Flux(Q_init)]

    t_curr = 0  # CHANGED: was dt, which labelled every stored state one step late
    iteration_count = 0
    start_time = datetime.now()
    while t_max - t_curr > 1e-9 * dt:  # CHANGED: tolerance so float round-off cannot add a sliver step
        if (t_curr + dt) > t_max:
            dt = t_max - t_curr

        Q_curr = Q_hist[-1]

        #RK 4 Implementation 
            # Note that discrete Q_dot is independent of time
        k1 = dt*calc_discrete_Q_dot(Q_curr)
        k2 = dt*calc_discrete_Q_dot(Q_curr+k1/2)
        k3 = dt*calc_discrete_Q_dot(Q_curr+k2/2)
        k4 = dt*calc_discrete_Q_dot(Q_curr+k3)

        # Update next value of y
        Q_next = Q_curr + (1/6)*(k1 + 2 * k2 + 2 * k3 + k4)

        Q_hist.append(Q_next)
        t_curr += dt
        t_hist.append(t_curr)
        F = calc_Flux(Q_next)
        F_hist.append(F)
        # if iteration_count > 1:
        #     raise KeyError  

            # Progress Monitoring: Increment counter and log every N iterations
        iteration_count += 1
        if iteration_count % 10 == 0:
            current_time = datetime.now()
            elapsed_seconds = (current_time - start_time).total_seconds()
            print(f"Progress: {iteration_count} | iterations: ({elapsed_seconds:.3f}s) | Sim Time: ({t_curr:.6f}s) ")

    print(f"Simulation Complete")
    Results = EulerResults(Q_hist,
                           F_hist,
                           t_hist,
                           config)

    return Results


    