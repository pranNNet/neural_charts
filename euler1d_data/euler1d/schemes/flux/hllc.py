'''
Primary References: 
- https://edanya.uma.es/NSPDE/images/ef%20toro/2-HLLHLLC.pdf
- https://github.com/fhermet/euler-1d-solver/blob/main/docs/04_flux_schemes.md
- https://github.com/fhermet/euler-1d-solver/blob/main/euler1d/schemes/flux/hllc.py

'''
import numpy as np


def HLLC(QL, QR, dx = 1.0, gamma = 1.4):
    '''
    HLLC Scheme to resolve the interface Reimann problem and determine flux

    :param nd.array Q_L: Reconstructed left state at interface (3, N + 1)
    :param nd.array Q_R: Reconstructed right state at interface (3, N + 1)
    :param float dx: Grid spacing, default = 1.0
    :param float gamma: Specific heat ratio, default (air) = 1.4

    :return nd.array F: Flux at interfaces (3, N + 1)
    '''
    def cons_to_prim(Q):
        '''
        Convert conservative Q vector to primitive R vector
        '''
        rho = Q[0, :]
        u = Q[1, :]/rho
        P = (gamma - 1) * (Q[2, :] - 0.5 * rho * u**2) #Equation of State
        return np.array([rho, u, P])
    def compute_flux(Q):
        '''
        Compute flux from conservative Q
        '''
        rho, u, P = cons_to_prim(Q)
        return np.array([rho*u, rho*u**2 + P, u*(Q[2] + P)])

    rho_L, u_L, P_L  = cons_to_prim(QL)
    rho_R, u_R, P_R = cons_to_prim(QR)

    #left/right fluxes
    FL = compute_flux(QL)
    FR = compute_flux(QR)

    #wave speeds
    
    a_L = np.sqrt(gamma*P_L/rho_L)
    a_R = np.sqrt(gamma*P_R/rho_R)

    S_L = np.minimum(u_L - a_L, u_R - a_R)
    S_R = np.maximum(u_L + a_L, u_R + a_R)

    S_star = ((P_R - P_L + rho_L*u_L * (S_L - u_L) - rho_R*u_R * (S_R - u_R))
              /(rho_L*(S_L - u_L) - rho_R*(S_R - u_R)))

    #Intermediate States
    QL_star = rho_L * (S_L - u_L)/(S_L - S_star)*np.array([np.ones_like(S_star),
                                                           S_star,
                                                           QL[2]/rho_L + (S_star - u_L)*(S_star + P_L/(rho_L*(S_L-u_L)))])
    QR_star = rho_R * (S_R - u_R)/(S_R - S_star)*np.array([np.ones_like(S_star),
                                                           S_star,
                                                           QR[2]/rho_R + (S_star - u_R)*(S_star + P_R/(rho_R*(S_R-u_R)))])

    #wave cases (wave regions)
        #Flux frfr
    F = np.empty_like(QL)

    #left undisturbed
    mask = S_L >= 0
    F[:, mask] = FL[:, mask]
    #right undisturbed
    mask = S_R <= 0
    F[:, mask] = FR[:, mask]
    #left-star (contact disturbed)
    mask = (S_L < 0) & (S_star >= 0)
    F[:, mask] = FL[:, mask] + S_L[mask]*(QL_star[:, mask] - QL[:, mask])
    #right-star (contact disturbed)
    mask = (S_R > 0) & (S_star < 0)
    F[:, mask] = FR[:, mask] + S_R[mask]*(QR_star[:, mask] - QR[:, mask])


    return F 
