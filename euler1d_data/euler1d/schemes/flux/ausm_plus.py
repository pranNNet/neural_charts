'''
Primary References: 
- Liou, M.-S., "A sequel to AUSM: AUSM+," J. Comput. Phys., 129, pp. 364-382 (1996).
- https://github.com/fhermet/euler-1d-solver/blob/main/docs/04_flux_schemes.md
- https://github.com/fhermet/euler-1d-solver/blob/main/euler1d/schemes/flux/ausm_plus.py

'''
import numpy as np


def AUSMp(QL, QR, dx=1.0, gamma=1.4):
    def cons_to_prim(Q):
        rho = Q[0, :]
        u = Q[1, :] / rho  # Fixed: divide by density to get velocity
        P = (gamma - 1) * (Q[2, :] - 0.5 * rho * u**2)
        return np.array([rho, u, P])

    def M_split_plus(M):
        return np.where(
            np.abs(M) >= 1.0,
            0.5 * (M + np.abs(M)),
            0.25 * (M + 1.0) ** 2 + 0.125 * (M**2 - 1.0) ** 2,
        )

    def M_split_minus(M):
        return np.where(
            np.abs(M) >= 1.0,
            0.5 * (M - np.abs(M)),
            -0.25 * (M - 1.0) ** 2 - 0.125 * (M**2 - 1.0) ** 2,
        )

    def P_split_plus(M):
        return np.where(
            np.abs(M) >= 1.0,
            0.5 * (1.0 + np.sign(M)),
            0.25 * (M + 1.0) ** 2 * (2.0 - M) + 0.1875 * M * (M**2 - 1.0) ** 2,
        )

    def P_split_minus(M):
        return np.where(
            np.abs(M) >= 1.0,
            0.5 * (1.0 - np.sign(M)),
            0.25 * (M - 1.0) ** 2 * (2.0 + M) - 0.1875 * M * (M**2 - 1.0) ** 2,
        )

    rho_L, u_L, P_L = cons_to_prim(QL)
    rho_R, u_R, P_R = cons_to_prim(QR)

    # Total enthalpy
    H_L = (QL[2] + P_L) / rho_L
    H_R = (QR[2] + P_R) / rho_R

    # Wave speeds
    a_L = np.sqrt(gamma * P_L / rho_L)
    a_R = np.sqrt(gamma * P_R / rho_R)

    # Interface speed of sound (simple average)
    a_half = 0.5 * (a_L + a_R)

    # Left/right Mach numbers
    M_L = u_L / a_half
    M_R = u_R / a_half

    # Interface Mach number
    M_half = M_split_plus(M_L) + M_split_minus(M_R)

    # Mass flux (upwind on sign of M_half)
    m_dot = a_half * np.where(M_half >= 0.0, M_half * rho_L, M_half * rho_R)

    # Convective flux
    F = np.empty_like(QL)
    F[0] = m_dot
    F[1] = np.where(m_dot >= 0.0, m_dot * u_L, m_dot * u_R)
    F[2] = np.where(m_dot >= 0.0, m_dot * H_L, m_dot * H_R)

    # Pressure flux
    p_half = P_split_plus(M_L) * P_L + P_split_minus(M_R) * P_R
    F[1] += p_half

    return F