'''
Primary References: 
- https://academicweb.nd.edu/~yzhang10/WENO_ENO.pdf
- https://github.com/fhermet/euler-1d-solver/blob/main/docs/05_reconstruction.md
- https://github.com/fhermet/euler-1d-solver/blob/main/euler1d/schemes/reconstruction/weno5.py

'''
import numpy as np


def WENO5(Q, dx=1.0):
    def Smooth_Ind(W_im2, W_im1, W_i, W_ip1, W_ip2):
        beta0 = 13/12 * (W_i - 2 * W_ip1 + W_ip2)**2 + 1/4 * (3*W_i - 4*W_ip1 + W_ip2)**2
        beta1 = 13/12 * (W_im1 - 2 * W_i + W_ip1)**2 + 1/4 * (W_im1 - W_ip1)**2
        beta2 = 13/12 * (W_im2 - 2 * W_im1 + W_i)**2 + 1/4 * (W_im2 - 4*W_im1 + 3*W_i)**2
        return (beta0, beta1, beta2)

    def JS_weights(d_ks, betas, dx=dx):
        beta_k = np.array(betas)
        d_k = np.array(d_ks)

        eps = 1e-6 * dx**2
        alpha_k = np.array([d_k[i] / (eps + beta_k[i])**2 for i in [0, 1, 2]])
        # Fixed: Sum over stencil axis (axis=0) with keepdims=True
        omega_k = alpha_k / np.sum(alpha_k, axis=0, keepdims=True)
        return omega_k

    n_cell = Q.shape[1] - 6

    dLs = (3/10, 3/5, 1/10)
    dRs = (1/10, 3/5, 3/10)

    # Stencils for left and right reconstruction
    QL_im2 = Q[:, 0:n_cell+1]
    QL_im1 = Q[:, 1:n_cell+2]
    QL_i   = Q[:, 2:n_cell+3]
    QL_ip1 = Q[:, 3:n_cell+4]
    QL_ip2 = Q[:, 4:n_cell+5]

    QR_im2 = Q[:, 1:n_cell+2]
    QR_im1 = Q[:, 2:n_cell+3]
    QR_i   = Q[:, 3:n_cell+4]
    QR_ip1 = Q[:, 4:n_cell+5]
    QR_ip2 = Q[:, 5:n_cell+6]

    # Stencil weights
    betaLs = Smooth_Ind(QL_im2, QL_im1, QL_i, QL_ip1, QL_ip2)
    omegaL = JS_weights(dLs, betaLs, dx)

    betaRs = Smooth_Ind(QR_im2, QR_im1, QR_i, QR_ip1, QR_ip2)
    omegaR = JS_weights(dRs, betaRs, dx)

    # Candidate polynomials
    pL0 =  1/3 * QL_i   + 5/6 * QL_ip1 - 1/6 * QL_ip2
    pL1 = -1/6 * QL_im1 + 5/6 * QL_i   + 1/3 * QL_ip1
    pL2 =  1/3 * QL_im2 - 7/6 * QL_im1 + 11/6 * QL_i
    pL = np.array([pL0, pL1, pL2])

    pR0 = 11/6 * QR_i   - 7/6 * QR_ip1 + 1/3 * QR_ip2
    pR1 =  1/3 * QR_im1 + 5/6 * QR_i   - 1/6 * QR_ip1
    pR2 = -1/6 * QR_im2 + 5/6 * QR_im1 + 1/3 * QR_i
    pR = np.array([pR0, pR1, pR2])

    # Reconstructed states
    QL = np.sum(omegaL * pL, axis=0)
    QR = np.sum(omegaR * pR, axis=0)

    # Positivity preservation
    QL[0] = np.maximum(1e-10, QL[0])
    QR[0] = np.maximum(1e-10, QR[0])

    return (QL, QR)