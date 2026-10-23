'''
Primary References: 
- https://academicweb.nd.edu/~yzhang10/WENO_ENO.pdf
- https://github.com/fhermet/euler-1d-solver/blob/main/docs/05_reconstruction.md
- https://github.com/fhermet/euler-1d-solver/blob/main/euler1d/schemes/reconstruction/weno5.py

'''
import numpy as np


def WENO5Z(Q, dx = 1.0):
    '''
    Function that takes in current state and outputs left & right states for each interface 

    :param ndarray Q:  Conservative state matrix (3, N + 6) 
    :param float dx: Grid spacing, default = 1.0

   
    :return nd.array Q_L: Reconstructed left state at interface (3, N + 1)
    :return nd.array Q_R: Reconstructed right state at interface (3, N + 1)
    '''

    def Smooth_Ind(W_im2, W_im1, W_i, W_ip1, W_ip2):
        '''
        Calculates Smoothness Indicator (Beta0, Beta1, Beta2) for given stencil 
        '''
        beta0 = 13/12 * (W_i - 2 * W_ip1 + W_ip2)**2 \
                + 1/4 * (3*W_i - 4*W_ip1 + W_ip2)**2
        
        beta1 = 13/12 * (W_im1 - 2 *W_i + W_ip1)**2 \
                + 1/4 * (W_im1 - W_ip1)**2
        beta2 = 13/12 * (W_im2 - 2 * W_im1 + W_i)**2 \
                + 1/4 * (W_im2 - 4*W_im1 + 3*W_i)**2
        return (beta0, beta1, beta2)
    
    def Z_weights(d_ks, betas, dx = dx):
        '''
        Calculates Z-weights for given stencil 
        '''
        beta_k = np.array(betas)
        d_k = np.array(d_ks)
        tau_s = np.abs(beta_k[0] - beta_k[2])
        eps = dx**2
        # alpha_k = d_k * (1 + tau_s/(beta_k + eps))
        # omega_k = alpha_k/(np.sum(alpha_k))
        alpha_k = np.array([d_k[i] * (1 + tau_s/(beta_k[i] + eps)) for i in [0, 1, 2]])
        omega_k = alpha_k/(np.sum(alpha_k, axis=0, keepdims=True))

        return omega_k


    n_int = Q.shape[1] - 5
    n_cell = Q.shape[1] - 6

    dLs = (3/10, 3/5, 1/10)
    dRs = (1/10, 3/5, 3/10)

    #stencils for L(eft) and R(ight)
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

    #Weighting for Left Stencil
    betaLs = Smooth_Ind(QL_im2, QL_im1, QL_i, QL_ip1, QL_ip2)
    omegaL = Z_weights(dLs, betaLs, dx)

    #Weighting for Right Stencil
    betaRs = Smooth_Ind(QR_im2, QR_im1, QR_i, QR_ip1, QR_ip2)
    omegaR = Z_weights(dRs, betaRs, dx)

    #Polynomial Reconstruction
    #Left
    p0 = 1/3 * QL_i + 5/6 * QL_ip1 - 1/6 * QL_ip2
    p1 = -1/6 * QL_im1 + 5/6 * QL_i + 1/3 * QL_ip1
    p2 = 1/3 * QL_im2 - 7/6 * QL_im1 + 11/6 * QL_i
    pL = np.array([p0, p1, p2])


    #Polynomial Reconstruction
    #Right
    p0 = 11/6 * QR_i - 7/6 * QR_ip1 + 1/3 * QR_ip2
    p1 = 1/3 * QR_im1 + 5/6 * QR_i - 1/6 * QR_ip1
    p2 = -1/6 * QR_im2 + 5/6 * QR_im1 + 1/3 * QR_i
    pR = np.array([p0, p1, p2])

    #Weighted Reconstruction

    QL = np.sum(omegaL*pL, axis = 0)
    QR = np.sum(omegaR*pR, axis = 0)      

    #clamp to positivy? 
    QL[0] = np.maximum(1e-10, QL[0])
#     QL[1] = np.maximum(1e-10, QL[1])

    QR[0] = np.maximum(1e-10, QR[0])
#     QR[1] = np.maximum(1e-10, QR[1])


    return (QL, QR)


  
