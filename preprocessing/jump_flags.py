'''
Jump flags: 1 at the one cell face where the FV data jumps, for each jump.

Generic face tests on primitives, with no wave order and no Sod logic, so they carry over to 2D
(apply them along each axis, with the velocity component along that axis).
'''

import numpy as np


def face_jumps(W, gamma):
    '''Normalized jumps across the faces of W [..., N, 3] = (rho, v, p): g_p, g_rho, g_v, each [..., N-1].'''
    rho, v, p = W[..., 0], W[..., 1], W[..., 2]
    c = np.sqrt(gamma * p / rho)
    g_p = np.abs(np.diff(p, axis=-1)) / np.minimum(p[..., 1:], p[..., :-1])
    g_rho = np.abs(np.diff(rho, axis=-1)) / np.minimum(rho[..., 1:], rho[..., :-1])
    g_v = np.diff(v, axis=-1) / (0.5 * (c[..., 1:] + c[..., :-1]))     # signed; < 0 is compression
    return g_p, g_rho, g_v


def one_per_band(mask, score):
    '''Keep only the highest-score face of each run of adjacent True faces (a smeared jump is one band).'''
    out = np.zeros_like(mask)
    ends = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    for a, b in zip(ends[::2], ends[1::2]):                            # band = faces a .. b-1
        out[a + np.argmax(score[a:b])] = True
    return out


def jump_flags(W, t, gamma, cfg):
    '''
    :param W: [T, N, 3] primitives on the snapshots t
    :param cfg: the jump_flags section of the config
    :return: [T, N-1] int8: 1 jump, 0 no jump, -1 ignored (t < t_min)
    '''
    g_p, g_rho, g_v = face_jumps(W, gamma)
    shock = (g_p > cfg['tau_s']) & (g_v < -cfg['tau_v'])               # pressure jump with compression
    contact = (g_rho > cfg['tau_c']) & (g_p < cfg['tau_c_p'])          # density jump at nearly constant pressure
    flags = np.zeros(g_p.shape, np.int8)
    for j in range(len(t)):
        flags[j] = one_per_band(shock[j], g_p[j]) | one_per_band(contact[j], g_rho[j])
    flags[t < cfg['t_min'] - 1e-12] = -1
    return flags
