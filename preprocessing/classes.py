'''
Class labels from entropy: class = 2 * origin + shocked, origin 0 = left gas, 1 = right gas.

Entropy s = ln(p / rho^gamma) is constant along particle paths and rises only through a shock, so a
cell's s tells where its gas came from (the initial level s_q just below it). A weak shock barely raises
s (by 5e-4 at r = 2), so a cell also counts as shocked when its pressure is above its gas's initial one:
in a Riemann problem only a shock compresses gas.
'''

import numpy as np

CLASS_TABLE = ['left', 'left_shocked', 'right', 'right_shocked']


def entropy(W, gamma):
    '''s = ln(p / rho^gamma) per cell of W [..., 3] = (rho, v, p).'''
    return np.log(W[..., 2] / W[..., 0] ** gamma)


def entropy_classes(W, flags, states, gamma, cfg):
    '''
    :param W: [T, N, 3] primitives
    :param flags: [T, N-1] jump flags; snapshots flagged -1 (t < t_min) get -1 here too
    :param states: [Q, 3] initial (rho, v, p) of each origin, [left, right]
    :param cfg: the classes section of the config
    :return: [T, N] int8 class per cell, -1 where ambiguous
    '''
    tol, tol_p, m = cfg['tol'], cfg['tol_p'], cfg['flag_margin']
    states = np.asarray(states, dtype=float)
    s = entropy(W, gamma)[..., None] - entropy(states, gamma)            # [T, N, Q]: s_cell - s_q
    below = s >= -tol                                                    # levels with s_q <= s_cell + tol
    origin = np.where(below, np.arange(len(states)), -1).max(axis=-1)    # the highest such level
    rise = np.take_along_axis(s, origin.clip(0)[..., None], axis=-1)[..., 0]
    p_q = states[origin.clip(0), 2]
    shocked = (rise > tol) | (W[..., 2] > (1 + tol_p) * p_q)
    classes = (2 * origin + shocked).astype(np.int8)

    near_two = (np.abs(s) <= tol).sum(axis=-1) >= 2
    # cells within m cells of a flagged face: face k touches cells k-m+1 .. k+m
    cs = np.pad(np.cumsum(flags == 1, axis=-1), ((0, 0), (1, 0)))
    c = np.arange(W.shape[1])
    near_flag = (cs[:, np.clip(c + m, 0, cs.shape[1] - 1)] - cs[:, np.clip(c - m, 0, cs.shape[1] - 1)]) > 0
    classes[(origin < 0) | near_two | near_flag | (flags[:, :1] < 0)] = -1
    return classes
