'''
Exact cell averages for the Sod family (left rarefaction, contact, right shock). Evaluation only.
'''

import numpy as np

from euler1d_data.infrastructure.solver_config import prim_to_cons
from euler1d_data.infrastructure.results import ExactRiemannSolver

_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)


def wave_speeds(sol):
    '''Speeds of the rarefaction head and tail, the contact and the shock.'''
    if not (sol.P_R < sol.P_star <= sol.P_L):
        raise ValueError('expected a left rarefaction and a right shock')
    g = sol.gamma
    a_star_L = sol.a_L * (sol.P_star / sol.P_L) ** ((g - 1) / (2 * g))
    S = sol.u_R + sol.a_R * np.sqrt((g + 1) / (2 * g) * sol.P_star / sol.P_R + (g - 1) / (2 * g))
    return np.array([sol.u_L - sol.a_L, sol.u_star - a_star_L, sol.u_star, S])


def exact_cell_averages(left, right, x0, gamma, edges, t):
    '''
    :return: U [T, N, 3] conservative exact cell averages, and the 4 wave speeds.
        Each cell is split at the waves inside it and each piece integrated with 8-point Gauss-Legendre
        (exact on constant pieces, accurate in the smooth fan).
    '''
    sol = ExactRiemannSolver(tuple(left), tuple(right), x_interface=x0, gamma=gamma)
    speeds = wave_speeds(sol)
    N = len(edges) - 1
    U = np.empty((len(t), N, 3))
    for j, tj in enumerate(t):
        xw = x0 + speeds * tj
        pts = np.unique(np.concatenate([edges, xw[(xw > edges[0]) & (xw < edges[-1])]]))
        a, b = pts[:-1], pts[1:]
        cell = np.searchsorted(edges, 0.5 * (a + b)) - 1
        xq = 0.5 * (a + b)[:, None] + 0.5 * (b - a)[:, None] * _GL_X
        Uq = prim_to_cons(np.array(sol.sample(xq.ravel(), tj)), gamma).reshape(3, *xq.shape)
        piece = np.einsum('cpq,q->pc', Uq, _GL_W) * (0.5 * (b - a))[:, None]
        acc = np.zeros((N, 3))
        np.add.at(acc, cell, piece)
        U[j] = acc / np.diff(edges)[:, None]
    return U, speeds
