'''
Sod validation for the ladder step 1 setup (run.py settings), against the exact Riemann solver.

1. Exact solver: Toro Table 4.3 star states (tests 1-5) and the sweep table in plan 1.1;
   exact_cell_averages vs brute-force quadrature of sample (r = 10, 16); the sonic fan at r = 16
2. FV (HLLC + WENO5Z + RK4) Sod, r = 10: time stamps, mass/energy conservation, L1 error of the
   stored 512-cell averages against exact cell averages at t = 0.16 and t = t_max
3. Grid convergence of the native-resolution L1 error at t = 0.16 (N = 512 .. 4096)
4. Figure sod_validation.png (stored FV vs exact cell averages)

Self-contained: checks the solver only, and depends on nothing in the preprocessing package.

Settings (states, grid, times, scheme) come from generation_config.yaml, as for the dataset.

Usage: python Sod_Shock_Tube_Validation.py [--config generation_config.yaml] [--quick] [--animate]
'''

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import run as R
from infrastructure.results import ExactRiemannSolver
from infrastructure.solver_config import prim_to_cons, cons_to_prim


T_CHECK = 0.16                  # time at which the FV error and the convergence are checked
CFG = None                      # generation config, loaded in __main__ (--config)


TORO = [  # left, right, p*, u*  (Toro, Table 4.3)
    ((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 0.30313, 0.92745),
    ((1.0, -2.0, 0.4), (1.0, 2.0, 0.4), 0.00189, 0.0),
    ((1.0, 0.0, 1000.0), (1.0, 0.0, 0.01), 460.894, 19.5975),
    ((1.0, 0.0, 0.01), (1.0, 0.0, 100.0), 46.0950, -6.19633),
    ((5.99924, 19.5975, 460.894), (5.99242, -6.19633, 46.0950), 1691.64, 8.68975),
]

PLAN_TABLE = {  # r: p*, u*, rho*_L, rho*_R, S, Mn, head, tail  (plan 1.1)
    2: (0.125, 0.172, 0.715, 0.147, 1.166, 1.10, -0.529, -0.323),
    4: (0.172, 0.426, 0.546, 0.183, 1.344, 1.27, -0.748, -0.237),
    10: (0.303, 0.927, 0.426, 0.266, 1.752, 1.66, -1.183, -0.070),
    16: (0.431, 1.278, 0.392, 0.326, 2.074, 1.96, -1.497, 0.037),
    24: (0.601, 1.645, 0.372, 0.386, 2.435, 2.30, -1.833, 0.141),
}


# ----------------------------------------------------------------------------------------------
# Exact reference, used here only to check the solver
# ----------------------------------------------------------------------------------------------

def to_prim(U, gamma):
    '''(N, 3) conservative -> primitive.'''
    return cons_to_prim(U.T, gamma).T


def exact_waves(sol):
    '''Wave speeds and side states for a left rarefaction, a contact and a right shock.'''
    g = sol.gamma
    a_star_L = sol.a_L * (sol.P_star / sol.P_L) ** ((g - 1.0) / (2.0 * g))
    S = sol.u_R + sol.a_R * np.sqrt((g + 1.0) / (2.0 * g) * (sol.P_star / sol.P_R) + (g - 1.0) / (2.0 * g))
    WL = np.array([sol.rho_L, sol.u_L, sol.P_L])
    WR = np.array([sol.rho_R, sol.u_R, sol.P_R])
    WsL = np.array([sol.rho_star_L, sol.u_star, sol.P_star])
    WsR = np.array([sol.rho_star_R, sol.u_star, sol.P_star])
    return {
        'head':    {'s': sol.u_L - sol.a_L, 'W_up': WL, 'W_down': WL},
        'tail':    {'s': sol.u_star - a_star_L, 'W_up': WsL, 'W_down': WsL},
        'contact': {'s': sol.u_star, 'W_up': WsL, 'W_down': WsR},
        'shock':   {'s': S, 'W_up': WR, 'W_down': WsR, 'Mn': (S - WR[1]) / sol.a_R},
    }


_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)


def exact_cell_averages(sol, edges, t):
    '''Exact conservative cell averages (T, N, 3): each cell is split at the waves inside it and
    each piece integrated with 8-point Gauss-Legendre (exact on constant pieces, smooth in the fan).'''
    w = exact_waves(sol)
    speeds = np.array([w[k]['s'] for k in ('head', 'tail', 'contact', 'shock')])
    N = len(edges) - 1
    out = np.empty((len(t), N, 3))
    for j, tj in enumerate(t):
        xw = sol.x_int + speeds * tj
        pts = np.unique(np.concatenate([edges, xw[(xw > edges[0]) & (xw < edges[-1])]]))
        a, b = pts[:-1], pts[1:]
        cell = np.searchsorted(edges, 0.5 * (a + b)) - 1
        xq = 0.5 * (a + b)[:, None] + 0.5 * (b - a)[:, None] * _GL_X[None, :]
        rho, u, P = sol.sample(xq.ravel(), tj)
        Uq = prim_to_cons(np.array([rho, u, P]), sol.gamma).reshape(3, *xq.shape)
        piece = np.einsum('cpq,q->pc', Uq, _GL_W) * (0.5 * (b - a))[:, None]
        acc = np.zeros((N, 3))
        np.add.at(acc, cell, piece)
        out[j] = acc / np.diff(edges)[:, None]
    return out


def check_exact():
    ok = True
    for i, (L, Rt, ps, us) in enumerate(TORO, 1):
        sol = ExactRiemannSolver(L, Rt, x_interface=0.5)
        rtol = 3e-3 if i == 2 else 1e-4          # test 2's p* is given to 3 significant figures only
        good = abs(sol.P_star - ps) < rtol * ps and abs(sol.u_star - us) < 1e-4 * max(abs(us), 1.0)
        ok &= good
        print(f'  Toro test {i}: p* = {sol.P_star:.6g} (ref {ps}), u* = {sol.u_star:.6g} (ref {us})  '
              f'{"ok" if good else "MISMATCH"}')
    for r, ref in PLAN_TABLE.items():
        sol = R.exact_solver(CFG, r)
        w = exact_waves(sol)
        got = (sol.P_star, sol.u_star, sol.rho_star_L, sol.rho_star_R, w['shock']['s'], w['shock']['Mn'],
               w['head']['s'], w['tail']['s'])
        err = max(abs(a - b) for a, b in zip(got, ref))
        good = err < 6e-3                       # table is rounded to 3 decimals (Mn to 2)
        ok &= good
        print(f'  plan table r = {r:2d}: max abs diff {err:.1e}  {"ok" if good else "MISMATCH"}')
    return ok


def check_cell_averages(n=10_000):
    '''
    exact_cell_averages vs n-point midpoint quadrature of sample in every cell, at t = T_CHECK.
    Cells without a jump must agree to 1e-10. In the contact and shock cells the midpoint rule itself
    is only first order (it can misplace up to half a sub-interval), so there the bound is 0.5 |jump| / n.
    '''
    ok = True
    t = T_CHECK
    e = CFG.x_edges(CFG.n_x)
    for r in (10.0, 16.0):
        sol = R.exact_solver(CFG, r)
        Ue = exact_cell_averages(sol, e, [t])[0]
        xm = e[:-1, None] + (np.arange(n) + 0.5)[None, :] / n * np.diff(e)[:, None]
        rho, u, P = sol.sample(xm.ravel(), t)
        Ub = prim_to_cons(np.array([rho, u, P]), sol.gamma).reshape(3, CFG.n_x, n).mean(axis=2).T
        d = np.abs(Ue - Ub)
        w = exact_waves(sol)
        jump = np.zeros(CFG.n_x)
        for k in ('contact', 'shock'):
            dU = prim_to_cons(w[k]['W_down'], sol.gamma) - prim_to_cons(w[k]['W_up'], sol.gamma)
            jump[np.searchsorted(e, sol.x_int + w[k]['s'] * t) - 1] += np.abs(dU).max()
        smooth = jump == 0
        good = d[smooth].max() < 1e-10 and np.all(d[~smooth].max(axis=1) <= 0.5 * jump[~smooth] / n + 1e-10)
        ok &= good
        print(f'  cell averages r = {r:2.0f}: max diff {d[smooth].max():.1e} away from jumps, '
              f'{d[~smooth].max():.1e} in the {(~smooth).sum()} jump cells (bound {0.5 * jump.max() / n:.1e})  '
              f'{"ok" if good else "MISMATCH"}')
    return ok


def check_sonic_fan(r=16.0, eps=1e-9):
    '''r = 16: the left fan straddles xi = 0. sample must be continuous at the head, the tail and xi = 0,
    and sonic (u = a = 2 a_L / (g + 1), since u_L = 0) at xi = 0.'''
    sol = R.exact_solver(CFG, r)
    w = exact_waves(sol)
    g, t = sol.gamma, T_CHECK
    xis = {'head': w['head']['s'], 'xi = 0': 0.0, 'tail': w['tail']['s']}
    ok = xis['head'] < 0.0 < xis['tail']
    for name, xi in xis.items():
        W = np.array(sol.sample(sol.x_int + t * np.array([xi - eps, xi + eps]), t))
        gap = np.abs(W[:, 1] - W[:, 0]).max()
        good = gap < 1e-6
        ok &= good
        print(f'  sonic fan r = {r:.0f}, {name} (xi = {xi:+.4f}): jump across +-{eps:.0e} is {gap:.1e}  '
              f'{"ok" if good else "MISMATCH"}')
    rho, u, P = (v[0] for v in sol.sample(np.array([sol.x_int]), t))
    a_sonic = 2.0 * sol.a_L / (g + 1.0)
    err = max(abs(u - a_sonic), abs(np.sqrt(g * P / rho) - a_sonic))
    good = err < 1e-12
    ok &= good
    print(f'  sonic fan r = {r:.0f}, at xi = 0: u = {u:.6f}, a = {np.sqrt(g * P / rho):.6f}, '
          f'2 a_L / (g + 1) = {a_sonic:.6f}  {"ok" if good else "MISMATCH"}')
    return ok


def l1(U, Ue):
    return np.abs(U - Ue).mean(axis=0)          # per conservative variable, mean over cells


def check_fv(quick=False):
    r = 10.0
    sol = R.exact_solver(CFG, r)
    t = CFG.t_snapshots()
    j_tr = int(round(T_CHECK / CFG.dt_snapshot))

    out = R.run_case(CFG, r)
    U_fine = out['U_steps'][out['store_idx']]          # the 121 states on the 0.002 grid
    del out['U_steps']
    U = R.coarsen(U_fine, CFG.coarsen)
    Ue = exact_cell_averages(sol, CFG.x_edges(CFG.n_x), t)
    dx = CFG.length / CFG.n_x
    tot = U.sum(axis=1) * dx
    drift = np.abs(tot[:, [0, 2]] - tot[0, [0, 2]]).max(axis=0) / tot[0, [0, 2]]
    print(f'  r = 10 on {CFG.n_fine} cells: dt = {out["dt"]:.3e} ({out["steps_per_snapshot"]} steps per snapshot), '
          f'CFL = {out["cfl"]:.3f}, {len(t)} snapshots')
    print(f'  mass / energy relative drift: {drift[0]:.1e} / {drift[1]:.1e}')
    print(f'  t = 0 matches exact cell averages: max diff {np.abs(U[0] - Ue[0]).max():.1e}')
    for j in (j_tr, len(t) - 1):
        print(f'  stored 512-cell L1 error (rho, m, E) at t = {t[j]:.2f}: {np.array2string(l1(U[j], Ue[j]), formatter={'float_kind': lambda v: f'{v:.2e}'})}')
    ok = drift.max() < 1e-12 and l1(U[j_tr], Ue[j_tr]).max() < 1e-3

    # Native-resolution grid convergence at T_CHECK
    Ns = [512, 1024, 2048] if quick else [512, 1024, 2048, 4096]
    errs = []
    for N in Ns:
        if N == CFG.n_fine:
            Uf = U_fine
        else:
            o = R.run_case(CFG, r, N=N)
            Uf = o['U_steps'][o['store_idx']]
        errs.append(l1(Uf[j_tr], exact_cell_averages(sol, CFG.x_edges(N), [t[j_tr]])[0])[0])
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    print(f'  density L1 at t = {t[j_tr]:.2f}, N = {Ns}: {np.array2string(np.array(errs), formatter={'float_kind': lambda v: f'{v:.2e}'})}, '
          f'observed order {np.array2string(rates, precision=2)}')
    plot(U, Ue, t, j_tr)
    return ok


def plot(U, Ue, t, j):
    x = CFG.x_centers(CFG.n_x)
    W, We = to_prim(U[j], CFG.gamma), to_prim(Ue[j], CFG.gamma)
    fig, axs = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    for k, (ax, name) in enumerate(zip(axs, [r'$\rho$', r'$v$', r'$p$'])):
        ax.plot(x, We[:, k], 'k-', lw=1.5, label='exact cell averages')
        ax.plot(x, W[:, k], 'r.', ms=2.5, label='FV (4096 -> 512)')
        ax.set_ylabel(name)
        ax.grid(alpha=0.4)
    axs[0].legend()
    axs[0].set_title(f'Sod (r = 10), t = {t[j]:.2f}: {CFG.flux} + {CFG.reconstruction} + RK4')
    axs[-1].set_xlabel('x')
    fig.tight_layout()
    fig.savefig('sod_validation.png', dpi=130)
    print('  wrote sod_validation.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=R.DEFAULT_CONFIG, help='generation config (YAML)')
    ap.add_argument('--quick', action='store_true', help='skip the 4096-cell convergence point')
    ap.add_argument('--animate', action='store_true', help="also show the original animate_comparison")
    args = ap.parse_args()
    CFG = R.load_config(args.config)
    if not args.animate:
        matplotlib.use('Agg')

    print('Exact solver')
    ok_exact = check_exact()
    ok_exact &= check_cell_averages()
    ok_exact &= check_sonic_fan()
    print('FV solver vs exact')
    ok_fv = check_fv(args.quick)
    print('PASS' if ok_exact and ok_fv else 'FAIL')

    if args.animate:
        from infrastructure.solver_config import EulerConfig, prim_to_cons
        from infrastructure.results import animate_comparison
        from euler1d.solver import EulerSolver
        left, right = CFG.states(10.0)
        N = CFG.n_x
        solver_cfg = EulerConfig(domain_size=CFG.length, N_cells=N, IC=R.initial_condition(CFG, left, right, N),
                                 QL=prim_to_cons(np.array(left)), QR=prim_to_cons(np.array(right)),
                                 x_split_percent=CFG.x0 / CFG.length, t_max=CFG.t_max, dt=0.4 / N / 1.9)
        animate_comparison(EulerSolver(solver_cfg), R.exact_solver(CFG, 10.0))
