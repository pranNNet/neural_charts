'''
Single-case runner for the raw dataset: one Sod shock tube for a given pressure ratio r.

All settings come from generation_config.yaml (see load_config):
- left state (rho_L, v_L, r * p_R), right state (rho_R, v_R, p_R), diaphragm at x0
- solved on n_fine cells with the euler1d solver up to t_max, keeping every solver step;
  dt divides dt_snapshot exactly, so the snapshots t = 0, dt_snapshot, ... are solver steps (store_idx)
- coarsen() averages groups of fine cells into the saved grid (done in data_generation.py)

Arrays here are (T, N, 3) with the last axis (rho, rho*v, E).
'''

import io
import os
from pathlib import Path
import contextlib
from dataclasses import dataclass

import numpy as np
import yaml

from infrastructure.solver_config import EulerConfig, prim_to_cons
from infrastructure.physics_config import riemann_speeds
from infrastructure.results import ExactRiemannSolver
from euler1d.solver import EulerSolver


DEFAULT_CONFIG = Path(__file__).parent.parent / 'configs' / 'generation_config.yaml'

@dataclass(frozen=True)
class GenerationConfig:
    '''generation_config.yaml, flattened, plus derived quantities.'''
    raw: dict                    # the YAML as loaded
    text: str                    # the YAML text, stored in the HDF5 file for provenance
    gamma: float
    length: float
    x0: float
    left_rho: float
    left_v: float
    right: tuple                 # (rho, v, p)
    n_fine: int
    coarsen: int
    t_max: float
    dt_snapshot: float
    flux: str
    reconstruction: str
    cfl: float
    bc: str
    n_ghost: int
    cases: list                  # [{'split', 'n', 'r_min', 'r_max', 'seed'}]
    output_path: str
    gzip_level: int
    workers: object              # int, or None for all cores

    @property
    def n_x(self):
        return self.n_fine // self.coarsen

    @property
    def n_snapshots(self):
        return int(round(self.t_max / self.dt_snapshot)) + 1

    def states(self, r):
        '''Primitive left and right states for pressure ratio r.'''
        return (self.left_rho, self.left_v, self.right[2] * r), self.right

    def x_edges(self, N):
        return np.linspace(0.0, self.length, N + 1)

    def x_centers(self, N):
        e = self.x_edges(N)
        return 0.5 * (e[1:] + e[:-1])

    def t_snapshots(self):
        return self.dt_snapshot * np.arange(self.n_snapshots)


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        text = f.read()
    c = yaml.safe_load(text)
    p, g, t, s, o, e = c['physics'], c['grid'], c['time'], c['scheme'], c['output'], c['execution']
    cfg = GenerationConfig(
        raw=c, text=text,
        gamma=float(p['gamma']), length=float(p['length']), x0=float(p['x0']),
        left_rho=float(p['left']['rho']), left_v=float(p['left']['v']),
        right=(float(p['right']['rho']), float(p['right']['v']), float(p['right']['p'])),
        n_fine=int(g['n_fine']), coarsen=int(g['coarsen']),
        t_max=float(t['t_max']), dt_snapshot=float(t['dt_snapshot']),
        flux=s['flux'], reconstruction=s['reconstruction'], cfl=float(s['cfl']),
        bc=s['bc'], n_ghost=int(s['n_ghost']),
        cases=list(c['cases']), output_path=o['path'], gzip_level=int(o['gzip_level']),
        workers=None if e['workers'] is None else int(e['workers']))
    if cfg.n_fine % cfg.coarsen:
        raise ValueError('grid.n_fine must be a multiple of grid.coarsen')
    if abs((cfg.n_snapshots - 1) * cfg.dt_snapshot - cfg.t_max) > 1e-12:
        raise ValueError('time.t_max must be a multiple of time.dt_snapshot')
    return cfg


def initial_condition(cfg, left, right, N):
    '''(3, N) conservative cell averages of the step at x0; the cell that holds x0 gets the mixed average.'''
    e = cfg.x_edges(N)
    frac_left = np.clip((cfg.x0 - e[:-1]) / (e[1:] - e[:-1]), 0.0, 1.0)   # fraction of each cell left of x0
    UL, UR = prim_to_cons(np.array(left), cfg.gamma), prim_to_cons(np.array(right), cfg.gamma)
    return UL[:, None] * frac_left + UR[:, None] * (1.0 - frac_left)


def coarsen(U, factor):
    '''Conservative coarsening: mean of `factor` fine cells. U is (T, N, 3).'''
    T, N, C = U.shape
    return U.reshape(T, N // factor, factor, C).mean(axis=2)


def exact_solver(cfg, r):
    '''Exact solution of case r (used here only for the wave speeds that set the time step).'''
    left, right = cfg.states(r)
    return ExactRiemannSolver(left, right, x_interface=cfg.x0, gamma=cfg.gamma)


def run_case(cfg, r, N=None, verbose=False):
    '''
    Run one case on N cells (default grid.n_fine), keeping every solver step.

    :return dict: t_steps [S+1], U_steps [S+1, N, 3] float64 (every step, fine grid),
                  store_idx (steps at the snapshot times), dt, steps_per_snapshot, cfl (from exact-state speeds)
    '''
    N = N or cfg.n_fine
    left, right = cfg.states(r)
    dx = cfg.length / N
    # Fixed step that divides the snapshot interval exactly, so the snapshots are the solver's own steps
    _, _, s_max = riemann_speeds(exact_solver(cfg, r))
    steps_per_snapshot = int(np.ceil(cfg.dt_snapshot * s_max / (cfg.cfl * dx)))
    dt = cfg.dt_snapshot / steps_per_snapshot

    config = EulerConfig(domain_size=cfg.length,
                         N_cells=N,
                         IC=initial_condition(cfg, left, right, N),
                         QL=prim_to_cons(np.array(left), cfg.gamma),
                         QR=prim_to_cons(np.array(right), cfg.gamma),
                         x_split_percent=cfg.x0 / cfg.length,
                         BC=cfg.bc,
                         N_ghost=cfg.n_ghost,
                         gamma=cfg.gamma,
                         t_max=cfg.t_max,
                         dt=dt)

    sink = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with sink:
        results = EulerSolver(config, flux=cfg.flux, reconstruction=cfg.reconstruction)

    t_steps = np.array(results.t_hist)
    store_idx = steps_per_snapshot * np.arange(cfg.n_snapshots)
    if len(t_steps) != store_idx[-1] + 1 or np.max(np.abs(t_steps[store_idx] - cfg.t_snapshots())) > 1e-9:
        raise RuntimeError(f'r={r}: solver steps do not land on the snapshot times ({len(t_steps)} states)')
    U_steps = np.stack([Q.T for Q in results.Q_hist])          # (S+1, N, 3)
    return {'t_steps': t_steps, 'U_steps': U_steps, 'store_idx': store_idx, 'dt': dt,
            'steps_per_snapshot': steps_per_snapshot, 'cfl': s_max * dt / dx}


if __name__ == '__main__':
    import time
    import argparse
    ap = argparse.ArgumentParser(description='Run and time one case.')
    ap.add_argument('--config', default=DEFAULT_CONFIG)
    ap.add_argument('--r', type=float, default=10.0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    t0 = time.time()
    out = run_case(cfg, args.r)
    print(f"r = {args.r}: dt = {out['dt']:.3e}, {out['steps_per_snapshot']} steps per snapshot, "
          f"CFL = {out['cfl']:.3f}, {len(out['t_steps'])} states, {time.time() - t0:.1f} s")
