'''
Generates the raw FV data for ladder step 1 (Sod pressure-ratio sweep), plan sections 1.2 to 1.5.

- All settings (states, grid, times, scheme, the values of r per split, output path) come from
  generation_config.yaml; a copy of the YAML is stored in the file (attribute config_yaml)
- Per case: FV conservative cell averages at every solver step (4096 cells, coarsened to 512),
  the step times, and store_idx (the steps on the 0.002 grid)
- Writes one raw HDF5 file (output.path in the config) and a short run report next to it


Usage:
    python data_generation.py                                   # full raw dataset, all cores
    python data_generation.py --config my_config.yaml           # another config
    python data_generation.py --limit 1 --out data/smoke_raw.h5 # first case of each split
'''

import os
import json
import time
import argparse
import subprocess
from multiprocessing import Pool

import numpy as np
import h5py

import run as R


def stratified_log_uniform(n, lo, hi, seed):
    '''log r_i = log lo + (i + u_i)/n * log(hi/lo), u_i ~ U(0, 1).'''
    u = np.random.default_rng(seed).uniform(size=n)
    return np.exp(np.log(lo) + (np.arange(n) + u) / n * np.log(hi / lo))


def make_cases(cfg):
    '''List of (case_id, r, split) in config order; ids are consecutive integers.'''
    cases = []
    for c in cfg.cases:
        for r in stratified_log_uniform(int(c['n']), float(c['r_min']), float(c['r_max']), int(c['seed'])):
            cases.append((len(cases), float(r), c['split']))
    return cases


def generate_case(job):
    cfg, case = job
    cid, r, split = case
    t0 = time.time()
    fv = R.run_case(cfg, r)
    U_steps = R.coarsen(fv['U_steps'], cfg.coarsen)     # every solver step, saved grid
    del fv['U_steps']

    # Conservation check: mass and energy (no wave reaches a wall, v = 0 there)
    dx = cfg.length / cfg.n_x
    tot = U_steps.sum(axis=1) * dx
    drift = (np.abs(tot[:, [0, 2]] - tot[0, [0, 2]]).max(axis=0) / np.abs(tot[0, [0, 2]])).tolist()

    report = {'id': cid, 'r': r, 'split': split, 'dt': fv['dt'], 'cfl': fv['cfl'],
              'steps': len(fv['t_steps']) - 1, 'mass_energy_rel_drift': drift, 'seconds': time.time() - t0}
    return case, {'U_fv_steps': U_steps, **fv}, report


def _git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'none'


def write_header(h5, cfg, cases):
    a = h5.attrs
    a['config_yaml'] = cfg.text
    a['gamma'] = cfg.gamma
    a['x0'] = cfg.x0
    a['domain'] = [0.0, cfg.length]
    a['Nx'] = cfg.n_x
    a['Nx_fine'] = cfg.n_fine
    a['dt_store'] = cfg.dt_snapshot
    a['t_max'] = cfg.t_max
    a['solver'] = 'euler1d.solver.EulerSolver'
    a['flux'] = cfg.flux
    a['reconstruction'] = cfg.reconstruction
    a['time_integrator'] = 'RK4, fixed dt per case dividing dt_store'
    a['cfl'] = cfg.cfl
    a['coarsening'] = f'mean of {cfg.coarsen} fine cells ({cfg.n_fine} -> {cfg.n_x})'
    a['code_git_hash'] = _git_hash()
    g = h5.create_group('grid')
    g['x_edges'] = cfg.x_edges(cfg.n_x)
    g['x_centers'] = cfg.x_centers(cfg.n_x)
    g['t'] = cfg.t_snapshots()                          # the snapshot times that store_idx points to
    s = h5.create_group('splits')
    for c in cfg.cases:
        s[c['split']] = np.array([cid for cid, _, sp in cases if sp == c['split']], dtype=np.int64)
    h5.create_group('cases')


def write_case(h5, cfg, case, d):
    cid, r, split = case
    left, right = cfg.states(r)
    g = h5['cases'].create_group(f'{cid:03d}')
    g.attrs['r'] = r
    g.attrs['mu'] = np.log(r)
    g.attrs['left'] = left
    g.attrs['right'] = right
    g.attrs['split'] = split
    g.attrs['dt_solver'] = d['dt']
    g.attrs['cfl_actual'] = d['cfl']
    g.create_dataset('U_fv_steps', data=d['U_fv_steps'].astype(np.float32), chunks=(64, cfg.n_x, 3),
                     compression='gzip', compression_opts=cfg.gzip_level, shuffle=True)
    g['t_steps'] = d['t_steps']
    g['store_idx'] = d['store_idx']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=R.DEFAULT_CONFIG, help='generation config (YAML)')
    ap.add_argument('--out', default=None, help='overrides output.path in the config')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--limit', type=int, default=None, help='only the first N cases of each split (smoke test)')
    args = ap.parse_args()

    cfg = R.load_config(args.config)
    out_path = args.out or cfg.output_path
    cases = make_cases(cfg)
    if args.limit is not None:
        cases = [x for c in cfg.cases for x in [x for x in cases if x[2] == c['split']][:args.limit]]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    reports = []
    t0 = time.time()
    with h5py.File(out_path, 'w') as h5, Pool(args.workers) as pool:
        write_header(h5, cfg, cases)
        for i, (case, data, rep) in enumerate(pool.imap_unordered(generate_case, [(cfg, c) for c in cases])):
            write_case(h5, cfg, case, data)
            reports.append(rep)
            print(f"[{i + 1}/{len(cases)}] case {case[0]:03d} {case[2]:8s} r = {case[1]:6.3f}  "
                  f"{rep['steps']} steps  {rep['seconds']:.0f} s  (total {time.time() - t0:.0f} s)", flush=True)

    reports.sort(key=lambda r: r['id'])
    worst = np.max([r['mass_energy_rel_drift'] for r in reports], axis=0).tolist()
    with open(os.path.splitext(out_path)[0] + '_report.json', 'w') as f:
        json.dump({'config': args.config, 'mass_energy_rel_drift_max': worst, 'cases': reports}, f, indent=1)
    print(f'max mass / energy relative drift: {worst[0]:.1e} / {worst[1]:.1e}')


if __name__ == '__main__':
    main()
