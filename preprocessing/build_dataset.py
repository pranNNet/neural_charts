'''
Training preprocessing: raw FV file -> training file. See README.md.

Per case: the 121 snapshots on the 0.002 grid as primitives (W_fv), the jump flags, and the exact
cell averages (U_exact, evaluation only). Data comes from the raw file only; settings from the config.

Usage (from /mnt/project-files):
    python -m preprocessing.build_dataset [--config <yaml>] [--raw <raw.h5>] [--out <train.h5>]
'''

import os
from pathlib import Path
import argparse
from functools import partial
from multiprocessing import Pool

import numpy as np
import h5py
import yaml

from euler1d_data.infrastructure.solver_config import cons_to_prim
from preprocessing.jump_flags import jump_flags
from preprocessing.exact import exact_cell_averages

# DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'preprocessing_config.yaml')
DEFAULT_CONFIG = Path(__file__).parent / 'configs' / 'preprocessing_config.yaml'


def load_config(path):
    '''The YAML config as a dict, with its paths made absolute (they are relative to the config file).'''
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = os.path.dirname(os.path.abspath(path))
    cfg['paths'] = {k: os.path.normpath(os.path.join(base, v)) for k, v in cfg['paths'].items()}
    return cfg


def process_case(raw_path, cfg, cid):
    with h5py.File(raw_path, 'r') as h5:
        gamma, x0 = float(h5.attrs['gamma']), float(h5.attrs['x0'])
        edges, t = h5['grid/x_edges'][()], h5['grid/t'][()]
        g = h5['cases'][cid]
        attrs = dict(g.attrs)
        U = g['U_fv_steps'][g['store_idx'][()]].astype(np.float64)        # [121, 512, 3]
    W = np.moveaxis(cons_to_prim(np.moveaxis(U, -1, 0), gamma), 0, -1)
    flags = jump_flags(W, t, gamma, cfg['jump_flags'])
    U_exact, speeds = exact_cell_averages(attrs['left'], attrs['right'], x0, gamma, edges, t)

    # Check: on every used snapshot, one flag per jump, at the exact contact and shock
    dx, faces = edges[1] - edges[0], edges[1:-1]
    used = np.flatnonzero(flags[:, 0] >= 0)
    n_flags = flags[used].sum(axis=1)
    err = max(np.abs(faces[flags[j] == 1][:, None] - (x0 + speeds[2:] * t[j])).min(axis=0).max() / dx
              for j in used)
    check = (int(n_flags.min()), int(n_flags.max()), float(err))
    return cid, attrs, W.astype(np.float32), flags, U_exact.astype(np.float32), check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=DEFAULT_CONFIG)
    ap.add_argument('--raw', help='overrides paths.raw in the config')
    ap.add_argument('--out', help='overrides paths.out in the config')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    args = ap.parse_args()

    cfg = load_config(args.config)
    for k in ['raw', 'out']:
        if getattr(args, k):
            cfg['paths'][k] = os.path.abspath(getattr(args, k))
    raw_path, out_path = cfg['paths']['raw'], cfg['paths']['out']
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    kw = dict(compression='gzip', compression_opts=4, shuffle=True)
    with h5py.File(raw_path, 'r') as raw, h5py.File(out_path, 'w') as out:
        out.attrs.update(raw.attrs)                                        # provenance from the raw file
        out.attrs['raw_file'] = raw_path
        out.attrs['T_train'] = cfg['T_train']
        out.attrs['preprocessing_config'] = yaml.safe_dump(cfg, sort_keys=False)
        raw.copy('grid', out)
        raw.copy('splits', out)
        cids = sorted(raw['cases'].keys())

        print('case  split     r       flags per snapshot  max flag-to-exact (dx)')
        with Pool(args.workers) as pool:
            for cid, attrs, W, flags, U_exact, (n_lo, n_hi, err) in pool.imap(partial(process_case, raw_path, cfg), cids):
                g = out.create_group(f'cases/{cid}')
                g.attrs.update(attrs)
                g.create_dataset('W_fv', data=W, **kw)
                g.create_dataset('jump_flags', data=flags, **kw)
                g.create_dataset('U_exact', data=U_exact, **kw)
                print(f"{cid}   {attrs['split']:8s}  {attrs['r']:6.3f}  {n_lo} to {n_hi}              {err:.2f}", flush=True)


if __name__ == '__main__':
    main()
