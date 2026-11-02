'''
Shared by evaluation.diagnostics and evaluation.animate: paths and constants, the data (FV, exact cell averages,
classes), the exact Sod solution, model loading and the rollout. No figures here.

Every rollout starts from the FV state at t0 = 0.02 and runs to 0.24, each step fed the model's own output.
t > T_train = 0.16 is the held-out time band.
'''

from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model.neural_chart import NeuralChart1DST
from train.dataset import SnapshotDataset
from train.helper import points

ROOT = Path(__file__).parent.parent
DATA = ROOT / 'data' / 'step1_sod_pr.h5'
CFG = ROOT / 'configs' / 'models_cfg'
RUNS = ROOT / 'results' / 'models'
OUT = ROOT / 'results' / 'diagnostics'

def model_table(runs=RUNS, cfg=CFG):
    '''arm: (model config, checkpoint). Arms without a checkpoint are skipped.'''
    runs, cfg = Path(runs), Path(cfg)
    return {'full':       (cfg / 'model_one' / 'model_config.yaml',      runs / 'test_model_1D' / 'trained_model.pt'),
            'free-jump':  (cfg / 'model_one_free' / 'model_config.yaml', runs / 'test_model_free_1D' / 'trained_model.pt'),
            'one-region': (cfg / 'baseline_one' / 'model_config.yaml',   runs / 'baseline_1D' / 'trained_model.pt')}

MODELS = model_table()

def configure(runs=None, cfg=None, out=None):
    '''Point the scripts at another set of models (runs: the folder holding <name>/trained_model.pt, cfg: the
    models_cfg folder) and/or another output folder. Used by the --runs / --cfg / --out flags.'''
    global MODELS, OUT
    if runs or cfg:
        MODELS = model_table(runs or RUNS, cfg or CFG)
    if out:
        OUT = Path(out)
COLORS = {'full': 'C0', 'free-jump': 'C1', 'one-region': 'C2', 'FV': '0.5'}
SPLITS = ('test_id', 'test_ood')                     # default splits; --splits val picks a model set on validation
ALL_SPLITS = ('train', 'val', 'test_id', 'test_ood')

GAMMA, X0, RIGHT = 1.4, 0.4, (0.125, 0.0, 0.1)       # Sod family: left = (1, 0, 0.1 r)
T_TRAIN, T_MIN, T_END = 0.16, 0.02, 0.24
N = 512
DX = 1 / N
XC = (np.arange(N) + 0.5) * DX                       # cell centres, data x
SIDE = 4                                             # cells: states either side of the shock are read this far off
WAVES = ('contact', 'shock')
VARS = ('rho', 'v', 'p')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ======================================================================================
# Data and the exact solution
# ======================================================================================

class Split:
    '''One split from t = 0 to T_END: FV W [C, S, 512, 3], exact W_ex (primitives), classes [C, S, 512], t [S],
    r [C]. n0 is the rollout start (first t >= T_MIN); rollouts predict snapshots n0+1 .. S-1.'''

    def __init__(self, split):
        ds = SnapshotDataset(DATA, split, t_max=T_END, exact=True)
        self.name, self.r, self.t = split, ds.r.numpy(), ds.t.numpy() * T_TRAIN
        self.W = ds.W.double().numpy()
        self.W_ex = cons_to_prim(ds.U_exact.double().numpy())
        self.x_cell = ds.x_centers                                           # x~, for the model
        with h5py.File(DATA, 'r') as f:
            self.classes = np.stack([f['cases'][c]['classes'][()][ds.use] for c in ds.case_ids])
        self.n0 = int((self.t < T_MIN - 1e-9).sum())
        self.C = len(self.r)
        self.exact = [exact_sod(r) for r in self.r]

    def x_exact(self, c, t):
        '''Exact (contact, shock) positions [..., 2] of case c at times t.'''
        return X0 + self.exact[c]['speeds'][2:] * np.asarray(t)[..., None]

    def mid_case(self):
        return int(np.argsort(self.r)[self.C // 2])

def cons_to_prim(U):
    rho, m, E = U[..., 0], U[..., 1], U[..., 2]
    v = m / rho
    return np.stack([rho, v, (GAMMA - 1) * (E - 0.5 * rho * v ** 2)], axis=-1)

def exact_sod(r):
    '''Exact Riemann solution of the Sod case with left (1, 0, 0.1 r): wave speeds (fan head, fan tail, contact,
    shock), the star states and the shock Mach number. Newton on Toro's pressure function.'''
    g = GAMMA
    rL, uL, pL = 1.0, 0.0, 0.1 * r
    rR, uR, pR = RIGHT
    aL, aR = np.sqrt(g * pL / rL), np.sqrt(g * pR / rR)

    def f(p, rk, pk, ak):
        if p > pk:                                                            # shock branch
            A, B = 2 / ((g + 1) * rk), (g - 1) / (g + 1) * pk
            q = np.sqrt(A / (p + B))
            return (p - pk) * q, q * (1 - 0.5 * (p - pk) / (p + B))
        e = (p / pk) ** ((g - 1) / (2 * g))                                   # rarefaction branch
        return 2 * ak / (g - 1) * (e - 1), (p / pk) ** (-(g + 1) / (2 * g)) / (rk * ak)

    p = 0.5 * (pL + pR)
    for _ in range(100):
        (fl, dl), (fr, dr) = f(p, rL, pL, aL), f(p, rR, pR, aR)
        dp = (fl + fr + uR - uL) / (dl + dr)
        p = max(p - dp, 1e-12)
        if abs(dp) < 1e-14 * p:
            break
    u = 0.5 * (uL + uR) + 0.5 * (f(p, rR, pR, aR)[0] - f(p, rL, pL, aL)[0])
    k = (g - 1) / (g + 1)
    rho_sL = rL * (p / pL) ** (1 / g)
    rho_sR = rR * (p / pR + k) / (k * p / pR + 1)
    a_sL = aL * (p / pL) ** ((g - 1) / (2 * g))
    S = uR + aR * np.sqrt((g + 1) / (2 * g) * p / pR + (g - 1) / (2 * g))
    return {'speeds': np.array([uL - aL, u - a_sL, u, S]),
            'W_star_L': np.array([rho_sL, u, p]), 'W_star_R': np.array([rho_sR, u, p]),
            'Mn': S / aR}

def exact_profile(r, t):
    '''Exact point values (rho, v, p) [512, 3] at the cell centres at time t > 0, any t (infinite domain): for
    times past the data. The data's U_exact are cell averages; these differ only in the cells a wave crosses.'''
    e, g = exact_sod(r), GAMMA
    s = e['speeds']
    pL, aL = 0.1 * r, np.sqrt(GAMMA * 0.1 * r)
    xi = (XC - X0) / t
    a = 2 / (g + 1) * (aL - (g - 1) / 2 * np.clip(xi, s[0], s[1]))
    fan = np.stack([(a / aL) ** (2 / (g - 1)), 2 / (g + 1) * (aL + np.clip(xi, s[0], s[1])),
                    pL * (a / aL) ** (2 * g / (g - 1))], -1)
    W = np.select([xi[:, None] < s[0], xi[:, None] < s[1], xi[:, None] < s[2], xi[:, None] < s[3]],
                  [np.array([1.0, 0.0, pL]), fan, e['W_star_L'], e['W_star_R']], np.array(RIGHT))
    return W

# ======================================================================================
# Running the model
# ======================================================================================

def load_model(model_cfg, ckpt):
    with open(model_cfg) as f:
        cfg = yaml.safe_load(f)
    model = NeuralChart1DST(config=cfg).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE)['model'])
    return model.eval()

def available_models():
    out = {}
    for arm, (cfg, ckpt) in MODELS.items():
        if Path(ckpt).exists():
            out[arm] = load_model(cfg, ckpt)
        else:
            print(f'skipping {arm}: no checkpoint at {ckpt}')
    return out

@torch.no_grad()
def step(model, W, x_cell):
    '''One step from W [B, 512, 3] (numpy or tensor) at t_n -> numpy dict at t_n+1: W [B, 512, 3],
    labels [B, 512], W_regions [B, 512, K, 3], phi [B, 512, K] (advected scores; None for one-region).'''
    W_grid = torch.as_tensor(W).float().to(DEVICE).transpose(1, 2)
    B = len(W_grid)
    x, group = points(x_cell.to(DEVICE), B)
    out = model(x, W_grid, group)
    view = lambda v: None if v is None else v.reshape(B, N, *v.shape[1:]).double().cpu().numpy()
    return {'W': view(out['W']), 'labels': out['labels'].view(B, N).cpu().numpy(),
            'W_regions': view(out['W_regions']), 'phi': view(out['phi_hat'])}

def rollout(model, sp, cases=None, n_steps=None):
    '''The cases of split sp (default: all) from t0 on the model's own output, n_steps steps (default: to the
    last snapshot, T_END). Arrays [C, T, ...], T = n_steps.'''
    W, steps = sp.W[:, sp.n0] if cases is None else sp.W[cases, sp.n0], []
    for _ in range(n_steps or len(sp.t) - sp.n0 - 1):
        s = step(model, W, sp.x_cell)
        W = s['W']
        steps.append(s)
    return {k: None if steps[0][k] is None else np.stack([s[k] for s in steps], 1) for k in steps[0]}

def rollouts(models, splits):
    '''{split: {arm: rollout}}, computed once and shared by the figures.'''
    return {name: {arm: rollout(m, sp) for arm, m in models.items()} for name, sp in splits.items()}

# ======================================================================================
# Plot helpers
# ======================================================================================

def held_out(ax):
    ax.axvspan(T_TRAIN, T_END, color='0.92', zorder=0)

def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT / name}')