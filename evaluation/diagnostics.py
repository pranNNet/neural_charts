'''
Diagnostics for ladder step 1 (plan §9; "Diagnostics round 2", section 0): full model, free-jump arm,
one-region arm.

Core (tier 1): C1 accuracy vs r, C2 profiles, C3 H1 slope, C4 shock accuracy vs r, C5 error split,
C6 front error vs t, table T. Explainability (tier 2): X1 zone-to-region confusion, X2 zone fields,
X3 region map with zero contours, X4 latents vs r, X5 oracle swaps. Tier 3 lives in debug.py.

Encoder latents only. The exact solution is the reference, FV alongside.
Every metric that names a front uses the model front matched to the exact wave (plan §9.2).
For the one-region arm the fronts are the faces jump_flags marks on its cell averages.
Evaluated on test_id and test_ood, every stored snapshot t in (0, 0.24].

    python -m evaluation.diagnostics          (from the project root)

Figures and the table go to results/diagnostics/.
'''

from pathlib import Path
import os
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch

# import preprocessing                                    # puts euler1d-data on the path
from euler1d_data.infrastructure.results import ExactRiemannSolver
from preprocessing.exact import wave_speeds
from preprocessing.jump_flags import jump_flags
from model.neural_chart import NeuralChart1DST
from model.encoder import LatentEncoder
from model.transforms import MSFourier_Features
from model.shock_interface import RH_residual, Rankine_Hugoniot
from train.dataset import load_splits


ROOT = Path(__file__).parent.parent
DATA = ROOT / 'data' / 'step1_sod_pr.h5'
FLAGS_CFG = ROOT / 'configs' / 'preprocessing_config.yaml'
OUT = ROOT / 'results' / 'diagnostics'

# name: (model config, checkpoint). Arms whose checkpoint is missing are skipped.
MODELS = {'full':       (ROOT / 'configs' / 'models_cfg' / 'model_one' / 'model_config.yaml',      ROOT / 'results' / 'models' / 'test_model_1D' / 'trained_model.pt'),
          'free-jump':  (ROOT / 'configs' / 'models_cfg' / 'model_one_free' / 'model_config.yaml',   ROOT / 'results' / 'models' / 'test_model_free_1D' / 'trained_model.pt'),
          'one-region': (ROOT / 'configs' / 'models_cfg' / 'baseline_one' / 'model_config.yaml',  ROOT / 'results' / 'models' / 'baseline_1D' / 'trained_model.pt')}
COLORS = {'full': 'C0', 'free-jump': 'C1', 'one-region': 'C2'}

# C3, latent-size sweep: arm -> {d: (model config, checkpoint)}. Missing checkpoints are skipped.
SWEEP_D = (2, 4, 8, 16, 32)
SWEEP = {arm: {d: (ROOT / 'configs' / 'sweep' / f'{tag}_d{d}.yaml',
                   ROOT / 'results' / 'sweep' / f'{tag}_d{d}' / 'trained_model.pt') for d in SWEEP_D}
         for arm, tag in [('full', 'full'), ('one-region', 'one_region')]}

GAMMA, X0, RIGHT = 1.4, 0.4, (0.125, 0.0, 0.1)      # Sod family, plan §1.1: left = (1, 0, 0.1 r)
T_TRAIN, T_MIN = 0.16, 0.02                         # T_MIN: waves too close together before this (as in jump_flags)
DX = 1 / 512
TOL = 3                                             # cells: a model front this close to an exact wave is a hit
SIDE = 4                                            # cells: one-region side states are read this far off the shock face
GL_X, GL_W = np.polynomial.legendre.leggauss(4)
XC = (np.arange(512) + 0.5) * DX                                                   # cell centres
XQ = ((np.arange(512) + 0.5)[:, None] + 0.5 * GL_X).reshape(-1) * DX               # [2048] Gauss points
REGIONS = ['left', 'fan', 'star-L', 'star-R', 'right']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ======================================================================================
# Loading
# ======================================================================================

def load_model(model_cfg, ckpt):
    '''Same construction as train_NC1DST.'''
    with open(model_cfg, 'r') as file:
        cfg = yaml.safe_load(file)

    model = NeuralChart1DST(config = cfg, N_input = cfg['architecture']['coordinate_input']).to(DEVICE)
    encoder = LatentEncoder(N_in = cfg['encoder']['encodernet']['input'],
                            z_geo = cfg['architecture']['latent']['z_geo'],
                            z_field = cfg['architecture']['latent']['z_field'],
                            hidden_layers = cfg['encoder']['encodernet']['hidden_layers']).to(DEVICE)

    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state['model'])
    encoder.load_state_dict(state['encoder'])
    return model.eval(), encoder.eval()


# ======================================================================================
# Exact solution
# ======================================================================================

def rh_jacobian(S):
    '''[3, 4] d W_RH / d (rho, v, p, s_ud) at the exact upstream state (right state), n_ud = -1.'''
    n = torch.tensor([-1.0], dtype=torch.float64)
    f = lambda u: Rankine_Hugoniot(u[None, :3], n, u[None, 3], GAMMA)[0][0]
    return torch.autograd.functional.jacobian(f, torch.tensor([*RIGHT, -S], dtype=torch.float64)).numpy()


def exact_case(r):
    '''Everything exact about one r: waves, star states, shock Mn, jumps and the RH Jacobian.'''
    sol = ExactRiemannSolver((1.0, 0.0, 0.1 * r), RIGHT, x_interface=X0, gamma=GAMMA)
    speeds = wave_speeds(sol)                                       # fan head, fan tail, contact, shock
    W_star_R = np.array([sol.rho_star_R, sol.u_star, sol.P_star])
    jump = np.array([abs(sol.rho_star_L - sol.rho_star_R),          # |[W]| summed over (rho, v, p):
                     np.abs(W_star_R - RIGHT).sum()])               # contact (only rho jumps), shock
    return dict(sol=sol, speeds=speeds, W_star_R=W_star_R, jump=jump,
                Mn_ex=speeds[3] / sol.a_R, J=rh_jacobian(speeds[3]))


def exact_waves(r):
    '''Speeds (fan head, fan tail, contact, shock) and |[W]| at (contact, shock).'''
    e = exact_case(r)
    return e['speeds'], e['jump']


def exact_region(e, k, x, t):
    '''Exact state of region k (REGIONS order) at (x, t), extended outside the region (fan: xi clamped).'''
    sol, s = e['sol'], e['speeds']
    if k == 1:
        g, aL = GAMMA, sol.a_L
        xi = np.clip((x - X0) / t, s[0], s[1])
        a = 2 / (g + 1) * (aL - (g - 1) / 2 * xi)                   # u_L = 0
        W = [sol.rho_L * (a / aL) ** (2 / (g - 1)), 2 / (g + 1) * (aL + xi), sol.P_L * (a / aL) ** (2 * g / (g - 1))]
        return np.stack(np.broadcast_arrays(*W), axis=-1)
    const = {0: (sol.rho_L, sol.u_L, sol.P_L), 2: (sol.rho_star_L, sol.u_star, sol.P_star),
             3: tuple(e['W_star_R']), 4: RIGHT}[k]
    return np.broadcast_to(np.array(const), np.broadcast(x, t).shape + (3,))


def region_ids(case):
    '''[T, 2048] exact region (0..4, REGIONS order) at every Gauss point.'''
    edges = X0 + case['speeds'][None, :] * case['t'][:, None]                      # [T, 4]
    return np.stack([np.searchsorted(e, XQ) for e in edges])


def cons_to_prim(U):
    rho, m, E = U[..., 0], U[..., 1], U[..., 2]
    v = m / rho
    return np.stack([rho, v, (GAMMA - 1) * (E - 0.5 * rho * v ** 2)], axis=-1)


def cell_average(Wp):
    '''[..., 2048, 3] point values at the Gauss points -> [..., 512, 3] cell averages.'''
    Wp = Wp.reshape(*Wp.shape[:-2], 512, 4, 3)
    return (Wp * GL_W[:, None] / 2).sum(-2)


def rel_l1(W, W_ex):
    '''Relative L1 per variable over all given snapshots and cells, then the mean over (rho, v, p).'''
    ax = tuple(range(W.ndim - 1))
    return (np.abs(W - W_ex).sum(ax) / np.abs(W_ex).sum(ax)).mean()


# ======================================================================================
# Running a model on one case
# ======================================================================================

def front_summary(model, fronts, t_hat, z_field):
    '''Fronts of one snapshot as numpy, in data units, plus RH residual, Mn and the composed downstream state.'''
    fr = fronts[0]
    sh = fr['is_shock']

    # downstream state of each front, as the composed field has it: the RH state where the arm
    # overwrites, else the free downstream field read side_offset off the front (as detection read it)
    overwrite = (sh if model.jump_mode == 'rh' else torch.zeros_like(sh))[:, None]
    W_dn = torch.where(overwrite, fr['W_RH'], fr['W_dn'])

    W_up = fr['W_up']
    c_up = torch.sqrt(model.gamma * W_up[:, 2] / W_up[:, 0])
    res = RH_residual(W_up[sh], W_dn[sh], fr['s_ud'][sh], fr['n_ud'][sh], model.gamma)

    out = {key: val.detach().cpu().numpy() for key, val in fr.items()}
    out.update(x = ((fr['x'] + 1) / 2).cpu().numpy(),                 # x~ -> x
               speed = (fr['dxdt'] * model.k).cpu().numpy(),          # dx~/dt~ -> dx/dt
               is_shock = sh.cpu().numpy().astype(bool),
               Mn = fr['Mn_raw'][sh].cpu().numpy(),                   # detected shocks only
               Mn_all = ((W_up[:, 1] * fr['n_ud'] - fr['s_ud']) / c_up).cpu().numpy(),   # every front
               W_dn_comp = W_dn.cpu().numpy(),
               rh = res.cpu().numpy())
    return out


@torch.no_grad()
def run_case(model, encoder, r, t_hat, x_edges, keep=False, blend='hard', eps=None, z=None):
    '''
    Every snapshot of one case. Cell averages from 4 Gauss points per cell
    (not split at the model's fronts: simpler than D42, off by at most part of one cell per front).
    keep: also return W_regions [T, 2048, K, 3] and phi_hat [T, 2048, K] at the Gauss points.
    z: (z_geo, z_field) [1, d] to use instead of the encoder's (debug.py swap test).
    '''
    z_geo, z_field = encoder(torch.tensor([[r]], device=DEVICE)) if z is None else z

    centres = 0.5 * (x_edges[1:] + x_edges[:-1])
    xq = (centres[:, None] + 0.5 * (x_edges[1] - x_edges[0]) * torch.tensor(GL_X)).reshape(-1)
    xq = xq.float().to(DEVICE)                                                     # [2048] x~
    n = len(xq)
    group = torch.zeros(n, dtype=torch.long, device=DEVICE)                        # one snapshot per call

    W, labels, fronts, W_regions, phi_hat = [], [], [], [], []
    for tj in t_hat.tolist():
        x = torch.stack([xq, torch.full_like(xq, tj)], dim=-1)
        out = model(x, z_geo.expand(n, -1), z_field.expand(n, -1), blend=blend, eps=eps, group=group)

        W.append(out['W'].double().cpu().numpy())
        labels.append(out['labels'].cpu().numpy())
        fronts.append(front_summary(model, out['fronts'], tj, z_field) if out['fronts'] else None)
        if keep:
            W_regions.append(out['W_regions'].float().cpu().numpy())
            if out['phi_hat'] is not None:
                phi_hat.append(out['phi_hat'].float().cpu().numpy())

    W = np.stack(W)                                                                # [T, 2048, 3] point values
    res = {'W': cell_average(W),                                                   # [T, 512, 3]
           'W_min': W.min(axis=(0, 1)),                                            # pointwise min (rho, v, p)
           'e_min': (W[..., 2] / ((GAMMA - 1) * W[..., 0])).min(),                 # pointwise min internal energy
           'labels': np.stack(labels),                                             # [T, 2048]
           'fronts': fronts}                                                       # None for one-region
    if keep:
        res['W_regions'] = np.stack(W_regions)
        res['phi_hat'] = np.stack(phi_hat) if phi_hat else None
    return res


def zone_stats(case):
    '''X1 and X5 for a level-set case: zone-to-region counts and the oracle-swap L1 (t in [T_MIN, T_train]).'''
    use = (case['t'] >= T_MIN) & (case['t'] <= T_TRAIN + 1e-9)
    reg, lab = region_ids(case)[use], case['labels'][use]
    K = case['W_regions'].shape[2]

    conf = np.zeros((5, K))
    np.add.at(conf, (reg.ravel(), lab.ravel()), 1)
    zone_of_region, region_of_zone = conf.argmax(1), conf.argmax(0)

    # oracle labels: each point takes the zone that mostly covers its exact region
    W_lab = np.take_along_axis(case['W_regions'][use], zone_of_region[reg][..., None, None], axis=2)[:, :, 0]
    # oracle fields: each zone's value is the exact state of the region it mostly covers
    kmap, t = region_of_zone[lab], case['t'][use][:, None]
    W_fld = np.zeros(kmap.shape + (3,))
    for k in range(5):
        W_fld[kmap == k] = exact_region(case, k, XQ[None], t)[kmap == k]

    W_ex = case['W_ex'][use]
    case['conf'] = conf
    case['swap'] = {'model': rel_l1(case['W'][use], W_ex),
                    'oracle labels': rel_l1(cell_average(W_lab), W_ex),
                    'oracle fields': rel_l1(cell_average(W_fld), W_ex)}


def evaluate(model, encoder, ds, flags_cfg, keep=()):
    '''One dict per case of the split: model output plus the references it is compared with.
    keep: case indices whose W_regions and phi_hat are kept (for X2, X3).'''
    t = ds.t.numpy() * T_TRAIN
    faces = np.arange(1, 512) * DX
    levelset = model.geo_mode == 'levelset'
    cases = []
    for c in range(ds.C):
        r = float(ds.r[c])
        out = run_case(model, encoder, r, ds.t, ds.x_edges, keep=levelset or c in keep)
        out.update(exact_case(r))
        out.update(r = r, t = t, levelset = levelset,
                   x_ex = X0 + out['speeds'][None, 2:] * t[:, None],              # [T, 2] contact, shock
                   W_ex = cons_to_prim(ds.U_exact[c].double().numpy()),
                   W_fv = ds.W[c].double().numpy())
        if levelset:
            zone_stats(out)
        else:                            # one-region: its fronts are the faces jump_flags marks on its output
            out['flags'] = jump_flags(out['W'], t, GAMMA, flags_cfg)
            out['fronts'] = [{'x': faces[f == 1]} for f in out['flags']]
        if c not in keep:
            out.pop('W_regions', None), out.pop('phi_hat', None)
        out['match'] = match_fronts(out)
        cases.append(out)
    return cases


# ======================================================================================
# Per-case quantities
# ======================================================================================

def match_fronts(case):
    '''Model front nearest each exact (contact, shock): idx [T, 2] (-1 if no front), dist [T, 2] in cells.'''
    idx = np.full(case['x_ex'].shape, -1)
    dist = np.full(case['x_ex'].shape, np.nan)
    for j, fr in enumerate(case['fronts']):
        if len(fr['x']):
            d = np.abs(fr['x'][:, None] - case['x_ex'][j][None])                   # [F, 2]
            idx[j], dist[j] = d.argmin(0), d.min(0) / DX
    return {'idx': idx, 'dist': dist}


def front_errors(case):
    '''[T, 2] |x_hat - x_exact| / dx at (contact, shock), matched model front; nan if no front.'''
    return case['match']['dist']


def front_counts(case):
    '''Per snapshot (t >= T_MIN): missed exact waves, spurious model fronts, and is_shock hits / tries.'''
    missed = spurious = right = tried = 0
    use = np.flatnonzero(case['t'] >= T_MIN)
    for j in use:
        fr = case['fronts'][j]
        d = np.abs(fr['x'][:, None] - case['x_ex'][j][None]) / DX < TOL            # [F, 2] hits
        missed += (~d.any(0)).sum()
        spurious += (~d.any(1)).sum()
        if case['levelset']:
            for w in np.flatnonzero(d.any(0)):                                      # 0 contact, 1 shock
                right += fr['is_shock'][case['match']['idx'][j, w]] == (w == 1)
                tried += 1
    return missed / len(use), spurious / len(use), right, tried


def shock_metrics(case):
    '''C4 on the front matched to the exact shock, T_MIN <= t <= T_train: relative errors of the
    line-fit speed, the local speed (median, level-set arms), Mn (median) and the downstream state (median).'''
    t, S = case['t'], case['speeds'][3]
    idx = case['match']['idx'][:, 1]
    use = np.flatnonzero((t >= T_MIN) & (t <= T_TRAIN + 1e-9) & (idx >= 0))
    if len(use) < 2:
        return dict(fit=np.nan, local=np.nan, Mn=np.nan, down=np.nan)

    x_s = np.array([case['fronts'][j]['x'][idx[j]] for j in use])
    S_fit = np.polyfit(t[use], x_s, 1)[0]

    if case['levelset']:
        local = np.median([abs(case['fronts'][j]['speed'][idx[j]] - S) / S for j in use])
        Mn = np.array([case['fronts'][j]['Mn_all'][idx[j]] for j in use])
        W_dn = np.array([case['fronts'][j]['W_dn_comp'][idx[j]] for j in use])
    else:
        # cell averages SIDE cells either side of the shock face; Mn from the fitted speed
        face = np.rint(x_s / DX).astype(int) - 1
        W_up = case['W'][use, np.minimum(face + 1 + SIDE, 511)]
        W_dn = case['W'][use, np.maximum(face - SIDE, 0)]
        Mn = (S_fit - W_up[:, 1]) / np.sqrt(GAMMA * W_up[:, 2] / W_up[:, 0])
        local = np.nan

    W_ex = case['W_star_R']
    return dict(fit = abs(S_fit - S) / S, local = local,
                Mn = np.median(np.abs(Mn - case['Mn_ex'])),
                down = np.median(np.abs(W_dn - W_ex).sum(-1) / np.abs(W_ex).sum()))


def split_terms(case):
    '''
    C5(a), against exact, T_MIN <= t <= T_train, as shares of sum |W_ex| (sum over rho, v, p):
    geometry = cells between each matched model front and its exact front; front band = the other cells
    within 2 cells of an exact front; field = everything else. They partition the cells, so the plan's
    remainder is zero by construction and is not reported.
    '''
    geo = band = field = ref = 0.0
    for j in np.flatnonzero((case['t'] >= T_MIN) & (case['t'] <= T_TRAIN + 1e-9)):
        err = np.abs(case['W'][j] - case['W_ex'][j]).sum(-1)                       # [512]
        x_ex, idx = case['x_ex'][j], case['match']['idx'][j]
        wrong = np.zeros(512, bool)
        for w in np.flatnonzero(idx >= 0):
            a, b = sorted((case['fronts'][j]['x'][idx[w]], x_ex[w]))
            wrong |= (XC > a) & (XC < b)
        guard = np.abs(XC[:, None] - x_ex[None]).min(1) < 2 * DX
        geo += err[wrong].sum()
        band += err[guard & ~wrong].sum()
        field += err[~guard & ~wrong].sum()
        ref += np.abs(case['W_ex'][j]).sum()
    return {'geometry': geo / ref, 'front band': band / ref, 'field': field / ref}


def geometry_error(case, j):
    '''(measured, predicted) geometry term of one snapshot, against the exact solution (§9.3).'''
    idx, x_ex = case['match']['idx'][j], case['x_ex'][j]
    x_hat = case['fronts'][j]['x'][idx]                                             # matched model fronts

    err = np.abs(case['W'][j] - case['W_ex'][j]).sum(-1)                            # [512]
    wrong = np.zeros(512, bool)
    for a, b in zip(np.minimum(x_hat, x_ex), np.maximum(x_hat, x_ex)):
        wrong |= (XC > a) & (XC < b)                                                # cells between the two fronts
    return err[wrong].sum() * DX, (case['jump'] * np.abs(x_hat - x_ex)).sum()


def jump_error(case, j):
    '''(measured, predicted) jump term at the matched shock of one snapshot (level-set arms).
    Predicted = linearized RH map: |J [W_up - W_R; s_ud - s_exact]|, summed over (rho, v, p).'''
    fr, i = case['fronts'][j], case['match']['idx'][j, 1]
    measured = np.abs(fr['W_dn_comp'][i] - case['W_star_R']).sum()
    du = np.concatenate([fr['W_up'][i] - RIGHT, [fr['s_ud'][i] + case['speeds'][3]]])
    return measured, np.abs(case['J'] @ du).sum()


def snapshots(case, t_lo=T_MIN, t_hi=T_TRAIN):
    '''Snapshot indices with t_lo <= t <= t_hi where both exact waves have a matched front.'''
    t = case['t']
    return np.flatnonzero((t >= t_lo) & (t <= t_hi + 1e-9) & (case['match']['idx'] >= 0).all(1))


def picks(res):
    '''(split, case index, title) for low r, mid r and OOD-far.'''
    ref = next(iter(res.values()))
    id_r = [c['r'] for c in ref['test_id']]
    return [('test_id', int(np.argmin(id_r)), 'low r'),
            ('test_id', int(np.argsort(id_r)[len(id_r) // 2]), 'mid r'),
            ('test_ood', int(np.argmax([c['r'] for c in ref['test_ood']])), 'OOD-far')]


def all_cases(res, name):
    return res[name]['test_id'] + res[name]['test_ood']


def levelset_arms(res):
    return [name for name in res if res[name]['test_id'][0]['levelset']]


# ======================================================================================
# Tier 1: core figures and table
# ======================================================================================

def fig_C1_accuracy(res, out):
    '''C1: relative L1 against exact vs r, t <= T_train, ID band shaded, FV-vs-exact as the floor.'''
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axvspan(2, 10, color='0.9', label='ID')
    use = lambda c: c['t'] <= T_TRAIN + 1e-9
    for name in res:
        cases = sorted(all_cases(res, name), key=lambda c: c['r'])
        ax.plot([c['r'] for c in cases], [rel_l1(c['W'][use(c)], c['W_ex'][use(c)]) for c in cases],
                'o-', ms=3, color=COLORS[name], label=name)
    ax.plot([c['r'] for c in cases], [rel_l1(c['W_fv'][use(c)], c['W_ex'][use(c)]) for c in cases],
            'k--', lw=1, label='FV vs exact')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('r')
    ax.set_ylabel('relative L1 vs exact')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'C1_accuracy_vs_r.png', dpi=150)


def fig_C2_profiles(res, out):
    '''C2: rho, v, p at t = T_train for low r, mid r and OOD-far.'''
    fig, ax = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
    for col, (split, c, title) in enumerate(picks(res)):
        ref = next(iter(res.values()))[split][c]
        j = int(np.argmin(np.abs(ref['t'] - T_TRAIN)))
        for row, var in enumerate(['rho', 'v', 'p']):
            a = ax[row, col]
            a.plot(XC, ref['W_fv'][j, :, row], '.', ms=2, color='0.6', label='FV')
            a.plot(XC, ref['W_ex'][j, :, row], 'k-', lw=1, label='exact')
            for name in res:
                a.plot(XC, res[name][split][c]['W'][j, :, row], color=COLORS[name], lw=1, label=name)
            a.set_ylabel(var)
        ax[0, col].set_title(f'{title} (r = {ref["r"]:.2f})')
        ax[-1, col].set_xlabel('x')
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(f't = {T_TRAIN}')
    fig.tight_layout()
    fig.savefig(out / 'C2_profiles.png', dpi=150)


def sweep_l1(test_id):
    '''C3: arm -> {d: [test_id relative L1 per case]}, t <= T_train, for the sweep checkpoints that exist.'''
    use = test_id.t <= 1 + 1e-6                                     # t~ <= 1, i.e. t <= T_train
    out = {}
    for arm, runs in SWEEP.items():
        for d, (cfg, ckpt) in runs.items():
            if not Path(ckpt).exists():
                continue
            model, encoder = load_model(cfg, ckpt)
            out.setdefault(arm, {})[d] = [
                rel_l1(run_case(model, encoder, float(test_id.r[c]), test_id.t[use], test_id.x_edges)['W'],
                       cons_to_prim(test_id.U_exact[c][use].double().numpy()))
                for c in range(test_id.C)]
    return out


def fig_C3_slope(sweep, out, n_boot=1000):
    '''C3: log-log test_id L1 vs latent size d, fitted slopes with bootstrap 95% intervals over cases.'''
    if not any(len(v) >= 2 for v in sweep.values()):
        print('C3 skipped: fewer than 2 sweep checkpoints per arm')
        return
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(5, 4))
    for arm, runs in sweep.items():
        if len(runs) < 2:
            continue
        d = np.array(sorted(runs))
        L = np.array([runs[k] for k in d])                                         # [D, cases]
        slope = np.polyfit(np.log(d), np.log(L.mean(1)), 1)[0]
        boot = [np.polyfit(np.log(d), np.log(L[:, rng.integers(0, L.shape[1], L.shape[1])].mean(1)), 1)[0]
                for _ in range(n_boot)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ax.loglog(d, L.mean(1), 'o-', color=COLORS[arm], label=f'{arm}: slope {slope:.2f} [{lo:.2f}, {hi:.2f}]')
    ax.loglog(d, L.mean(1)[0] * (d / d[0]) ** -0.5, 'k:', lw=1, label='d^(-1/2)')
    ax.set_xlabel('latent size d')
    ax.set_ylabel('test_id relative L1')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'C3_h1_slope.png', dpi=150)


def fig_C4_shock(res, out):
    '''C4: (a) shock-speed error (line fit), (b) Mn error, (c) downstream-state error, against r.'''
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name in res:
        cases = sorted(all_cases(res, name), key=lambda c: c['r'])
        r = [c['r'] for c in cases]
        m = [shock_metrics(c) for c in cases]
        ax[0].plot(r, [100 * x['fit'] for x in m], 'o-', ms=3, color=COLORS[name], label=name)
        ax[1].plot(r, [x['Mn'] for x in m], 'o-', ms=3, color=COLORS[name])
        ax[2].plot(r, [100 * x['down'] for x in m], 'o-', ms=3, color=COLORS[name])
    labels = ['shock-speed error, line fit (%)', '|Mn - Mn_exact| (median over t)',
              'downstream-state error (%, median over t)']
    for a, lab in zip(ax, labels):
        a.axvspan(2, 10, color='0.9')
        a.set_xscale('log')
        a.set_yscale('log')
        a.set_xlabel('r')
        a.set_title(lab, fontsize=10)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'C4_shock_vs_r.png', dpi=150)


def fig_C5_split(res, out, deltas=(2, 4, 8, 16, 32)):
    '''C5: (a) error split bars, (b) band profile with FV vs exact, (c) geometry and (d) jump term scatter.'''
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    splits = [('test_id', '-', 'o'), ('test_ood', '--', 'x')]

    # (a) stacked bars, mean over cases
    terms, colors = ['geometry', 'front band', 'field'], ['C3', 'C4', 'C7']
    ticks = []
    for i, (name, (split, _, _)) in enumerate([(n, s) for n in res for s in splits]):
        parts = [split_terms(c) for c in res[name][split]]
        bottom = 0
        for term, col in zip(terms, colors):
            h = np.mean([p[term] for p in parts])
            ax[0].bar(i, h, bottom=bottom, color=col, label=term if i == 0 else None)
            bottom += h
        ticks.append(f'{name}\n{split[5:]}')
    ax[0].set_xticks(range(len(ticks)), ticks, fontsize=7)
    ax[0].set_ylabel('share of sum |W_exact| (t <= T_train)')
    ax[0].legend(fontsize=8)

    # (b) band profile: fraction of L1 within delta cells of the exact contact and shock
    def band(cases, key):
        inside, total = np.zeros(len(deltas)), 0.0
        for c in cases:
            use = c['t'] <= T_TRAIN + 1e-9
            err = np.abs(c[key][use] - c['W_ex'][use]).sum(-1)                              # [T, 512]
            dist = np.abs(XC[None, :, None] - c['x_ex'][use][:, None, :]).min(-1) / DX       # [T, 512]
            inside += [err[dist < d].sum() for d in deltas]
            total += err.sum()
        return inside / total
    for split, ls, _ in splits:
        for name in res:
            ax[1].plot(deltas, band(res[name][split], 'W'), ls, marker='o', color=COLORS[name],
                       label=f'{name}, {split}')
        ax[1].plot(deltas, band(next(iter(res.values()))[split], 'W_fv'), ls, color='k', label=f'FV, {split}')
    ax[1].set_xscale('log', base=2)
    ax[1].set_xlabel('δ (cells)')
    ax[1].set_ylabel('fraction of L1 within δ of exact fronts')
    ax[1].legend(fontsize=7)

    # (c), (d) measured vs predicted, level-set arms, one point per (case, snapshot)
    for a, fn, xlab, ylab in [(ax[2], geometry_error, 'predicted  Σ |[W]| |x̂ − x|', 'measured  L1 over misassigned cells'),
                              (ax[3], jump_error, 'predicted  |J_RH Δ(W_up, s)|', 'measured  |W_down − W*_R|')]:
        for name in levelset_arms(res):
            for split, _, mk in splits:
                pts = np.array([fn(c, j) for c in res[name][split] for j in snapshots(c)])
                if len(pts):
                    a.loglog(pts[:, 1], pts[:, 0], mk, ms=3, color=COLORS[name], label=f'{name}, {split}')
        lim = a.get_xlim()
        a.plot(lim, lim, 'k--', lw=1)
        a.set_xlabel(xlab)
        a.set_ylabel(ylab)
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'C5_error_split.png', dpi=150)


def fig_C6_front_vs_t(res, out):
    '''C6, preview: front-position error against t, mean over cases and (contact, shock); ID and OOD.'''
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a, split in zip(ax, ('test_id', 'test_ood')):
        for name in [n for n in ('full', 'free-jump') if n in res]:
            err = np.nanmean([front_errors(c).mean(-1) for c in res[name][split]], axis=0)
            a.plot(res[name][split][0]['t'], err, color=COLORS[name], label=name)
        a.axvline(T_TRAIN, color='k', ls=':', label='T_train')
        a.set_xlabel('t')
        a.set_title(f'{split} (preview: H4 is formally at step 4)', fontsize=10)
    ax[0].set_ylabel('front-position error (cells)')
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(out / 'C6_front_error_vs_t.png', dpi=150)


def table_T(res, out):
    '''Certificates and front matching per arm (all test cases, all t), then the matched-ID-L1 table [D60].'''
    cols = ['RH median', 'RH max', 'Lax viol.', 'min rho', 'min p', 'min e', 'neg cells',
            'missed/snap', 'spur./snap', 'is_shock acc', 'local S err']
    lines = [f'{"model":<12}' + ''.join(f'{c:>13}' for c in cols)]
    fmt = lambda v, f: f'{"-":>13}' if v is None or np.isnan(v) else f'{v:>13{f}}'
    for name in res:
        cases = all_cases(res, name)
        W_min = np.min([c['W_min'] for c in cases], axis=0)
        neg = sum(((c['W'][..., 0] <= 0) | (c['W'][..., 2] <= 0)).sum() for c in cases)
        counts = np.array([front_counts(c) for c in cases])
        rh = Mn = acc = local = None
        if cases[0]['levelset']:
            fr = [f for c in cases for f in c['fronts']]
            rh, Mn = np.concatenate([f['rh'] for f in fr]), np.concatenate([f['Mn'] for f in fr])
            acc = counts[:, 2].sum() / max(counts[:, 3].sum(), 1)
            local = np.nanmedian([shock_metrics(c)['local'] for c in cases])
        row = [np.median(rh) if rh is not None and len(rh) else None, rh.max() if rh is not None and len(rh) else None,
               (Mn < 1).mean() if Mn is not None and len(Mn) else None,
               W_min[0], W_min[2], np.min([c['e_min'] for c in cases]), neg,
               counts[:, 0].mean(), counts[:, 1].mean(), acc, local]
        fmts = ['.2e', '.2e', '.3f', '.4f', '.4f', '.4f', '.0f', '.2f', '.2f', '.3f', '.2e']
        lines.append(f'{name:<12}' + ''.join(fmt(v, f) for v, f in zip(row, fmts)))

    # matched ID L1 [D60]: final checkpoints only, so this just shows whether the arms are within 10%
    use = lambda c: c['t'] <= T_TRAIN + 1e-9
    l1 = {name: {s: np.mean([rel_l1(c['W'][use(c)], c['W_ex'][use(c)]) for c in res[name][s]])
                 for s in ('test_id', 'test_ood')} for name in res}
    base = l1[next(iter(res))]['test_id']
    lines += ['', f'{"model":<12}{"ID L1":>13}{"OOD L1":>13}{"ID / first":>13}{"within 10%":>13}']
    for name, v in l1.items():
        ratio = v['test_id'] / base
        lines.append(f'{name:<12}{v["test_id"]:>13.4f}{v["test_ood"]:>13.4f}{ratio:>13.3f}{str(abs(ratio - 1) <= 0.1):>13}')

    text = '\n'.join(lines)
    print(text)
    (out / 'table.txt').write_text(text + '\n')


# ======================================================================================
# Tier 2: explainability
# ======================================================================================

def fig_X1_confusion(res, out):
    '''X1: fraction of each exact region's (x, t) area carried by each zone, aggregated over cases.'''
    arms = levelset_arms(res)
    if not arms:
        return
    fig, ax = plt.subplots(len(arms), 2, figsize=(9, 3.5 * len(arms)), squeeze=False)
    for i, name in enumerate(arms):
        for a, split in zip(ax[i], ('test_id', 'test_ood')):
            conf = sum(c['conf'] for c in res[name][split])
            conf = conf / conf.sum(1, keepdims=True)
            a.imshow(conf, cmap='Blues', vmin=0, vmax=1)
            for (k, z), v in np.ndenumerate(conf):
                a.text(z, k, f'{v:.2f}', ha='center', va='center', fontsize=7,
                       color='w' if v > 0.5 else 'k')
            a.set_yticks(range(5), REGIONS)
            a.set_xticks(range(conf.shape[1]))
            a.set_xlabel('zone id')
            a.set_title(f'{name}, {split}', fontsize=10)
    fig.tight_layout()
    fig.savefig(out / 'X1_zone_confusion.png', dpi=150)


def fig_X2_zone_fields(res, out):
    '''X2: every zone field W_k against x at T_train, solid in its support, dashed outside; low r and OOD-far.'''
    cases = [p for p in picks(res) if p[2] != 'mid r']
    for name in levelset_arms(res):
        fig, ax = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
        for col, (split, c, title) in enumerate(cases):
            case = res[name][split][c]
            j = int(np.argmin(np.abs(case['t'] - T_TRAIN)))
            Wr, lab = case['W_regions'][j], case['labels'][j]                      # [2048, K, 3], [2048]
            for row, var in enumerate(['rho', 'v', 'p']):
                a = ax[row, col]
                a.plot(XC, case['W_ex'][j, :, row], 'k-', lw=2, alpha=0.3, label='exact')
                for k in range(Wr.shape[1]):
                    a.plot(XQ, Wr[:, k, row], '--', lw=0.7, color=f'C{k}')
                    a.plot(XQ, np.where(lab == k, Wr[:, k, row], np.nan), '-', lw=1.5, color=f'C{k}', label=f'zone {k}')
                a.set_ylabel(var)
            ax[0, col].set_title(f'{title} (r = {case["r"]:.2f})')
            ax[-1, col].set_xlabel('x')
        ax[0, 0].legend(fontsize=7)
        fig.suptitle(f'{name}: zone fields at t = {T_TRAIN}')
        fig.tight_layout()
        fig.savefig(out / f'X2_zone_fields_{name}.png', dpi=150)


def fig_X3_region_map(res, out):
    '''X3: zones in (x, t) with the zero contour of every phi_hat_k (black) and the exact waves (dashed).'''
    arms = levelset_arms(res)
    if not arms:
        return
    cases = [p for p in picks(res) if p[2] != 'low r']
    fig, ax = plt.subplots(len(arms), 2, figsize=(11, 4.5 * len(arms)), squeeze=False)
    for i, name in enumerate(arms):
        for a, (split, c, title) in zip(ax[i], cases):
            case = res[name][split][c]
            t = case['t']
            a.imshow(case['labels'], origin='lower', aspect='auto', cmap='tab10', vmin=-0.5, vmax=9.5,
                     interpolation='nearest', extent=[0, 1, t[0], t[-1]])
            for k in range(case['phi_hat'].shape[-1]):
                a.contour(XQ, t, case['phi_hat'][..., k], levels=[0], colors='k', linewidths=0.6)
            for s in case['speeds']:
                a.plot(X0 + s * t, t, 'w--', lw=1)
            a.axhline(T_TRAIN, color='w', lw=0.8, ls=':')
            a.set_xlim(0, 1)
            a.set_xlabel('x')
            a.set_ylabel('t')
            a.set_title(f'{name}, {title} (r = {case["r"]:.2f})', fontsize=10)
    fig.tight_layout()
    fig.savefig(out / 'X3_region_map.png', dpi=150)


@torch.no_grad()
def latents(encoder, r_lo=2.0, r_hi=16.0, n=200, folds=5, lam=1e-3):
    '''X4: encoder latents on a fine r grid, and the 5-fold ridge R^2 of z_geo -> the 4 exact wave speeds.'''
    r = np.geomspace(r_lo, r_hi, n)
    z_geo, z_field = encoder(torch.tensor(r[:, None], dtype=torch.float32, device=DEVICE))
    z_geo, z_field = z_geo.cpu().numpy(), z_field.cpu().numpy()
    speeds = np.array([exact_waves(x)[0] for x in r])                              # [n, 4]

    X = np.hstack([z_geo, np.ones((n, 1))])
    pred = np.zeros_like(speeds)
    for test in np.array_split(np.random.default_rng(0).permutation(n), folds):
        train = np.setdiff1d(np.arange(n), test)
        A = X[train]
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ speeds[train])
        pred[test] = X[test] @ w
    R2 = 1 - ((pred - speeds) ** 2).sum(0) / ((speeds - speeds.mean(0)) ** 2).sum(0)

    zc = z_field - z_field.mean(0)
    pca = zc @ np.linalg.svd(zc, full_matrices=False)[2][:2].T                     # top 2 directions
    return dict(r=r, z_geo=z_geo, z_field_pca=pca, R2=R2)


def fig_X4_latents(lat, out):
    '''X4: z_geo components and the top 2 PCA directions of z_field against r, per arm.'''
    fig, ax = plt.subplots(len(lat), 2, figsize=(11, 3.5 * len(lat)), squeeze=False)
    for i, (name, d) in enumerate(lat.items()):
        for a, key, title in [(ax[i, 0], 'z_geo', 'z_geo'), (ax[i, 1], 'z_field_pca', 'z_field, top 2 PCA')]:
            a.axvspan(2, 10, color='0.9')
            for k in range(d[key].shape[1]):
                a.plot(d['r'], d[key][:, k], lw=1, label=f'{k}')
            a.set_xscale('log')
            a.set_xlabel('r')
            a.set_title(f'{name}: {title}', fontsize=10)
        R2 = ', '.join(f'{w} {v:.3f}' for w, v in zip(['head', 'tail', 'contact', 'shock'], d['R2']))
        ax[i, 0].legend(fontsize=7, title=f'R² z_geo→speed: {R2}', title_fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'X4_latents_vs_r.png', dpi=150)


def fig_X5_swaps(res, out):
    '''X5: relative L1 (t in [T_MIN, T_train]) of the model, with oracle labels and with oracle fields.'''
    arms = levelset_arms(res)
    if not arms:
        return
    keys = ['model', 'oracle labels', 'oracle fields']
    fig, ax = plt.subplots(figsize=(7, 4))
    ticks = []
    for i, (name, split) in enumerate([(n, s) for n in arms for s in ('test_id', 'test_ood')]):
        for k, key in enumerate(keys):
            ax.bar(i + 0.25 * (k - 1), np.mean([c['swap'][key] for c in res[name][split]]), 0.25,
                   color=f'C{k + 3}', label=key if i == 0 else None)
        ticks.append(f'{name}\n{split[5:]}')
    ax.set_xticks(range(len(ticks)), ticks, fontsize=8)
    ax.set_yscale('log')
    ax.set_ylabel('relative L1 vs exact')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'X5_oracle_swaps.png', dpi=150)


# ======================================================================================
# Training cases against FV (zone-split L1)
# ======================================================================================

def train_vs_fv(model, encoder, ds, c):
    '''
    One training case against its FV data, every snapshot (t < T_MIN included).
    Overwritten zone = downstream zone of each shock; a cell is in it if any of its 4 Gauss points is.
    '''
    r, t = float(ds.r[c]), ds.t.numpy() * T_TRAIN
    out = run_case(model, encoder, r, ds.t, ds.x_edges)
    err = np.abs(out['W'] - ds.W[c].double().numpy()).sum(-1)                      # [T, 512]

    S = exact_waves(r)[0][3]
    sol = ExactRiemannSolver((1.0, 0.0, 0.1 * r), RIGHT, x_interface=X0, gamma=GAMMA)
    x_s = X0 + S * t

    d = {k: np.full(len(t), np.nan) for k in ('L1_zone', 'L1_rest', 'speed', 'u_RH', 'p_RH')}
    d.update(r=r, t=t, S=S, u_star=sol.u_star, p_star=sol.P_star, L1=err.mean(1),
             n_fronts=np.zeros(len(t), int), n_shocks=np.zeros(len(t), int))

    for j, fr in enumerate(out['fronts']):
        d['n_fronts'][j], d['n_shocks'][j] = len(fr['x']), fr['is_shock'].sum()
        sh = fr['is_shock'].astype(bool)
        down = np.where(fr['up_is_a'], fr['b'], fr['a'])[sh]
        zone = np.isin(out['labels'][j].reshape(512, 4), down).any(1)              # [512]
        if zone.any():
            d['L1_zone'][j] = err[j, zone].mean()
        d['L1_rest'][j] = err[j, ~zone].mean()

        if sh.any():                                                               # shock nearest the exact one
            i = np.argmin(np.abs(fr['x'][sh] - x_s[j]))
            d['speed'][j] = fr['speed'][sh][i]
            d['u_RH'][j], d['p_RH'][j] = fr['W_RH'][sh][i, 1], fr['W_RH'][sh][i, 2]
    return d


def fig_train_vs_fv(diag, out):
    '''Training cases vs FV: L1, L1 by zone, counts, shock speed, RH star state, all against t.'''
    if not diag:
        return
    fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for name, cases in diag.items():
        for d, ls in zip(cases, ['-', '--', ':']):
            kw = dict(color=COLORS[name], ls=ls)
            lab = f'{name}, r = {d["r"]:.2f}'
            ax[0, 0].semilogy(d['t'], d['L1'], label=lab, **kw)
            ax[0, 1].semilogy(d['t'], d['L1_zone'], marker='o', ms=2, **kw)        # dots: overwritten zone
            ax[0, 1].semilogy(d['t'], d['L1_rest'], **kw)                          # line: everything else
            ax[0, 2].plot(d['t'], d['n_fronts'], marker='o', ms=2, **kw)
            ax[0, 2].plot(d['t'], d['n_shocks'], marker='x', ms=3, **kw)
            ax[1, 0].plot(d['t'], 100 * np.abs(d['speed'] - d['S']) / d['S'], **kw)
            ax[1, 1].plot(d['t'], 100 * np.abs(d['p_RH'] - d['p_star']) / d['p_star'], **kw)
            ax[1, 2].plot(d['t'], 100 * np.abs(d['u_RH'] - d['u_star']) / d['u_star'], **kw)

    titles = ['L1 vs FV (mean over x)', 'L1: overwritten zone (dots) vs rest (line)',
              'fronts (o, expect 2) and shocks (x, expect 1)',
              'shock speed error (%)', 'RH p* error (%)', 'RH u* error (%)']
    for a, title in zip(ax.flat, titles):
        a.axvline(T_MIN, color='k', ls=':', lw=1)
        a.set_title(title, fontsize=10)
    for a in ax[1]:
        a.set_xlabel('t')
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'train_vs_fv.png', dpi=150)


# ======================================================================================
# Master
# ======================================================================================

def load_all(models=None):
    '''Evaluate every arm with a checkpoint. Returns (res, models {name: (model, encoder)}, datasets).'''
    with open(FLAGS_CFG, 'r') as file:
        flags_cfg = yaml.safe_load(file)['jump_flags']

    _, _, test_id, test_ood = load_splits(DATA, t_max=0.24, exact=True)
    r_id, r_ood = test_id.r.numpy(), test_ood.r.numpy()
    keep_id = {int(np.argmin(r_id)), int(np.argsort(r_id)[len(r_id) // 2])}        # low r, mid r (as picks)
    keep_ood = {int(np.argmax(r_ood))}                                             # OOD-far

    res, nets = {}, {}
    for name, (cfg, ckpt) in (models or MODELS).items():
        if not Path(ckpt).exists():
            print(f'skipping {name}: no checkpoint at {ckpt}')
            continue
        print(f'evaluating {name}')
        model, encoder = load_model(cfg, ckpt)
        nets[name] = (model, encoder)
        res[name] = {'test_id': evaluate(model, encoder, test_id, flags_cfg, keep_id),
                     'test_ood': evaluate(model, encoder, test_ood, flags_cfg, keep_ood)}
    return res, nets, (test_id, test_ood, flags_cfg)


def main():
    os.makedirs(OUT, exist_ok=True)
    res, nets, (test_id, _, _) = load_all()
    if not res:
        return

    train = load_splits(DATA)[0]                                    # t <= T_train, FV only
    order = np.argsort(train.r.numpy())
    train_picks = [int(order[len(order) // 2])]                     # mid r; add more indices for more cases
    diag = {name: [train_vs_fv(m, e, train, c) for c in train_picks]
            for name, (m, e) in nets.items() if name in ('full', 'free-jump')}

    # tier 1
    fig_C1_accuracy(res, OUT)
    fig_C2_profiles(res, OUT)
    fig_C3_slope(sweep_l1(test_id), OUT)
    fig_C4_shock(res, OUT)
    fig_C5_split(res, OUT)
    fig_C6_front_vs_t(res, OUT)
    table_T(res, OUT)

    # tier 2
    fig_X1_confusion(res, OUT)
    fig_X2_zone_fields(res, OUT)
    fig_X3_region_map(res, OUT)
    fig_X4_latents({name: latents(e) for name, (_, e) in nets.items()}, OUT)
    fig_X5_swaps(res, OUT)

    fig_train_vs_fv(diag, OUT)


if __name__ == '__main__':
    main()
