'''
Diagnostics for ladder step 1 (plan §9; "Diagnostics round 2", section 0): full model, free-jump arm,
one-region arm. Time-stepper version: the model is a step map W(t_n) -> W(t_n+1).

Core (tier 1): C1 accuracy vs r, C2 profiles, C3 H1 slope, C4 shock accuracy vs r, C5 error split,
C6 front error and L1 vs t (rollout vs one step), C7 jump capturing vs t, C8 cell values at the fronts,
table T, front table F. Explainability (tier 2): X1 zone-to-region
confusion, X2 zone fields, X3 region map with zero contours, X4 latents, X5 oracle swaps. Tier 3 lives in debug.py.

Every case is rolled out from the FV state at t0 = T_MIN = 0.02 through T_train = 0.16 and on to 0.24
(the held-out time band), each step fed the model's own output, as in rollout_L1. The model is queried at
the cell centres, and those values are compared with the FV and exact cell averages as they are.
The exact solution is the reference, FV alongside.
Fronts come from the jump hooks (advected frame at t_n+1), clustered by evaluation.fronts.front_lists.
Every metric that names a front uses the model front matched to the exact wave (plan §9.2).
For the one-region arm the fronts are the faces jump_flags marks on its output.

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

import preprocessing                                    # puts euler1d-data on the path
from infrastructure.results import ExactRiemannSolver
from preprocessing.exact import wave_speeds
from preprocessing.jump_flags import jump_flags
from model.neural_chart import NeuralChart1DST
from model.encoder import FieldEncoder
from model.shock_interface import rankine_hugoniot, normal_mach
from train.dataset import load_splits
from train.helper import points
from evaluation.certificates import RH_residual
from evaluation.fronts import front_lists

ROOT = Path(__file__).parent.parent
DATA = ROOT / 'data' / 'step1_sod_pr.h5'
FLAGS_CFG = ROOT / 'configs' / 'preprocessing_config.yaml'
OUT = ROOT / 'results' / 'diagnostics'
CFG = ROOT / 'configs' / 'models_cfg'
RUNS = ROOT / 'results' / 'models'                  # path.save_path; trainloop saves <name>/trained_model.pt (= stage 3)


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
T_TRAIN, T_MIN, T_END = 0.16, 0.02, 0.24           # T_MIN = t0, the rollout start (data.t0); T_END: held-out band end
DT = 0.002                                          # one step (physics.time.dt)
DX = 1 / 512
CELL = 2 / 512                                      # one cell in x~ units
TOL = 3                                             # cells: a model front this close to an exact wave is a hit
SIDE = 4                                            # cells: one-region side states are read this far off the shock face
XC = (np.arange(512) + 0.5) * DX                                                   # cell centres
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
    enc_cfg = cfg['encoder']
    encoder = FieldEncoder(N_channels = 3,
                           N_grid = enc_cfg['N_grid'],
                           conv_channels = enc_cfg['conv_channels'],
                           kernel = enc_cfg['kernel'],
                           hidden_layers = enc_cfg['encodernet']['hidden_layers'],
                           z_geo = cfg['architecture']['latent']['z_geo'],
                           z_field = cfg['architecture']['latent']['z_field']).to(DEVICE)

    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state['model'])
    encoder.load_state_dict(state['encoder'])
    return model.eval(), encoder.eval()


# ======================================================================================
# Exact solution
# ======================================================================================

def rh_jacobian(S):
    '''[3, 4] d W_RH / d (rho, v, p, s_ud) at the exact upstream state (right state), n_ud = -1.'''
    n = torch.tensor([[-1.0]], dtype=torch.float64)
    f = lambda u: rankine_hugoniot(u[None, :3], n, u[3:], GAMMA)[0]
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
    '''[T, 512] exact region (0..4, REGIONS order) at every cell centre.'''
    edges = X0 + case['speeds'][None, :] * case['t'][:, None]                      # [T, 4]
    return np.stack([np.searchsorted(e, XC) for e in edges])


def start_index(ds):
    '''Index of the first snapshot with t >= T_MIN: the rollout input (as RolloutDataset.n0).'''
    return int((ds.t.numpy() * T_TRAIN < T_MIN - 1e-9).sum())


def cons_to_prim(U):
    rho, m, E = U[..., 0], U[..., 1], U[..., 2]
    v = m / rho
    return np.stack([rho, v, (GAMMA - 1) * (E - 0.5 * rho * v ** 2)], axis=-1)


def rel_l1(W, W_ex):
    '''Relative L1 per variable over all given snapshots and cells, then the mean over (rho, v, p).'''
    ax = tuple(range(W.ndim - 1))
    return (np.abs(W - W_ex).sum(ax) / np.abs(W_ex).sum(ax)).mean()


def rel_l1_t(W, W_ex):
    '''[T] relative L1 of each snapshot, mean over (rho, v, p).'''
    return (np.abs(W - W_ex).sum(-2) / np.abs(W_ex).sum(-2)).mean(-1)


# ======================================================================================
# Running a model: one step and rollouts
# ======================================================================================

def front_summary(fr):
    '''
    Fronts of one snapshot (fronts.front_lists) as numpy, x in data units, plus the upstream and composed
    downstream states, s_ud (advection speed, normal from up to down), s_RH_ud (the speed RH uses, from the pin
    pressure), Mn_all (predicted upstream plateau with the advection speed, every front) and the RH residual of
    the composed jump with the advection speed (nan where not a shock): is the jump consistent with how the
    front moves. RH with its own speed s_RH holds by construction, so it is not reported.
    Every array has one entry per front.
    '''
    up = fr['up_is_a'][:, None]
    W_up, W_dn = torch.where(up, fr['W_a'], fr['W_b']), torch.where(up, fr['W_b'], fr['W_a'])
    n_ud = 2 * fr['up_is_a'].to(W_up) - 1                                         # +1: downstream on the right
    s_ud = fr['S'] * n_ud
    rh = RH_residual(W_up, W_dn, s_ud, n_ud, GAMMA)

    out = {key: val.cpu().numpy() for key, val in fr.items()}
    out.update(x = ((fr['x'] + 1) / 2).cpu().numpy(),                             # x~ -> x
               speed = fr['S'].cpu().numpy(),                                     # advection dx/dt, field units
               speed_RH = fr['S_RH'].cpu().numpy(),                               # speed RH uses (shocks)
               s_RH_ud = (fr['S_RH'] * n_ud).cpu().numpy(),
               W_up = W_up.cpu().numpy(),
               W_dn_comp = W_dn.cpu().numpy(),
               s_ud = s_ud.cpu().numpy(),
               Mn_all = normal_mach(W_up, n_ud[:, None], s_ud, GAMMA).cpu().numpy(),
               rh = torch.where(fr['is_shock'], rh, torch.nan).cpu().numpy())
    return out


@torch.no_grad()
def step(model, encoder, W, x_cell, z=None):
    '''One step from W [B, 512, 3] at t_n, as cell_L1: the model output at t_n+1, plus 'group'.
    z: (z_geo, z_field) [B, d] to use instead of encoder(W) (debug.py swap test and traversals).'''
    W_grid = torch.as_tensor(W).float().to(DEVICE).transpose(1, 2)
    z_geo, z_field = encoder(W_grid) if z is None else z
    x, group = points(x_cell, len(W_grid))
    out = model(x, z_geo[group], z_field[group], W_grid, group)
    out['group'] = group
    return out


def unpack(model, out, B, fronts=True):
    '''One step's output for B rows as numpy: W [B, 512, 3], labels [B, 512], W_regions [B, 512, K, 3],
    phi_hat [B, 512, K] (raw phi, advected, at t_n+1) and fronts (B front_summary dicts). phi_hat and fronts are
    None for one-region; fronts also with fronts=False.'''
    view = lambda v: v.reshape(B, -1, *v.shape[1:]).float().cpu().numpy()
    fr = None
    if fronts and out['jump'] is not None:
        fr = [front_summary(f) for f in
              front_lists(out['jump'], out['group'], B, TOL * CELL, model.jump_mode == 'rh')]
    return {'W': view(out['W']).astype(np.float64),
            'labels': out['labels'].view(B, -1).cpu().numpy(),
            'W_regions': view(out['W_regions']),
            'phi_hat': None if out['phi_hat'] is None else view(out['phi_hat']),
            'fronts': fr}


def one_step(model, encoder, W_in, x_cell, chunk=32, fronts=True, z=None):
    '''One step from every row of W_in [B, 512, 3] (FV data: teacher forcing), chunk rows per call.
    z: (z_geo, z_field) [B, d], as in step. Returns unpack's dict for all B rows.'''
    parts = []
    for c in range(0, len(W_in), chunk):
        zc = None if z is None else (z[0][c:c + chunk], z[1][c:c + chunk])
        out = step(model, encoder, W_in[c:c + chunk], x_cell, z=zc)
        parts.append(unpack(model, out, len(W_in[c:c + chunk]), fronts))
    return {key: None if parts[0][key] is None else
                 sum((p[key] for p in parts), []) if key == 'fronts' else np.concatenate([p[key] for p in parts])
            for key in parts[0]}


def rollout(model, encoder, W0, n_steps, x_cell):
    '''
    n_steps steps from W0 [B, 512, 3] at t0, each fed the model's own output (as rollout_L1), all B cases
    at once. One dict per case: unpack's arrays with a time axis, [T, ...] at t0 + dt .. t0 + T dt (fronts:
    a list of T), plus W_min [T, 3] and e_min [T], the minimum over cells of (rho, v, p) and of p / ((g-1) rho).
    '''
    W, B = W0, len(W0)
    steps = []
    for _ in range(n_steps):
        out = step(model, encoder, W, x_cell)
        W = out['W'].view(B, -1, 3)
        steps.append(unpack(model, out, B))

    runs = []
    for c in range(B):
        run = {key: None if steps[0][key] is None else
                    [s[key][c] for s in steps] if key == 'fronts' else np.stack([s[key][c] for s in steps])
               for key in steps[0]}
        run['W_min'] = run['W'].min(1)
        run['e_min'] = (run['W'][..., 2] / ((GAMMA - 1) * run['W'][..., 0])).min(1)
        runs.append(run)
    return runs


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
    # oracle fields: zone k is exact on every region it owns (is the dominant zone of). A point labelled k
    # in a region k doesn't own (misassigned) gets the exact state of k's owned region nearest in wave
    # order, extended (fan: xi clamped). A zone that owns nothing uses the region it mostly covers.
    table = np.empty((K, 5), int)                                                  # [zone, region] -> region
    for k in range(K):
        owned = np.flatnonzero(zone_of_region == k)
        for R in range(5):
            table[k, R] = owned[np.argmin(np.abs(owned - R))] if len(owned) else region_of_zone[k]
    kmap, t = table[lab, reg], case['t'][use][:, None]
    W_fld = np.zeros(kmap.shape + (3,))
    for k in range(5):
        W_fld[kmap == k] = exact_region(case, k, XC[None], t)[kmap == k]

    W_ex = case['W_ex'][use]
    case['conf'] = conf
    case['swap'] = {'model': rel_l1(case['W'][use], W_ex),
                    'oracle labels': rel_l1(W_lab, W_ex),
                    'oracle fields': rel_l1(W_fld, W_ex)}


def evaluate(model, encoder, ds, flags_cfg, keep=()):
    '''One dict per case of the split: the rollout from t0 = T_MIN to the last snapshot, the one-step
    predictions from the FV data at the same times (W_1step), and the references they are compared with.
    keep: case indices whose W_regions are kept (for X2 and the zone maps); phi_hat is kept for every case.'''
    n0 = start_index(ds)
    t = ds.t[n0 + 1:].numpy() * T_TRAIN
    faces = np.arange(1, 512) * DX
    levelset = model.geo_mode == 'levelset'
    x_cell = ds.x_centers.to(DEVICE)

    runs = rollout(model, encoder, ds.W[:, n0], len(t), x_cell)
    W_1step = one_step(model, encoder, ds.W[:, n0:-1].reshape(-1, 512, 3), x_cell,
                       fronts=False)['W'].reshape(ds.C, len(t), 512, 3)
    cases = []
    for c, out in enumerate(runs):
        r = float(ds.r[c])
        out.update(exact_case(r))
        out.update(r = r, t = t, levelset = levelset, W_1step = W_1step[c],
                   x_ex = X0 + out['speeds'][None, 2:] * t[:, None],              # [T, 2] contact, shock
                   W_ex = cons_to_prim(ds.U_exact[c, n0 + 1:].double().numpy()),
                   W_fv = ds.W[c, n0 + 1:].double().numpy())
        if levelset:
            zone_stats(out)
        else:                            # one-region: its fronts are the faces jump_flags marks on its output
            out['flags'] = jump_flags(out['W'], t, GAMMA, flags_cfg)
            out['fronts'] = [{'x': faces[f == 1]} for f in out['flags']]
        if c not in keep:
            out.pop('W_regions')
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
    '''[T, 2] |x_hat - x_exact| / dx at (contact, shock), x_hat where rho crosses half way between its plateaus
    (front_quality): from the field, not the labels, so it also works for one-region and is not fooled by
    labels that sit apart from the jump they label.'''
    return np.abs(front_quality(case, 'W')['pos'])


def front_counts(case, band):
    '''Per snapshot (t >= T_MIN, in band [T]): missed exact waves, spurious model fronts, is_shock hits / tries.'''
    missed = spurious = right = tried = 0
    use = np.flatnonzero((case['t'] >= T_MIN) & band)
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


def shock_metrics(case, t_lo=T_MIN, t_hi=T_TRAIN):
    '''
    C4 on the front matched to the exact shock, t_lo <= t <= t_hi. Same estimator for every arm:
    line-fit speed; Mn from that speed and the cell average SIDE cells ahead of the shock; downstream
    state = the cell average SIDE cells behind. Level-set arms also get the values the jump module uses
    (zone fields read at the front): local = advection speed s, local_rh = the speed RH uses (from the pin
    pressure), Mn_rh = Mn of the predicted upstream state at the advection speed, down_rh = composed
    downstream state. Errors are relative except Mn.
    '''
    t, S = case['t'], case['speeds'][3]
    idx = case['match']['idx'][:, 1]
    use = np.flatnonzero((t >= t_lo) & (t <= t_hi + 1e-9) & (idx >= 0))
    out = dict(fit=np.nan, local=np.nan, local_rh=np.nan, Mn=np.nan, down=np.nan, Mn_rh=np.nan, down_rh=np.nan)
    if len(use) < 2:
        return out

    x_s = np.array([case['fronts'][j]['x'][idx[j]] for j in use])
    S_fit = np.polyfit(t[use], x_s, 1)[0]
    face = np.rint(x_s / DX).astype(int) - 1                                       # face nearest the front
    W_up = case['W'][use, np.minimum(face + 1 + SIDE, 511)]
    W_dn = case['W'][use, np.maximum(face - SIDE, 0)]
    Mn = (S_fit - W_up[:, 1]) / np.sqrt(GAMMA * W_up[:, 2] / W_up[:, 0])

    W_ex = case['W_star_R']
    down = lambda W: np.median(np.abs(W - W_ex).sum(-1) / np.abs(W_ex).sum())
    out.update(fit = abs(S_fit - S) / S, Mn = np.median(np.abs(Mn - case['Mn_ex'])), down = down(W_dn))

    if case['levelset']:
        fr = [case['fronts'][j] for j in use]
        i = idx[use]
        out.update(local = np.median([abs(f['speed'][k] - S) / S for f, k in zip(fr, i)]),
                   local_rh = np.nanmedian([abs(f['speed_RH'][k] - S) / S for f, k in zip(fr, i)]),
                   Mn_rh = np.median([abs(f['Mn_all'][k] - case['Mn_ex']) for f, k in zip(fr, i)]),
                   down_rh = down(np.array([f['W_dn_comp'][k] for f, k in zip(fr, i)])))
    return out


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
    Predicted = linearized RH map: |J [W_up - W_R; s_RH_ud - s_exact]|, summed over (rho, v, p), with the speed
    RH actually uses (from the pin pressure), not the advection speed.'''
    fr, i = case['fronts'][j], case['match']['idx'][j, 1]
    measured = np.abs(fr['W_dn_comp'][i] - case['W_star_R']).sum()
    du = np.concatenate([fr['W_up'][i] - RIGHT, [fr['s_RH_ud'][i] + case['speeds'][3]]])
    return measured, np.abs(case['J'] @ du).sum()


def snapshots(case, t_lo=T_MIN, t_hi=T_TRAIN):
    '''Snapshot indices with t_lo <= t <= t_hi where both exact waves have a matched front.'''
    t = case['t']
    return np.flatnonzero((t >= t_lo) & (t <= t_hi + 1e-9) & (case['match']['idx'] >= 0).all(1))


WAVES = ['contact', 'shock']


def front_quality(case, key, inner=3, outer=10):
    '''
    Jump capturing at the exact contact and shock, from cell values alone, so every arm, the FV data and
    the exact cell averages are measured the same way (key = 'W', 'W_fv' or 'W_ex'). Per snapshot and wave
    ([T, 2], nan where the waves are too close to read both plateaus):
        pos    signed position error (cells, + = ahead): where rho crosses half way between its plateaus,
               against the exact wave
        width  cells whose rho is 10% to 90% of the way across (exact cell averages: 0 or 1)
        jump   |plateau jump - exact jump| / |exact jump|; rho for the contact, mean over (rho, v, p) for the shock
        over   overshoot beyond either plateau, as a fraction of the jump
    Plateaus = median of the cells inner..outer cells from the exact wave on each side, outer cut to half
    the distance to the neighbouring wave. Cached in case['fq'][key].
    '''
    if key in case.setdefault('fq', {}):
        return case['fq'][key]
    W, t, sol = case[key], case['t'], case['sol']
    J_ex = [np.array([sol.rho_star_R - sol.rho_star_L]), np.array(RIGHT) - case['W_star_R']]
    chans = [[0], [0, 1, 2]]
    out = {k: np.full((len(t), 2), np.nan) for k in ('pos', 'width', 'jump', 'over')}
    for j, tj in enumerate(t):
        x = X0 + case['speeds'] * tj                                               # head, tail, contact, shock
        for w in range(2):
            xw = x[2 + w]
            gap_l, gap_r = (xw - x[1 + w]) / DX, ((x[3] - xw) / DX if w == 0 else (1 - xw) / DX)
            lo, hi = min(outer, gap_l / 2), min(outer, gap_r / 2)
            if min(lo, hi) < inner + 2:
                continue
            d = (XC - xw) / DX                                                     # signed distance in cells
            W_l = np.median(W[j, (d <= -inner) & (d >= -lo)], 0)
            W_r = np.median(W[j, (d >= inner) & (d <= hi)], 0)
            c = chans[w]
            out['jump'][j, w] = np.mean(np.abs((W_r - W_l)[c] - J_ex[w]) / np.abs(J_ex[w]))

            win = (d >= -lo) & (d <= hi)
            frac = (W[j, win, 0] - W_l[0]) / (W_r[0] - W_l[0])
            out['width'][j, w] = ((frac > 0.1) & (frac < 0.9)).sum()
            out['over'][j, w] = max(0.0, -frac.min(), frac.max() - 1)
            dw = d[win]
            k = np.flatnonzero((frac[:-1] - 0.5) * (frac[1:] - 0.5) <= 0)           # half-way crossings
            if len(k):
                k = k[np.argmin(np.abs(dw[k]))]                                    # the one nearest the exact wave
                f0, f1 = frac[k], frac[k + 1]
                out['pos'][j, w] = dw[k] + (0.5 - f0) / (f1 - f0) if f1 != f0 else dw[k]
    case['fq'][key] = out
    return out


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
    '''C1: relative L1 of the rollout against exact vs r, t <= T_train, ID band shaded, FV-vs-exact as the floor.'''
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


def fig_C2_profiles(res, out, t_plot=T_TRAIN):
    '''C2: rho, v, p of the rollout at t = t_plot for low r, mid r and OOD-far.'''
    fig, ax = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
    for col, (split, c, title) in enumerate(picks(res)):
        ref = next(iter(res.values()))[split][c]
        j = int(np.argmin(np.abs(ref['t'] - t_plot)))
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
    fig.suptitle(f't = {t_plot}' + (' (held-out time band)' if t_plot > T_TRAIN + 1e-9 else ''))
    fig.tight_layout()
    fig.savefig(out / f'C2_profiles_t{t_plot}.png', dpi=150)


def sweep_l1(test_id):
    '''C3: arm -> {d: [test_id relative L1 per case]}, rollout t0 -> T_train, for the sweep checkpoints that exist.'''
    n0 = start_index(test_id)
    n_steps = int((test_id.t[n0 + 1:] <= 1 + 1e-6).sum())          # t~ <= 1, i.e. t <= T_train
    out = {}
    for arm, sweep in SWEEP.items():
        for d, (cfg, ckpt) in sweep.items():
            if not Path(ckpt).exists():
                continue
            model, encoder = load_model(cfg, ckpt)
            runs = rollout(model, encoder, test_id.W[:, n0], n_steps, test_id.x_centers.to(DEVICE))
            out.setdefault(arm, {})[d] = [
                rel_l1(run['W'], cons_to_prim(test_id.U_exact[c, n0 + 1:n0 + 1 + n_steps].double().numpy()))
                for c, run in enumerate(runs)]
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
    '''C4: (a) shock-speed error: line fit (solid); level-set arms also the advection speed s (dashed) and the
    speed RH uses (dotted), (b) Mn error, (c) downstream-state error, against r.'''
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name in res:
        cases = sorted(all_cases(res, name), key=lambda c: c['r'])
        r = [c['r'] for c in cases]
        m = [shock_metrics(c) for c in cases]
        ax[0].plot(r, [100 * x['fit'] for x in m], 'o-', ms=3, color=COLORS[name], label=name)
        ax[1].plot(r, [x['Mn'] for x in m], 'o-', ms=3, color=COLORS[name])
        ax[2].plot(r, [100 * x['down'] for x in m], 'o-', ms=3, color=COLORS[name])
        if cases[0]['levelset']:
            ax[0].plot(r, [100 * x['local'] for x in m], '--', lw=1, color=COLORS[name], label=f'{name}, advection s')
            ax[0].plot(r, [100 * x['local_rh'] for x in m], ':', lw=1, color=COLORS[name], label=f'{name}, RH speed')
            ax[1].plot(r, [x['Mn_rh'] for x in m], '--', lw=1, color=COLORS[name], label=f'{name}, zone fields, advection s')
            ax[2].plot(r, [100 * x['down_rh'] for x in m], '--', lw=1, color=COLORS[name])
    labels = ['shock-speed error, line fit (%)', '|Mn - Mn_exact| (median over t; fit speed, cells)',
              'downstream-state error (%, median over t; cells)']
    for a, lab in zip(ax, labels):
        a.axvspan(2, 10, color='0.9')
        a.set_xscale('log')
        a.set_yscale('log')
        a.set_xlabel('r')
        a.set_title(lab, fontsize=10)
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=7)
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
    '''C6: against t, mean over cases; ID and OOD. Top: front-position error from the field (front_errors),
    mean over (contact, shock), every arm.
    Bottom: relative L1 vs exact of the rollout (solid) and of one step from the FV data (dotted): the gap
    is the error the rollout accumulates. FV vs exact as the floor.'''
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey='row')
    for col, split in enumerate(('test_id', 'test_ood')):
        for name in res:
            cases = res[name][split]
            t = cases[0]['t']
            err = np.nanmean([front_errors(c).mean(-1) for c in cases], axis=0)
            ax[0, col].plot(t, err, color=COLORS[name], label=name)
            ax[1, col].semilogy(t, np.mean([rel_l1_t(c['W'], c['W_ex']) for c in cases], 0),
                                color=COLORS[name], label=f'{name}, rollout')
            ax[1, col].semilogy(t, np.mean([rel_l1_t(c['W_1step'], c['W_ex']) for c in cases], 0),
                                ':', color=COLORS[name], label=f'{name}, one step')
        ax[1, col].semilogy(t, np.mean([rel_l1_t(c['W_fv'], c['W_ex']) for c in cases], 0), 'k--', lw=1, label='FV')
        for a in ax[:, col]:
            a.axvline(T_TRAIN, color='k', ls=':', lw=0.8)
        ax[0, col].set_title(f'{split} (right of the dotted line: held-out time band)', fontsize=10)
        ax[1, col].set_xlabel('t')
    ax[0, 0].set_ylabel('front-position error (cells)')
    ax[1, 0].set_ylabel('relative L1 vs exact')
    ax[0, 0].legend(fontsize=8)
    ax[1, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'C6_front_error_vs_t.png', dpi=150)


def table_T(res, out):
    '''Certificates and front matching per arm, split and time band, then the matched-ID-L1 table [D60].
    Bands: t <= T_train and t > T_train (held out). RH residual over detected shocks. Lax viol. = fraction
    of fronts matched to the exact shock whose Mn_all (zone fields, advection s) is <= 1: the shock test
    gates on Mn > Mn_gate, so the gated Mn itself cannot violate Lax.'''
    cols = ['RH median', 'RH max', 'Lax viol.', 'min rho', 'min p', 'min e', 'neg cells',
            'missed/snap', 'spur./snap', 'is_shock acc', 'adv. s err']
    fmts = ['.2e', '.2e', '.3f', '.4f', '.4f', '.4f', '.0f', '.2f', '.2f', '.3f', '.2e']
    lines = [f'{"model":<12}{"split":<10}{"band":<10}' + ''.join(f'{c:>13}' for c in cols)]
    fmt = lambda v, f: f'{"-":>13}' if v is None or np.isnan(v) else f'{v:>13{f}}'
    bands = [('t<=T', lambda t: t <= T_TRAIN + 1e-9, (T_MIN, T_TRAIN)),
             ('t>T', lambda t: t > T_TRAIN + 1e-9, (T_TRAIN + 1e-9, np.inf))]

    for name in res:
        for split in ('test_id', 'test_ood'):
            for band, sel, (t_lo, t_hi) in bands:
                cases = res[name][split]
                b = [sel(c['t']) for c in cases]
                W_min = np.min([c['W_min'][bi].min(0) for c, bi in zip(cases, b)], axis=0)
                e_min = np.min([c['e_min'][bi].min() for c, bi in zip(cases, b)])
                neg = sum(((c['W'][bi][..., 0] <= 0) | (c['W'][bi][..., 2] <= 0)).sum() for c, bi in zip(cases, b))
                counts = np.array([front_counts(c, bi) for c, bi in zip(cases, b)])
                rh = Mn = acc = local = None
                if cases[0]['levelset']:
                    fr = [f for c, bi in zip(cases, b) for f, keep in zip(c['fronts'], bi) if keep]
                    rh = np.concatenate([f['rh'][f['is_shock']] for f in fr])
                    Mn = np.array([c['fronts'][j]['Mn_all'][c['match']['idx'][j, 1]] for c, bi in zip(cases, b)
                                   for j in np.flatnonzero(bi & (c['match']['idx'][:, 1] >= 0))])
                    acc = counts[:, 2].sum() / max(counts[:, 3].sum(), 1)
                    local = np.nanmedian([shock_metrics(c, t_lo, t_hi)['local'] for c in cases])
                row = [np.median(rh) if rh is not None and len(rh) else None,
                       rh.max() if rh is not None and len(rh) else None,
                       (Mn <= 1).mean() if Mn is not None and len(Mn) else None,
                       W_min[0], W_min[2], e_min, neg, counts[:, 0].mean(), counts[:, 1].mean(), acc, local]
                lines.append(f'{name:<12}{split:<10}{band:<10}' + ''.join(fmt(v, f) for v, f in zip(row, fmts)))

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


def fig_C7_fronts(res, out, split):
    '''
    C7: jump capturing along the rollout, median over the split's cases, against t: |position error|, width,
    jump-size error and overshoot at the contact (top) and the shock (bottom), every arm with the FV data
    (grey) and the exact cell averages (black) measured the same way (front_quality). A sharp, well-placed
    front sits near the black line in every column; the FV line is how much the training data smears.
    '''
    cols = [('pos', '|position error| (cells)', np.abs), ('width', 'width, 10-90% (cells)', None),
            ('jump', 'jump-size error (%)', lambda v: 100 * v), ('over', 'overshoot (% of jump)', lambda v: 100 * v)]
    fig, ax = plt.subplots(2, 4, figsize=(18, 7), sharex=True)
    ref = res[next(iter(res))][split]
    lines = [(name, res[name][split], 'W', dict(color=COLORS[name], lw=1.2)) for name in res] + \
            [('FV', ref, 'W_fv', dict(color='0.5', lw=1, ls='--')), ('exact', ref, 'W_ex', dict(color='k', lw=0.8, ls=':'))]
    for name, cases, key, kw in lines:
        q = [front_quality(c, key) for c in cases]
        for (m, title, f), col in zip(cols, range(4)):
            v = np.stack([x[m] for x in q])                                        # [cases, T, 2]
            v = f(v) if f else v
            for w in range(2):
                ax[w, col].plot(cases[0]['t'], np.nanmedian(v[..., w], 0), label=name, **kw)
                ax[w, col].set_title(f'{WAVES[w]}: {title}', fontsize=10)
    for a in ax.flat:
        a.axvline(T_TRAIN, color='k', ls=':', lw=0.8)
    for a in ax[1]:
        a.set_xlabel('t')
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(f'{split}: jump capturing along the rollout (right of the dotted line: held-out time band)')
    fig.tight_layout()
    fig.savefig(out / f'C7_fronts_{split}.png', dpi=150)


def fig_C8_front_zoom(res, out, half=12):
    '''C8: rho (contact, shock) and p (shock) as cell values within half cells of the exact wave, mid-r case,
    at T_train and at the end of the held-out band. Steps show each cell, so smearing and misplacement are visible.'''
    split, c, title = picks(res)[1]
    ref = res[next(iter(res))][split][c]
    panels = [(0, 0, 'contact, rho'), (1, 0, 'shock, rho'), (1, 2, 'shock, p')]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for row, tt in enumerate((T_TRAIN, T_END)):
        j = int(np.argmin(np.abs(ref['t'] - tt)))
        for col, (w, v, lab) in enumerate(panels):
            xw = X0 + ref['speeds'][2 + w] * ref['t'][j]
            near = np.abs(XC - xw) <= half * DX
            d = (XC[near] - xw) / DX
            a = ax[row, col]
            a.plot(d, ref['W_ex'][j, near, v], 'k-', drawstyle='steps-mid', lw=1, label='exact')
            a.plot(d, ref['W_fv'][j, near, v], '--', color='0.5', drawstyle='steps-mid', lw=1, label='FV')
            for name in res:
                a.plot(d, res[name][split][c]['W'][j, near, v], color=COLORS[name], drawstyle='steps-mid',
                       lw=1.2, label=name)
            a.axvline(0, color='k', lw=0.5)
            a.set_title(f'{lab}, t = {ref["t"][j]:.2f}', fontsize=10)
            a.set_xlabel('cells from the exact wave')
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(f'{title} (r = {ref["r"]:.2f}): cell values at the fronts')
    fig.tight_layout()
    fig.savefig(out / 'C8_front_zoom.png', dpi=150)


def table_F(res, out):
    '''Front table: median over cases and snapshots of front_quality, per arm (and FV, exact), split,
    time band and wave. Written to fronts.txt.'''
    bands = [('t<=T', lambda t: t <= T_TRAIN + 1e-9), ('t>T', lambda t: t > T_TRAIN + 1e-9)]
    lines = [f'{"":<12}{"split":<10}{"band":<7}{"wave":<9}{"|pos| cells":>12}{"width cells":>12}'
             f'{"jump err %":>12}{"overshoot %":>12}']
    first = res[next(iter(res))]
    rows = [(name, res[name], 'W') for name in res] + [('FV', first, 'W_fv'), ('exact', first, 'W_ex')]
    for name, arm, key in rows:
        for split in ('test_id', 'test_ood'):
            for band, sel in bands:
                q = [(front_quality(c, key), sel(c['t'])) for c in arm[split]]
                for w in range(2):
                    v = {m: np.concatenate([x[m][b, w] for x, b in q]) for m in ('pos', 'width', 'jump', 'over')}
                    med = lambda a: np.nanmedian(a) if np.isfinite(a).any() else np.nan
                    lines.append(f'{name:<12}{split:<10}{band:<7}{WAVES[w]:<9}{med(np.abs(v["pos"])):>12.2f}'
                                 f'{med(v["width"]):>12.1f}{100 * med(v["jump"]):>12.1f}{100 * med(v["over"]):>12.1f}')
    text = '\n'.join(['jump capturing at the exact contact and shock, median over cases and snapshots '
                      '(front_quality; the exact cell averages are the floor)'] + lines)
    print(text)
    (out / 'fronts.txt').write_text(text + '\n')


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
            Wr, lab = case['W_regions'][j], case['labels'][j]                      # [512, K, 3], [512]
            for row, var in enumerate(['rho', 'v', 'p']):
                a = ax[row, col]
                a.plot(XC, case['W_ex'][j, :, row], 'k-', lw=2, alpha=0.3, label='exact')
                for k in range(Wr.shape[1]):
                    a.plot(XC, Wr[:, k, row], '--', lw=0.7, color=f'C{k}')
                    a.plot(XC, np.where(lab == k, Wr[:, k, row], np.nan), '-', lw=1.5, color=f'C{k}', label=f'zone {k}')
                a.set_ylabel(var)
            ax[0, col].set_title(f'{title} (r = {case["r"]:.2f})')
            ax[-1, col].set_xlabel('x')
        ax[0, 0].legend(fontsize=7)
        fig.suptitle(f'{name}: zone fields at t = {T_TRAIN}')
        fig.tight_layout()
        fig.savefig(out / f'X2_zone_fields_{name}.png', dpi=150)


def fig_X3_region_map(res, out):
    '''X3: rollout zones in (x, t) with the zone boundaries (black) and the exact waves (dashed).'''
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
            for k in range(case['phi_hat'].shape[-1]):                         # phi is raw: its zero level is not a front
                a.contour(XC, t, (case['labels'] == k).astype(float), levels=[0.5], colors='k', linewidths=0.6)
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
def latents(encoder, cases, full=False, folds=5, lam=1e-3):
    '''
    X4: encoder latents of the FV state at t = T_train, one per case, against r; and the ridge R^2 of
    z_geo -> the 4 exact wave positions, over every FV snapshot with t <= T_train of these cases
    (folds split by case). z_geo encodes the current field, so it should carry where the waves are now.
    full=True (one-region arm, where z_geo and z_field are one conditioning vector): the probe and the
    PCA use the whole latent [z_geo; z_field], returned as z_geo with z_field_pca = None.
    '''
    cases = sorted(cases, key=lambda c: c['r'])
    use = cases[0]['t'] <= T_TRAIN + 1e-9
    j = int(np.flatnonzero(use)[-1])                                                # snapshot at T_train
    W = torch.tensor(np.stack([c['W_fv'][use] for c in cases]), dtype=torch.float32, device=DEVICE)
    C, T = W.shape[:2]
    z_geo, z_field = encoder(W.flatten(0, 1).transpose(1, 2))
    z_geo, z_field = z_geo.cpu().numpy(), z_field.cpu().numpy()
    if full:
        z_geo = np.hstack([z_geo, z_field])
    pos = np.concatenate([X0 + c['speeds'][None] * c['t'][use][:, None] for c in cases])   # [C*T, 4]

    X = np.hstack([z_geo, np.ones((len(z_geo), 1))])
    pred = np.zeros_like(pos)
    case_of = np.repeat(np.arange(C), T)
    for test in np.array_split(np.random.default_rng(0).permutation(C), min(folds, C)):
        te = np.isin(case_of, test)
        A = X[~te]
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ pos[~te])
        pred[te] = X[te] @ w
    R2 = 1 - ((pred - pos) ** 2).sum(0) / ((pos - pos.mean(0)) ** 2).sum(0)

    at = lambda z: z.reshape(C, T, -1)[:, j]                                        # [C, d] at T_train
    top2 = lambda z: (z - z.mean(0)) @ np.linalg.svd(z - z.mean(0), full_matrices=False)[2][:2].T
    r = np.array([c['r'] for c in cases])
    if full:                                                                       # one panel: PCA of the whole latent
        return dict(r=r, z_geo=top2(at(z_geo)), z_field_pca=None, R2=R2, full=True)
    return dict(r=r, z_geo=at(z_geo), z_field_pca=top2(at(z_field)), R2=R2, full=False)


def fig_X4_latents(lat, out):
    '''X4: z_geo components and the top 2 PCA directions of z_field against r (FV state at T_train), per arm.'''
    fig, ax = plt.subplots(len(lat), 2, figsize=(11, 3.5 * len(lat)), squeeze=False)
    for i, (name, d) in enumerate(lat.items()):
        panels = ([(ax[i, 0], 'z_geo', 'whole latent [z_geo; z_field], top 2 PCA')] if d['full'] else
                  [(ax[i, 0], 'z_geo', 'z_geo'), (ax[i, 1], 'z_field_pca', 'z_field, top 2 PCA')])
        if d['full']:
            ax[i, 1].axis('off')
        for a, key, title in panels:
            a.axvspan(2, 10, color='0.9')
            for k in range(d[key].shape[1]):
                a.plot(d['r'], d[key][:, k], lw=1, label=f'{k}')
            a.set_xscale('log')
            a.set_xlabel('r')
            a.set_title(f'{name}: {title}', fontsize=10)
        R2 = ', '.join(f'{w} {v:.3f}' for w, v in zip(['head', 'tail', 'contact', 'shock'], d['R2']))
        src = 'whole latent' if d['full'] else 'z_geo'
        ax[i, 0].legend(fontsize=7, title=f'R² {src}→position: {R2}', title_fontsize=7)
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
    One training case against its FV data: the rollout from t0 to T_train.
    Overwritten zone = downstream zone of each shock; a cell is in it if its label is.
    '''
    n0 = start_index(ds)
    r, t = float(ds.r[c]), ds.t[n0 + 1:].numpy() * T_TRAIN
    out = rollout(model, encoder, ds.W[c:c + 1, n0], len(t), ds.x_centers.to(DEVICE))[0]
    err = np.abs(out['W'] - ds.W[c, n0 + 1:].double().numpy()).sum(-1)              # [T, 512]

    S = exact_waves(r)[0][3]
    sol = ExactRiemannSolver((1.0, 0.0, 0.1 * r), RIGHT, x_interface=X0, gamma=GAMMA)
    x_s = X0 + S * t

    d = {k: np.full(len(t), np.nan) for k in ('L1_zone', 'L1_rest', 'speed', 'speed_RH', 'u_RH', 'p_RH')}
    d.update(r=r, t=t, S=S, u_star=sol.u_star, p_star=sol.P_star, L1=err.mean(1),
             n_fronts=np.zeros(len(t), int), n_shocks=np.zeros(len(t), int))

    for j, fr in enumerate(out['fronts']):
        d['n_fronts'][j], d['n_shocks'][j] = len(fr['x']), fr['is_shock'].sum()
        sh = fr['is_shock']
        down = np.where(fr['up_is_a'], fr['b'], fr['a'])[sh]
        zone = np.isin(out['labels'][j], down)                                     # [512]
        if zone.any():
            d['L1_zone'][j] = err[j, zone].mean()
        d['L1_rest'][j] = err[j, ~zone].mean()

        if sh.any():                                                               # shock nearest the exact one
            i = np.argmin(np.abs(fr['x'][sh] - x_s[j]))
            d['speed'][j], d['speed_RH'][j] = fr['speed'][sh][i], fr['speed_RH'][sh][i]
            d['u_RH'][j], d['p_RH'][j] = fr['W_RH'][sh][i, 1], fr['W_RH'][sh][i, 2]
    return d


def fig_train_vs_fv(diag, out):
    '''Training cases vs FV, rollout t0 -> T_train: L1, L1 by zone, counts, shock speed, RH star state, against t.'''
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
            ax[1, 0].plot(d['t'], 100 * np.abs(d['speed_RH'] - d['S']) / d['S'], color=COLORS[name], ls=':', lw=0.8)
            ax[1, 1].plot(d['t'], 100 * np.abs(d['p_RH'] - d['p_star']) / d['p_star'], **kw)
            ax[1, 2].plot(d['t'], 100 * np.abs(d['u_RH'] - d['u_star']) / d['u_star'], **kw)

    titles = ['L1 vs FV (mean over x)', 'L1: overwritten zone (dots) vs rest (line)',
              'fronts (o, expect 2) and shocks (x, expect 1)',
              'shock speed error (%): advection s (line), RH speed (dotted)', 'RH p* error (%)', 'RH u* error (%)']
    for a, title in zip(ax.flat, titles):
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

    _, _, test_id, test_ood = load_splits(DATA, t_max=T_END, exact=True)
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
    fig_C2_profiles(res, OUT, T_TRAIN)
    fig_C2_profiles(res, OUT, T_END)
    fig_C3_slope(sweep_l1(test_id), OUT)
    fig_C4_shock(res, OUT)
    fig_C5_split(res, OUT)
    fig_C6_front_vs_t(res, OUT)
    fig_C7_fronts(res, OUT, 'test_id')
    fig_C7_fronts(res, OUT, 'test_ood')
    fig_C8_front_zoom(res, OUT)
    table_T(res, OUT)
    table_F(res, OUT)

    # tier 2
    fig_X1_confusion(res, OUT)
    fig_X2_zone_fields(res, OUT)
    fig_X3_region_map(res, OUT)
    fig_X4_latents({name: latents(e, all_cases(res, name), full=m.geo_mode == 'none')
                    for name, (m, e) in nets.items()}, OUT)
    fig_X5_swaps(res, OUT)

    fig_train_vs_fv(diag, OUT)


if __name__ == '__main__':
    main()
