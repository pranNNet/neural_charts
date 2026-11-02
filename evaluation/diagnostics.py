'''
Diagnostics for the localfield model: the figures that test the thesis. No oracle, no cross-model comparison.
Three arms: full (level sets + RH), free-jump, one-region. The profile animation is evaluation.animate.

    python -m evaluation.diagnostics                       (from the repo root: every figure)
    python -m evaluation.diagnostics --figs levelset rh    (a subset; names below)
    python -m evaluation.diagnostics --case 3 --split test_ood
    python -m evaluation.diagnostics --splits val --runs results/models_setB --out results/diagnostics_setB_val

The model is compared with the exact cell averages stored in the data (U_exact); the FV data is drawn alongside
as the floor. The held-out time band (t > 0.16) is shaded in every time plot.

Figures (results/diagnostics/):
    accuracy     accuracy_<split>.png + .csv   rollout rel. L1 vs t: rho, v, p stacked, arms and FV
    regions      region_fields.png             each zone's field W_k over the WHOLE domain (full arm): solid
                                               where the zone owns the cell, faint where it is an extension
    levelset     levelset.png                  one plot: the full arm's psi (median, in cells) against signed
                                               distance to the exact front, contact and shock, with the target
    fronts       fronts.png                    contact and shock position error vs t, from the field (rho
                                               crosses half way between the exact plateaus): same rule for
                                               every arm and the FV data
    rh           rh.png                        at the shock: RH residual of the states either side with the
                                               measured speed, shock speed error, post-shock state error
    basis        basis.png                     how the FieldNet bases are used: singular values of the per-basis
                                               contributions a_{k(x)m}(x) b_m(x) to rho, and their top modes
                                               (uses the FieldNet return_parts hook in model/region.py)
    zonemap      zonemap_<split>_<case>.png    x-t map of the zone labels with the psi = 0 fronts and the exact
                                               wave paths (level-set arms)
'''

import argparse
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt

from model.neural_chart import sample_grid, cell_offset
from train.helper import points
from train.fronts import KAPPA, CLIP
import evaluation.helper as H
from evaluation.helper import (COLORS, SPLITS, ALL_SPLITS, RIGHT, T_TRAIN, N, DX, XC, X0, SIDE, WAVES, VARS, GAMMA,
                               DEVICE, Split, available_models, rollouts, held_out, save)


# ======================================================================================
# Measurements (the same rule for every arm and the FV data)
# ======================================================================================

def rel_l1_t(W, W_ex):
    '''[..., T] relative L1 of each snapshot over the cells, mean over (rho, v, p).'''
    return (np.abs(W - W_ex).sum(-2) / np.abs(W_ex).sum(-2)).mean(-1)


def front_position(rho, x_ex, level, search=15):
    '''Where rho [512] crosses `level` (linear between centres), the crossing nearest x_ex within +-search cells;
    nan if none. level is half way between the two exact plateaus, so the rule needs no labels and works for
    every arm, the FV data and the exact cell averages alike.'''
    i = int(np.clip(round(x_ex / DX - 0.5), search, N - 1 - search))
    f = rho[i - search:i + search + 1] - level
    k = np.flatnonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)
    if not len(k):
        return np.nan
    xs = XC[i - search + k] + DX * f[k] / (f[k] - f[k + 1])
    return xs[np.argmin(np.abs(xs - x_ex))]


def fronts_of(W, sp, c, t):
    '''[T, 2] measured (contact, shock) positions of W [T, 512, 3] for case c at times t.'''
    e = sp.exact[c]
    levels = (0.5 * (e['W_star_L'][0] + e['W_star_R'][0]), 0.5 * (e['W_star_R'][0] + RIGHT[0]))
    x_ex = sp.x_exact(c, t)
    return np.array([[front_position(W[j, :, 0], x_ex[j, w], levels[w]) for w in range(2)] for j in range(len(t))])


def euler_flux(W):
    rho, v, p = W[..., 0], W[..., 1], W[..., 2]
    E = p / (GAMMA - 1) + 0.5 * rho * v * v
    return np.stack([rho * v, rho * v * v + p, v * (E + p)], -1), np.stack([rho, rho * v, E], -1)


def rh_residual(W_up, W_dn, s):
    '''|F(U_dn) - F(U_up) - s (U_dn - U_up)| / |F(U_up)|, lab-frame speed s. 0 for a jump that satisfies RH.'''
    (F_up, U_up), (F_dn, U_dn) = euler_flux(W_up), euler_flux(W_dn)
    num = np.linalg.norm(F_dn - F_up - s[..., None] * (U_dn - U_up), axis=-1)
    return num / np.linalg.norm(F_up, axis=-1)


def shock_terms(W, x_s, t, sp, c, half=5):
    '''At the measured shock x_s [T]: speed s (line fit over +-half steps), the states SIDE cells either side,
    and the three errors [T]: RH residual with the measured speed, |s - S| / S, post-shock state error.'''
    T = len(t)
    s = np.full(T, np.nan)
    for j in range(T):
        sl = slice(max(0, j - half), min(T, j + half + 1))
        ok = np.isfinite(x_s[sl])
        if ok.sum() >= 3:
            s[j] = np.polyfit(t[sl][ok], x_s[sl][ok], 1)[0]
    i = np.where(np.isfinite(x_s), np.floor(np.nan_to_num(x_s) / DX), 0).astype(int)
    W_up = W[np.arange(T), np.clip(i + 1 + SIDE, 0, N - 1)]
    W_dn = W[np.arange(T), np.clip(i - SIDE, 0, N - 1)]
    e = sp.exact[c]
    bad = ~np.isfinite(x_s)
    rh = np.where(bad, np.nan, rh_residual(W_up, W_dn, s))
    speed = np.abs(s - e['speeds'][3]) / e['speeds'][3]
    down = np.where(bad, np.nan, np.abs(W_dn - e['W_star_R']).sum(-1) / np.abs(e['W_star_R']).sum())
    return rh, speed, down


def zone_pair(cls, x):
    '''Zones (a, b) either side of position x from the data classes [512] (-1 = unlabelled): the nearest
    labelled cell on each side.'''
    lab = np.flatnonzero(cls >= 0)
    left, right = lab[XC[lab] < x], lab[XC[lab] > x]
    if not len(left) or not len(right):
        return None
    a, b = cls[left[-1]], cls[right[0]]
    return (int(a), int(b)) if a != b else None


# ======================================================================================
# Accuracy
# ======================================================================================

def fig_accuracy(runs, splits):
    """Rollout error against exact, one figure per split: rho, v, p stacked, rel. L1 of each variable vs t, mean
    over the cases. Arms and the FV data; the held-out time band shaded. Also writes accuracy.csv (mean over the
    variables, training window and held-out band)."""
    rows = []
    for name, sp in splits.items():
        t = sp.t[sp.n0 + 1:]
        W_ex = sp.W_ex[:, sp.n0 + 1:]
        series = {a: R['W'] for a, R in runs[name].items()}
        series['FV'] = sp.W[:, sp.n0 + 1:]
        fig, ax = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
        win, band = t <= T_TRAIN + 1e-9, t > T_TRAIN + 1e-9
        for a, W in series.items():
            err = np.abs(W - W_ex).sum(-2) / np.abs(W_ex).sum(-2)                 # [C, T, 3]
            for k, var in enumerate(VARS):
                ax[k].plot(t, err[..., k].mean(0), '--' if a == 'FV' else '-', color=COLORS[a], label=a)
            L = err.mean(-1)
            rows.append([name, a, L[:, win].mean(), L[:, band].mean()])
        for k, var in enumerate(VARS):
            held_out(ax[k])
            ax[k].set(ylabel=f'rel. L1, {var}', yscale='log')
        ax[0].set_title(f'{name}: rollout error vs exact')
        ax[0].legend(fontsize=8)
        ax[-1].set_xlabel('t')
        fig.tight_layout()
        save(fig, f'accuracy_{name}.png')

    H.OUT.mkdir(parents=True, exist_ok=True)
    with open(H.OUT / 'accuracy.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['split', 'arm', f'rel_L1_t<={T_TRAIN}', f'rel_L1_t>{T_TRAIN}'])
        w.writerows([[s, a, f'{x:.4g}', f'{y:.4g}'] for s, a, x, y in rows])
    print(f'{"split":9s} {"arm":11s} {"t<=T_train":>11s} {"held-out":>9s}')
    for s, a, x, y in rows:
        print(f'{s:9s} {a:11s} {x:11.4f} {y:9.4f}')


# ======================================================================================
# Region fields
# ======================================================================================

def fig_regions(runs, sp, c, arm='full', times=(0.12, 0.20)):
    '''Each zone's field W_k over the whole domain at two times (one in the training window, one held out):
    solid where zone k owns the cell, faint where it is only an extension. Black dashed: exact. Grey: the
    blended output W. A region field should be a smooth extension past its zone: the blend relies on it.'''
    R = runs[arm]
    t = sp.t[sp.n0 + 1:]
    K = R['W_regions'].shape[3]
    fig, ax = plt.subplots(3, len(times), figsize=(5.5 * len(times), 8), sharex=True, squeeze=False)
    for col, tt in enumerate(times):
        j = int(np.argmin(np.abs(t - tt)))
        lab = R['labels'][c, j]
        for row, var in enumerate(VARS):
            a = ax[row, col]
            a.plot(XC, sp.W_ex[c, sp.n0 + 1 + j, :, row], 'k--', lw=0.8, label='exact')
            a.plot(XC, R['W'][c, j, :, row], color='0.6', lw=3, alpha=0.5, label='W (blended)')
            for k in range(K):
                Wk = R['W_regions'][c, j, :, k, row]
                a.plot(XC, Wk, color=f'C{k + 3}', lw=0.8, alpha=0.35)
                a.plot(XC, np.where(lab == k, Wk, np.nan), color=f'C{k + 3}', lw=1.6, label=f'zone {k}')
            lo, hi = sp.W_ex[c, ..., row].min(), sp.W_ex[c, ..., row].max()
            pad = 0.4 * max(hi - lo, 0.1 * abs(hi))
            a.set_ylim(lo - pad, hi + pad)
            a.set_ylabel(var)
        ax[0, col].set_title(f'{arm}, r = {sp.r[c]:.2f}, t = {t[j]:.3f}' + (' (held out)' if t[j] > T_TRAIN else ''))
        ax[-1, col].set_xlabel('x')
    ax[0, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save(fig, 'region_fields.png')


# ======================================================================================
# Level-set check
# ======================================================================================

@torch.no_grad()
def psi_samples(model, sp, every=5, half=8, per_cell=4):
    '''One step from the FV data at every `every`-th snapshot of every case: psi = phi_a - phi_b on a fine grid
    (per_cell points per cell) within +-half cells of each exact front at t_n, with (a, b) the data zones either
    side. Returns {wave: (d [P] signed distance in cells, + on the a (left) side, psi [P], held_out [P])}.'''
    out = {w: ([], [], []) for w in WAVES}
    off = np.arange(-half * per_cell, half * per_cell + 1) / per_cell                 # cells from the front
    for n in range(sp.n0, len(sp.t) - 1, every):
        W_grid = torch.as_tensor(sp.W[:, n]).float().to(DEVICE).transpose(1, 2)
        model.set_features(W_grid)
        for c in range(sp.C):
            x_ex = sp.x_exact(c, sp.t[n])
            for w, wave in enumerate(WAVES):
                pair = zone_pair(sp.classes[c, n], x_ex[w])
                if pair is None:
                    continue
                x = x_ex[w] + off * DX
                u = torch.as_tensor(2 * x - 1).float().to(DEVICE)[:, None]
                phi = model.geo_phi(u, torch.full((len(u),), c, device=DEVICE)).cpu().numpy()
                out[wave][0].append(-off)
                out[wave][1].append(phi[:, pair[0]] - phi[:, pair[1]])
                out[wave][2].append(np.full(len(off), sp.t[n] > T_TRAIN + 1e-9))
    return {w: tuple(np.concatenate(v) for v in vals) for w, vals in out.items() if vals[0]}


def fig_levelset(models, splits, arm='full'):
    """One plot: psi = phi_a - phi_b of one arm against signed distance to the exact front, one step from FV data,
    median over every case and snapshot of the splits, for the contact and the shock. psi is divided by the
    target's slope 2 kappa, so it reads in cells and the target is the line clamp(d, +-3). A learned level-set
    function sits on that line and crosses 0 at d = 0; the legend gives psi there (cells)."""
    if arm not in models or models[arm].geo_mode != 'levelset':
        return
    samples = [psi_samples(models[arm], sp) for sp in splits.values()]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    dd = np.linspace(-8, 8, 200)
    ax.plot(dd, np.clip(dd, -CLIP, CLIP), 'k--', lw=1, label='target (distance, clipped at 3 cells)')
    for wave, color in zip(WAVES, ('C3', 'C4')):
        d, psi, _ = (np.concatenate(v) for v in zip(*[s[wave] for s in samples if wave in s]))
        y = psi / (2 * KAPPA)
        grid = np.unique(d)
        med = np.array([np.median(y[d == g]) for g in grid])
        at0 = med[np.argmin(np.abs(grid))]
        ax.plot(grid, med, color=color, lw=1.8, label=f'{wave} (psi at interface = {at0:+.3f} cells)')
    ax.axhline(0, color='0.8', lw=0.5)
    ax.axvline(0, color='0.8', lw=0.5)
    ax.set(xlabel='signed distance to exact front (cells)', ylabel='psi / 2 kappa (cells)',
           title=f'{arm}: learned level-set function', xlim=(-8, 8), ylim=(-4, 4))
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    save(fig, 'levelset.png')


# ======================================================================================
# Front error
# ======================================================================================

def fig_fronts(runs, splits):
    '''Contact and shock position minus exact (cells), median over the cases of each split, with the
    interquartile band. Measured from rho alone, the same way for every arm and the FV data.'''
    fig, ax = plt.subplots(2, len(splits), figsize=(5.5 * len(splits), 7), sharex=True, squeeze=False)
    for col, (name, sp) in enumerate(splits.items()):
        t = sp.t[sp.n0 + 1:]
        series = {a: R['W'] for a, R in runs[name].items()}
        series['FV'] = sp.W[:, sp.n0 + 1:]
        for a, W in series.items():
            err = np.stack([(fronts_of(W[c], sp, c, t) - sp.x_exact(c, t)) / DX for c in range(sp.C)])   # [C, T, 2]
            for w in range(2):
                med = np.nanmedian(err[..., w], 0)
                q1, q3 = np.nanpercentile(err[..., w], [25, 75], 0)
                ax[w, col].plot(t, med, '--' if a == 'FV' else '-', color=COLORS[a], label=a)
                ax[w, col].fill_between(t, q1, q3, color=COLORS[a], alpha=0.15, lw=0)
        for w, wave in enumerate(WAVES):
            held_out(ax[w, col])
            ax[w, col].axhline(0, color='k', lw=0.5)
            ax[w, col].set(title=f'{name}: {wave}', ylabel='x - x_exact (cells)')
        ax[1, col].set_xlabel('t')
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, 'fronts.png')


# ======================================================================================
# RH enforcement
# ======================================================================================

def fig_rh(runs, splits):
    '''At the shock, median over cases: (left) RH residual of the cell states SIDE cells either side with the
    measured shock speed (0 = the jump satisfies RH), (middle) shock speed error, (right) post-shock state
    error against the exact star-R state. Same measurement for every arm and the FV data.'''
    fig, ax = plt.subplots(len(splits), 3, figsize=(14, 3.6 * len(splits)), squeeze=False, sharex=True)
    titles = ('RH residual (measured states and speed)', '|s - S_exact| / S_exact', 'post-shock state rel. error')
    for row, (name, sp) in enumerate(splits.items()):
        t = sp.t[sp.n0 + 1:]
        series = {a: R['W'] for a, R in runs[name].items()}
        series['FV'] = sp.W[:, sp.n0 + 1:]
        for a, W in series.items():
            terms = np.stack([np.stack(shock_terms(W[c], fronts_of(W[c], sp, c, t)[:, 1], t, sp, c))
                              for c in range(sp.C)])                                         # [C, 3, T]
            for k in range(3):
                ax[row, k].plot(t, np.nanmedian(terms[:, k], 0), '--' if a == 'FV' else '-', color=COLORS[a], label=a)
        for k in range(3):
            held_out(ax[row, k])
            ax[row, k].set(title=f'{name}: {titles[k]}', yscale='log')
        for k in range(3):
            ax[-1, k].set_xlabel('t')
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, 'rh.png')


# ======================================================================================
# Basis usage
# ======================================================================================

@torch.no_grad()
def contributions(model, W, x_cell, var=0):
    '''Per-basis contributions to variable var at the cell centres, one step from W [B, 512, 3]:
    C [B, 512, M] = a_{k(x) m}(x) b_m(x), k(x) the zone that owns the cell at t_n (one-region: 0). Their sum is
    the pre-softplus field. Uses the FieldNet return_parts hook.'''
    W_grid = torch.as_tensor(W).float().to(DEVICE).transpose(1, 2)
    B = len(W_grid)
    x, group = points(x_cell.to(DEVICE), B)
    model.set_features(W_grid)
    g = sample_grid(model.g_grid, group, x)
    _, a, b = model.FieldNet(cell_offset(x, model.N_grid), g, return_parts=True)          # [P,K,M,F], [P,M,F]
    k = model.geo_phi(x, group).argmax(-1) if model.geo_mode == 'levelset' else torch.zeros_like(group)
    a_own = a[torch.arange(len(k)), k]                                                     # [P, M, F]
    return (a_own[..., var] * b[..., var]).view(B, N, -1).double().cpu().numpy()


def fig_basis(models, sp, c, every=10, t_show=0.12):
    '''How each arm uses its M FieldNet bases for rho, one step from the FV data of every case of the split.
    Left: singular values of the contribution matrix C [512, M] per snapshot, normalised by the largest, mean
    over snapshots; legend: effective rank exp(entropy of s^2 / sum s^2). A field that must carry the jumps
    itself (one-region) should need more modes. Right: the top 3 spatial modes (U_i s_i) of one snapshot, with
    the exact contact and shock marked: sharp features pinned at the fronts mean the bases build the jumps.'''
    snaps = list(range(sp.n0, len(sp.t) - 1, every))
    arms = list(models)
    fig = plt.figure(figsize=(12, 2.6 * len(arms)))
    gs = fig.add_gridspec(len(arms), 2, width_ratios=(1, 1.4))
    ax_s = fig.add_subplot(gs[:, 0])
    j = int(np.argmin(np.abs(sp.t - t_show)))
    for row, arm in enumerate(arms):
        spectra = []
        for n in snaps:
            C = contributions(models[arm], sp.W[:, n], sp.x_cell)
            for Cc in C:
                s = np.linalg.svd(Cc, compute_uv=False)
                spectra.append(s / s[0])
        spectra = np.array(spectra)
        p = spectra ** 2 / (spectra ** 2).sum(1, keepdims=True)
        rank = np.exp(-(p * np.log(p + 1e-30)).sum(1)).mean()
        m = np.arange(1, spectra.shape[1] + 1)
        ax_s.semilogy(m, spectra.mean(0), 'o-', color=COLORS[arm], label=f'{arm} (eff. rank {rank:.1f})')

        U, s, _ = np.linalg.svd(contributions(models[arm], sp.W[c:c + 1, j], sp.x_cell)[0], full_matrices=False)
        a = fig.add_subplot(gs[row, 1])
        for i in range(3):
            a.plot(XC, U[:, i] * s[i], lw=1, label=f'mode {i + 1}')
        for x in sp.x_exact(c, sp.t[j]):
            a.axvline(x, color='k', ls=':', lw=0.8)
        a.set(title=f'{arm}: top modes, r = {sp.r[c]:.2f}, t = {sp.t[j]:.3f}', ylabel='U_i s_i')
        if row == 0:
            a.legend(fontsize=7)
        if row == len(arms) - 1:
            a.set_xlabel('x')
    ax_s.set(title=f'rho basis usage ({sp.name}): singular values', xlabel='mode', ylabel='s_i / s_1')
    ax_s.legend(fontsize=8)
    fig.tight_layout()
    save(fig, 'basis.png')


# ======================================================================================
# x-t zone map
# ======================================================================================

def psi_zeros(phi, labels):
    '''Sub-cell fronts of one snapshot: at each face where the label changes (a | b), the zero of
    psi = phi_a - phi_b, linear between the centres (as cut_blend). Returns x [F].'''
    i = np.flatnonzero(labels[:-1] != labels[1:])
    a, b = labels[i], labels[i + 1]
    si = phi[i, a] - phi[i, b]
    sj = phi[i + 1, a] - phi[i + 1, b]
    frac = np.clip(si / np.where(si != sj, si - sj, 1), 0, 1)
    return XC[i] + frac * DX


def fig_zonemap(runs, sp, c):
    '''x-t map of the zone labels of the rollout of case c (level-set arms). Black dots: psi = 0, the model's
    sub-cell fronts. White dashed: the exact wave paths (fan head and tail, contact, shock). Above the grey line
    is the held-out time band. Shows whether zones stay contiguous and fronts track the exact waves.'''
    arms = [a for a, R in runs.items() if R['phi'] is not None]
    if not arms:
        return
    t = sp.t[sp.n0 + 1:]
    K = runs[arms[0]]['phi'].shape[-1]
    fig, ax = plt.subplots(1, len(arms), figsize=(5.5 * len(arms), 5), squeeze=False, sharey=True)
    for a, arm in zip(ax[0], arms):
        R = runs[arm]
        a.imshow(R['labels'][c], origin='lower', aspect='auto', cmap=plt.get_cmap('tab10', 10),
                 vmin=-0.5, vmax=9.5, extent=(0, 1, t[0] - 0.001, t[-1] + 0.001), interpolation='nearest')
        for j in range(len(t)):
            x = psi_zeros(R['phi'][c, j], R['labels'][c, j])
            a.plot(x, np.full(len(x), t[j]), 'k.', ms=1.5)
        tt = np.linspace(0, t[-1], 50)
        for s in sp.exact[c]['speeds']:
            a.plot(X0 + s * tt, tt, 'w--', lw=0.8)
        a.axhline(T_TRAIN, color='0.5', lw=1)
        a.set(xlim=(0, 1), ylim=(t[0], t[-1]), title=f'{arm}: zones (K = {K}), r = {sp.r[c]:.2f}', xlabel='x')
    ax[0, 0].set_ylabel('t')
    fig.tight_layout()
    save(fig, f'zonemap_{sp.name}_{c}.png')


# ======================================================================================

FIGS = ('accuracy', 'regions', 'levelset', 'fronts', 'rh', 'basis', 'zonemap')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--figs', nargs='+', default=list(FIGS), choices=FIGS)
    ap.add_argument('--splits', nargs='+', default=list(SPLITS), choices=ALL_SPLITS,
                    help='splits to evaluate (use val to choose between model sets, then report test)')
    ap.add_argument('--split', choices=ALL_SPLITS, help='split of the one-case figures (default: the first of --splits)')
    ap.add_argument('--case', type=int, help='case index of the one-case figures (default: mid r)')
    ap.add_argument('--runs', help='folder with <name>/trained_model.pt (default: results/models)')
    ap.add_argument('--cfg', help='models_cfg folder (default: configs/models_cfg)')
    ap.add_argument('--out', help='output folder (default: results/diagnostics)')
    args = ap.parse_args()
    H.configure(args.runs, args.cfg, args.out)

    models = available_models()
    if not models:
        return
    args.split = args.split or args.splits[0]
    splits = {s: Split(s) for s in dict.fromkeys(args.splits + [args.split])}
    sp = splits[args.split]
    splits = {s: splits[s] for s in args.splits}
    c = sp.mid_case() if args.case is None else args.case
    need_runs = set(args.figs) & {'accuracy', 'regions', 'fronts', 'rh', 'zonemap'}
    runs = rollouts(models, splits) if need_runs else None

    if 'accuracy' in args.figs:
        fig_accuracy(runs, splits)
    if 'regions' in args.figs and 'full' in models:
        fig_regions(runs[args.split], sp, c)
    if 'levelset' in args.figs:
        fig_levelset(models, splits)
    if 'fronts' in args.figs:
        fig_fronts(runs, splits)
    if 'rh' in args.figs:
        fig_rh(runs, splits)
    if 'basis' in args.figs:
        fig_basis(models, sp, c)
    if 'zonemap' in args.figs:
        fig_zonemap(runs[args.split], sp, c)


if __name__ == '__main__':
    main()
