'''
Lean evaluation core for the old (global-latent) model. Its job is to motivate the localfield architecture, so it
makes three figures only. No oracle, no cross-model comparison.

    python -m evaluation.motivation                        (from the repo root: every figure)
    python -m evaluation.motivation --figs profiles --split test_ood --case 3

Figures (results/motivation/):
    profiles   profiles_<split>_<case>.mp4   rho, v, p against x over the rollout (t0 = 0.02 to 0.24) for every
                                             arm with a checkpoint, over the FV data and the exact solution. The
                                             axes turn grey in the held-out time band (t > 0.16): where it fails.
    levelset   levelset.png                  psi = phi_a - phi_b over its fitted slope (cells) against signed
                                             distance to the exact front, one step from FV data. A level-set
                                             function is the line y = d through 0; a step (labels, not distances)
                                             is what stage 1's CE + boundary loss asks for.

    influence  influence.png                 which input cells the model reads to answer at one point behind
                                             the shock: the whole domain, through the global latents (compare
                                             localfield's influence.png: a +-8 / +-15 cell window)

Model loading and the rollout are the ones in evaluation.diagnostics.
'''

import argparse

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation

import evaluation.diagnostics as D
from evaluation.diagnostics import (MODELS, COLORS, DEVICE, XC, DX, T_TRAIN, T_MIN, T_END, X0,
                                    load_model, rollout, cons_to_prim, exact_case)
from train.dataset import SnapshotDataset
from model.transforms import MSFourier_Features

OUT = D.ROOT / "results" / "motivation"
SPLITS = ('test_id', 'test_ood')
WAVES = ('contact', 'shock')
CLIP = 3                                            # cells: localfield's distance target is clipped here
VARS = ('rho', 'v', 'p')

class Split:
    '''One split from t = 0 to T_END: FV W [C, S, 512, 3], exact W_ex (primitives), classes [C, S, 512], t [S],
    r [C]. n0 is the rollout start (first t >= T_MIN).'''

    def __init__(self, split):
        ds = SnapshotDataset(D.DATA, split, t_max=T_END, exact=True)
        self.name, self.r, self.t = split, ds.r.numpy(), ds.t.numpy() * T_TRAIN
        self.W = ds.W.double().numpy()
        self.W_ex = cons_to_prim(ds.U_exact.double().numpy())
        self.x_cell = ds.x_centers
        with h5py.File(D.DATA, 'r') as f:
            self.classes = np.stack([f['cases'][c]['classes'][()][ds.use] for c in ds.case_ids])
        self.n0 = int((self.t < T_MIN - 1e-9).sum())
        self.C = len(self.r)
        self.speeds = [exact_case(float(r))['speeds'] for r in self.r]

    def x_exact(self, c, t):
        '''Exact (contact, shock) positions [..., 2] of case c at times t.'''
        return X0 + self.speeds[c][2:] * np.asarray(t)[..., None]

    def mid_case(self):
        return int(np.argsort(self.r)[self.C // 2])

def available_models():
    out = {}
    for arm, (cfg, ckpt) in MODELS.items():
        if ckpt.exists():
            out[arm] = load_model(cfg, ckpt)                                   # (model, encoder)
        else:
            print(f'skipping {arm}: no checkpoint at {ckpt}')
    return out

def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT / name}')

# ======================================================================================
# Profile animation
# ======================================================================================

def fig_profiles(models, sp, c, fps=12):
    '''rho, v, p against x over the rollout of case c: every arm, the FV data (dots) and the exact solution.'''
    t = sp.t[sp.n0:]
    W_fv, W_ex = sp.W[c, sp.n0:], sp.W_ex[c, sp.n0:]
    arms = {}
    for arm, (model, encoder) in models.items():
        W = rollout(model, encoder, sp.W[c:c + 1, sp.n0], len(t) - 1, sp.x_cell.to(DEVICE))[0]['W']
        arms[arm] = np.concatenate([W_fv[:1], W])                                # t0 is the shared input

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    lines = []
    for row, (ax, var) in enumerate(zip(axes, VARS)):
        lines.append((ax.plot(XC, W_fv[0, :, row], '.', ms=2, color='0.5', label='FV')[0], W_fv, row))
        lines.append((ax.plot(XC, W_ex[0, :, row], 'k-', lw=0.8, label='exact')[0], W_ex, row))
        for arm, W in arms.items():
            lines.append((ax.plot(XC, W[0, :, row], color=COLORS[arm], lw=1.2, label=arm)[0], W, row))
        lo, hi = W_ex[..., row].min(), W_ex[..., row].max()
        pad = 0.25 * max(hi - lo, 0.1 * abs(hi))
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(var)
    axes[-1].set_xlabel('x')
    axes[0].legend(fontsize=8, loc='upper right', ncol=2)
    title = fig.suptitle('')
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    def frame(j):
        for line, W, row in lines:
            line.set_ydata(W[j, :, row])
        out = t[j] > T_TRAIN + 1e-9
        for ax in axes:
            ax.set_facecolor('0.93' if out else 'white')
        title.set_text(f'old model, {sp.name}, r = {sp.r[c]:.2f}, t = {t[j]:.3f}' +
                       ('  (held-out time band)' if out else ''))
        return [l for l, _, _ in lines] + [title]

    anim = animation.FuncAnimation(fig, frame, frames=len(t))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'profiles_{sp.name}_{c}.mp4'
    anim.save(path, writer=animation.FFMpegWriter(fps=fps), dpi=120)
    plt.close(fig)
    print(f'saved {path}')

# ======================================================================================
# Level-set check
# ======================================================================================

def zone_pair(cls, x):
    '''Zones (a, b) either side of x from the data classes [512] (-1 = unlabelled): nearest labelled cells.'''
    lab = np.flatnonzero(cls >= 0)
    left, right = lab[XC[lab] < x], lab[XC[lab] > x]
    if not len(left) or not len(right):
        return None
    a, b = cls[left[-1]], cls[right[0]]
    return (int(a), int(b)) if a != b else None

@torch.no_grad()
def psi_samples(model, encoder, sp, every=5, half=8, per_cell=4):
    '''One step from the FV data at every `every`-th snapshot: psi = phi_a - phi_b on a fine grid within +-half
    cells of each exact front at t_n, (a, b) the data zones either side. The old geometry reads Fourier(x) and
    the global z_geo. Returns {wave: (d [P] cells, + on the left zone's side, psi [P], held_out [P])}.'''
    out = {w: ([], [], []) for w in WAVES}
    off = np.arange(-half * per_cell, half * per_cell + 1) / per_cell
    for n in range(sp.n0, len(sp.t) - 1, every):
        W_grid = torch.as_tensor(sp.W[:, n]).float().to(DEVICE).transpose(1, 2)
        z_geo, _ = encoder(W_grid)
        for c in range(sp.C):
            x_ex = sp.x_exact(c, sp.t[n])
            for w, wave in enumerate(WAVES):
                pair = zone_pair(sp.classes[c, n], x_ex[w])
                if pair is None:
                    continue
                u = torch.as_tensor(2 * (x_ex[w] + off * DX) - 1).float().to(DEVICE)[:, None]
                phi = model.geo_phi(u, z_geo[c].expand(len(u), -1)).cpu().numpy()
                out[wave][0].append(-off)
                out[wave][1].append(phi[:, pair[0]] - phi[:, pair[1]])
                out[wave][2].append(np.full(len(off), sp.t[n] > T_TRAIN + 1e-9))
    return {w: tuple(np.concatenate(v) for v in vals) for w, vals in out.items() if vals[0]}

def levelset_stats(d, psi, fit=2.5):
    '''Straight-line fit of psi within +-fit cells: its zero (cells), R^2 (1 = distance-like) and slope.'''
    m = np.abs(d) <= fit
    A = np.polyfit(d[m], psi[m], 1)
    res = psi[m] - np.polyval(A, d[m])
    return -A[1] / A[0], 1 - res.var() / psi[m].var(), A[0]

def fig_levelset(models, splits, arm='full'):
    """One plot: psi = phi_a - phi_b of the old full arm against signed distance to the exact front, one step from
    FV data, median over every case and snapshot, contact and shock. The old training has no distance target, so
    psi has no scale of its own: it is divided by the slope of its straight-line fit within 2.5 cells, which puts
    it in cells as in localfield's levelset.png. Dashed: localfield's target, clamp(d, +-3). The legend gives psi
    at the interface (cells) and the R^2 of that fit (1 = distance-like)."""
    if arm not in models or models[arm][0].geo_mode != 'levelset':
        return
    samples = [psi_samples(*models[arm], sp) for sp in splits.values()]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    dd = np.linspace(-8, 8, 200)
    ax.plot(dd, np.clip(dd, -CLIP, CLIP), 'k--', lw=1, label='localfield target (distance, clipped at 3 cells)')
    for wave, color in zip(WAVES, ('C3', 'C4')):
        d, psi, _ = (np.concatenate(v) for v in zip(*[s[wave] for s in samples if wave in s]))
        _, r2, slope = levelset_stats(d, psi)
        y = psi / slope
        grid = np.unique(d)
        med = np.array([np.median(y[d == g]) for g in grid])
        at0 = med[np.argmin(np.abs(grid))]
        ax.plot(grid, med, color=color, lw=1.8, label=f'{wave} (psi at interface = {at0:+.3f} cells, R² = {r2:.2f})')
    ax.axhline(0, color='0.8', lw=0.5)
    ax.axvline(0, color='0.8', lw=0.5)
    ax.set(xlabel='signed distance to exact front (cells)', ylabel='psi / fitted slope (cells)',
           title=f'old {arm}: level-set function', xlim=(-8, 8), ylim=(-6, 6))
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    save(fig, 'levelset.png')

def fig_influence(models, sp, c, arm='full', t_show=0.12, behind=4):
    """Which input cells the old model reads to answer at one point, the same figure as localfield's influence.png.
    One step from the FV data of case c at t_show, x0 = `behind` cells behind the exact shock; for each input cell
    x: |d out(x0) / d W(t_n)(x)|, summed over the input (rho, v, p) and the outputs, normalised by its maximum.
        geometry   out = phi_k(x0), all k            (reads Fourier(x0) and z_geo)
        field      out = W_k(x0) of the owning zone   (reads Fourier(x0) and z_field)
    The latents come from the encoder of the whole field, so the influence spreads over the whole domain
    (localfield: a +-8 / +-15 cell window, 0 outside)."""
    if arm not in models:
        return
    model, encoder = models[arm]
    j = int(np.argmin(np.abs(sp.t - t_show)))
    x0 = sp.x_exact(c, sp.t[j])[1] - behind * DX
    u0 = torch.tensor([[2 * x0 - 1]], dtype=torch.float32, device=DEVICE)

    def outputs(W_grid):
        z_geo, z_field = encoder(W_grid)
        out = {}
        if model.geo_mode == 'levelset':
            phi = model.geo_phi(u0, z_geo[:1])
            out['geometry: phi(x0)'] = phi[0]
            k = int(phi.argmax(-1))
        else:
            k = 0
            z_field = torch.cat([z_geo, z_field], dim=-1)
        W_k = model.heads_fn(model.FieldNet(MSFourier_Features(u0, model.B_field), z_field[:1]))[0, k]
        out['field: W_k(x0)'] = W_k
        return out

    curves = {}
    with torch.enable_grad():
        W_grid = torch.as_tensor(sp.W[c:c + 1, j]).float().to(DEVICE).transpose(1, 2)
        for name in outputs(W_grid):
            total = np.zeros(len(XC))
            for k in range(outputs(W_grid)[name].numel()):
                W_grid = W_grid.detach().requires_grad_(True)
                g, = torch.autograd.grad(outputs(W_grid)[name].reshape(-1)[k], W_grid)
                total += g[0].abs().sum(0).cpu().numpy()
            curves[name] = total / max(total.max(), 1e-30)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    for (name, inf), color in zip(curves.items(), ('C0', 'C1')):
        ax.plot(XC, inf, color=color, lw=1.5, label=name)
    ax.axvline(x0, color='k', lw=1, label='query point x0')
    for x in sp.x_exact(c, sp.t[j]):
        ax.axvline(x, color='0.5', ls=':', lw=1)
    ax.set(xlabel='input cell x', ylabel='influence on output at x0 (normalised)', xlim=(0, 1),
           title=f'old {arm}: what the model reads to answer at x0 ({behind} cells behind the shock), '
                 f'r = {sp.r[c]:.2f}, t = {sp.t[j]:.3f}')
    ax.title.set_fontsize(9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, 'influence.png')

FIGS = ('profiles', 'levelset', 'influence')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--figs', nargs='+', default=list(FIGS), choices=FIGS)
    ap.add_argument('--split', default=None, choices=SPLITS, help='profiles for this split only (default: both)')
    ap.add_argument('--case', type=int, help='case index for the profiles (default: mid r)')
    args = ap.parse_args()

    models = available_models()
    if not models:
        return
    splits = {s: Split(s) for s in SPLITS}
    if 'profiles' in args.figs:
        for name in ([args.split] if args.split else SPLITS):
            sp = splits[name]
            fig_profiles(models, sp, sp.mid_case() if args.case is None else args.case)
    if 'levelset' in args.figs:
        fig_levelset(models, splits)
    if 'influence' in args.figs:
        sp = splits['test_id']
        fig_influence(models, sp, sp.mid_case() if args.case is None else args.case)

if __name__ == '__main__':
    main()