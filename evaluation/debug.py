'''
Debug probes for the localfield model: look inside the jump layer when a diagnostics.py figure looks off.
No oracle. Rollouts from t0 = 0.02 to 0.24 on the model's own output, as in diagnostics.py.

    python -m evaluation.debug                        (from the repo root: every group)
    python -m evaluation.debug jump constraint        (any of the groups below)

    jump        what the jump layer itself sees at the shock (level-set arms), from its hooks, median over the
                band points the shock test passes, per case and step:
                front position (the Newton foot x_foot) - exact, lab-frame speed s n - exact S, Mn - exact Mn,
                and the share of shock-flagged band points that sit away from the exact shock (> 5 cells)
    constraint  the full arm's weights run twice, RH applied and switched off at eval time (jump_mode = free):
                rollout rel. L1 vs t and post-shock state error vs t. What the constraint contributes, with
                the fields fixed
    errormap    |W - W_ex| summed over (rho, v, p) / sum |W_ex| on the x-t plane for each arm, with the exact
                wave paths: where the error lives

Output goes to results/debug/.
'''

import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

from train.helper import points
from evaluation.helper import (COLORS, SPLITS, T_TRAIN, N, DX, X0, DEVICE, ROOT,
                               Split, available_models, rollout, held_out)
from evaluation.diagnostics import rel_l1_t, fronts_of, shock_terms

OUT = ROOT / 'results' / 'debug'
FAR = 5                                             # cells: a shock point farther than this from the exact shock is spurious

def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT / name}')

# ======================================================================================
# jump: the jump layer's own view of the shock
# ======================================================================================

@torch.no_grad()
def jump_rollout(model, sp):
    '''Rollout of every case of sp, keeping per case and step (all [C, T]) the jump hooks at the shock: median
    x_foot (data x), lab-frame speed and Mn over the shock points within FAR cells of the exact shock, and the
    share of shock points farther away (spurious).'''
    C, T = sp.C, len(sp.t) - sp.n0 - 1
    t = sp.t[sp.n0 + 1:]
    out = {k: np.full((C, T), np.nan) for k in ('x', 'S', 'Mn', 'spurious')}
    W = torch.as_tensor(sp.W[:, sp.n0]).float().to(DEVICE)
    x, group = points(sp.x_cell.to(DEVICE), C)
    for j in range(T):
        res = model(x, W.transpose(1, 2), group)
        W = res['W'].view(C, N, 3)
        jp = res['jump']
        sh = jp['shock']
        rows = group[jp['idx']][sh].cpu().numpy()
        xf = ((jp['x_foot'][sh, 0] + 1) / 2).cpu().numpy()
        S = (jp['s'] * jp['n'][:, 0])[sh].cpu().numpy()
        Mn = jp['Mn'][sh].cpu().numpy()
        for c in range(C):
            x_ex = sp.x_exact(c, t[j])[1]
            mine = rows == c
            near = mine & (np.abs(xf - x_ex) <= FAR * DX)
            if mine.any():
                out['spurious'][c, j] = 1 - near.sum() / mine.sum()
            if near.any():
                out['x'][c, j] = np.median(xf[near])
                out['S'][c, j] = np.median(S[near])
                out['Mn'][c, j] = np.median(Mn[near])
    return out

def group_jump(models, splits):
    arms = [a for a, m in models.items() if m.geo_mode == 'levelset']
    if not arms:
        return
    fig, ax = plt.subplots(len(splits), 4, figsize=(18, 3.6 * len(splits)), squeeze=False, sharex=True)
    titles = ('x_foot - x_exact (cells)', 's n - S_exact (rel.)', 'Mn - Mn_exact', 'spurious shock points (share)')
    for row, (name, sp) in enumerate(splits.items()):
        t = sp.t[sp.n0 + 1:]
        S_ex = np.array([e['speeds'][3] for e in sp.exact])[:, None]
        Mn_ex = np.array([e['Mn'] for e in sp.exact])[:, None]
        x_ex = np.stack([sp.x_exact(c, t)[:, 1] for c in range(sp.C)])
        for arm in arms:
            h = jump_rollout(models[arm], sp)
            terms = ((h['x'] - x_ex) / DX, (h['S'] - S_ex) / S_ex, h['Mn'] - Mn_ex, h['spurious'])
            for k, v in enumerate(terms):
                ax[row, k].plot(t, np.nanmedian(v, 0), color=COLORS[arm], label=arm)
                q1, q3 = np.nanpercentile(v, [25, 75], 0)
                ax[row, k].fill_between(t, q1, q3, color=COLORS[arm], alpha=0.15, lw=0)
        for k in range(4):
            held_out(ax[row, k])
            if k < 3:
                ax[row, k].axhline(0, color='k', lw=0.5)
            ax[row, k].set_title(f'{name}: {titles[k]}')
            ax[-1, k].set_xlabel('t')
    ax[0, 0].legend(fontsize=8)
    save(fig, 'jump.png')

# ======================================================================================
# constraint: the full arm with RH on and off at eval time
# ======================================================================================

def group_constraint(models, splits):
    if 'full' not in models:
        print('constraint: needs the full arm')
        return
    model = models['full']
    fig, ax = plt.subplots(len(splits), 2, figsize=(11, 3.6 * len(splits)), squeeze=False, sharex=True)
    for row, (name, sp) in enumerate(splits.items()):
        t = sp.t[sp.n0 + 1:]
        W_ex = sp.W_ex[:, sp.n0 + 1:]
        for mode, ls in (('rh', '-'), ('free', '--')):
            model.jump_mode = mode
            W = rollout(model, sp)['W']
            down = np.stack([shock_terms(W[c], fronts_of(W[c], sp, c, t)[:, 1], t, sp, c)[2] for c in range(sp.C)])
            label = 'RH on' if mode == 'rh' else 'RH off (eval only)'
            ax[row, 0].plot(t, rel_l1_t(W, W_ex).mean(0), ls, color=COLORS['full'], label=label)
            ax[row, 1].plot(t, np.nanmedian(down, 0), ls, color=COLORS['full'], label=label)
        model.jump_mode = 'rh'
        for k, title in enumerate(('rollout rel. L1 vs exact', 'post-shock state rel. error')):
            held_out(ax[row, k])
            ax[row, k].set(title=f'{name}, full arm: {title}', yscale='log')
            ax[-1, k].set_xlabel('t')
    ax[0, 0].legend(fontsize=8)
    save(fig, 'constraint.png')

# ======================================================================================
# errormap: where the error lives
# ======================================================================================

def group_errormap(models, splits, split='test_id'):
    sp = splits[split]
    c = sp.mid_case()
    t = sp.t[sp.n0 + 1:]
    W_ex = sp.W_ex[c, sp.n0 + 1:]
    arms = list(models)
    fig, ax = plt.subplots(1, len(arms), figsize=(5 * len(arms), 4.5), squeeze=False, sharey=True)
    for a, arm in zip(ax[0], arms):
        W = rollout(models[arm], sp, cases=[c])['W'][0]
        err = np.abs(W - W_ex).sum(-1) / np.abs(W_ex).sum(-1)                         # [T, 512]
        im = a.imshow(np.log10(err + 1e-6), origin='lower', aspect='auto', cmap='magma', vmin=-4, vmax=0,
                      extent=(0, 1, t[0], t[-1]))
        tt = np.linspace(0, t[-1], 50)
        for s in sp.exact[c]['speeds']:
            a.plot(X0 + s * tt, tt, 'c--', lw=0.6)
        a.axhline(T_TRAIN, color='w', lw=1)
        a.set(xlim=(0, 1), ylim=(t[0], t[-1]), title=f'{arm}: log10 rel. error, r = {sp.r[c]:.2f}', xlabel='x')
    ax[0, 0].set_ylabel('t')
    fig.colorbar(im, ax=ax[0, -1])
    save(fig, f'errormap_{split}_{c}.png')

GROUPS = {'jump': group_jump, 'constraint': group_constraint, 'errormap': group_errormap}

def main(groups):
    bad = [g for g in groups if g not in GROUPS]
    if bad:
        sys.exit(f'unknown group(s) {bad}; choose from {list(GROUPS)}')
    models = available_models()
    if not models:
        return
    splits = {s: Split(s) for s in SPLITS}
    for g in groups or GROUPS:
        GROUPS[g](models, splits)

if __name__ == '__main__':
    main(sys.argv[1:])