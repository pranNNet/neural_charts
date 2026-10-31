'''
Animation of the rollout profiles: rho, v, p against x from t0 = 0.02 to 0.24, one frame per step,
for every arm with a checkpoint (diagnostics.MODELS: full, free-jump, one-region), over the FV cell
averages and the exact solution. Frames past T_train = 0.16 are the held-out time band.

    python -m evaluation.animate                          (from the project root: mid-r case of test_id)
    python -m evaluation.animate test_ood 3 --ref exact   (split, case index; --ref fv | exact | both)

Output goes to results/animations/<split>_<case>.mp4 (.gif if ffmpeg is missing).
'''

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation

import evaluation.diagnostics as D
from evaluation.diagnostics import (T_TRAIN, T_END, MODELS, COLORS, DEVICE, XC,
                                    load_model, rollout, start_index, cons_to_prim)
from train.dataset import SnapshotDataset

OUT = D.ROOT / 'results' / 'animations'


def profiles(split, c):
    '''(t [T+1], r, {name: W [T+1, 512, 3]}, W_fv, W_ex): every arm's rollout of one case, t0 first.'''
    ds = SnapshotDataset(D.DATA, split, t_max=T_END, exact=True)
    n0 = start_index(ds)
    t = ds.t[n0:].numpy() * T_TRAIN
    W0 = ds.W[c:c + 1, n0]
    runs = {}
    for name, (cfg, ckpt) in MODELS.items():
        if not ckpt.exists():
            print(f'skipping {name}: no checkpoint at {ckpt}')
            continue
        model, encoder = load_model(cfg, ckpt)
        W = rollout(model, encoder, W0, len(t) - 1, ds.x_centers.to(DEVICE))[0]['W']
        runs[name] = np.concatenate([W0.double().numpy(), W])                      # t0 is the shared input
    W_ex = cons_to_prim(ds.U_exact[c, n0:].double().numpy())
    return t, float(ds.r[c]), runs, ds.W[c, n0:].double().numpy(), W_ex


def animate(split, c, ref='both', fps=12):
    t, r, runs, W_fv, W_ex = profiles(split, c)
    fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    lines = []
    for row, var in enumerate(['rho', 'v', 'p']):
        a = ax[row]
        if ref in ('fv', 'both'):
            lines.append((a.plot(XC, W_fv[0, :, row], '.', ms=2, color='0.6', label='FV')[0], W_fv, row))
        if ref in ('exact', 'both'):
            lines.append((a.plot(XC, W_ex[0, :, row], 'k-', lw=1, label='exact')[0], W_ex, row))
        for name, W in runs.items():
            lines.append((a.plot(XC, W[0, :, row], color=COLORS[name], lw=1.2, label=name)[0], W, row))
        lo, hi = W_ex[..., row].min(), W_ex[..., row].max()                      # exact range + 25%: a model
        pad = 0.25 * max(hi - lo, 0.1 * abs(hi))                                   # that leaves it is clipped
        a.set_ylim(lo - pad, hi + pad)
        a.set_ylabel(var)
    ax[-1].set_xlabel('x')
    ax[0].legend(fontsize=8, loc='upper right')
    title = fig.suptitle('')

    def frame(j):
        for line, W, row in lines:
            line.set_ydata(W[j, :, row])
        held = ' (held-out time band)' if t[j] > T_TRAIN + 1e-9 else ''
        title.set_text(f'{split}, case {c}, r = {r:.2f}:  t = {t[j]:.3f}{held}')
        return [line for line, _, _ in lines] + [title]

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    anim = animation.FuncAnimation(fig, frame, frames=len(t), blit=False)
    os.makedirs(OUT, exist_ok=True)
    if animation.writers.is_available('ffmpeg'):
        path = OUT / f'{split}_{c}.mp4'
        anim.save(path, writer=animation.FFMpegWriter(fps=fps), dpi=120)
    else:
        path = OUT / f'{split}_{c}.gif'
        anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    print(f'saved {path}')
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('split', nargs='?', default='test_id')
    ap.add_argument('case', nargs='?', type=int, help='case index in the split (default: mid r)')
    ap.add_argument('--ref', default='both', choices=['fv', 'exact', 'both'])
    ap.add_argument('--fps', type=int, default=12)
    args = ap.parse_args()

    c = args.case
    if c is None:                                                                  # mid r, as diagnostics.picks
        r = SnapshotDataset(D.DATA, args.split).r.numpy()
        c = int(np.argsort(r)[len(r) // 2])
    animate(args.split, c, args.ref, args.fps)


if __name__ == '__main__':
    main()
