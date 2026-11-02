'''
Profile animation: rho, v, p against x over the rollout (t0 = 0.02 to 0.24) for every arm with a checkpoint
(full, free-jump, one-region), over the FV data (dots) and the exact solution. The axes turn grey in the
held-out time band (t > 0.16).

    python -m evaluation.animate                      (from the repo root: mid-r case of test_id and test_ood)
    python -m evaluation.animate --split test_ood --case 3
    python -m evaluation.animate --t_end 0.32         (roll out past the data; exact only after t = 0.24)

Past the last snapshot (0.24) there is no FV data: the models keep stepping on their own output and are drawn
against the exact solution (point values). For the strongest cases the waves reach x = 0 or 1 near t = 0.34; the
exact solution is for an infinite domain, so stop before that.

Output: results/diagnostics/profiles_<split>_<case>.mp4 (needs ffmpeg).
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import evaluation.helper as H
from evaluation.helper import ( COLORS, SPLITS, T_TRAIN, T_END, XC, VARS, Split, available_models, rollout,
                               exact_profile)

def animate(models, sp, c, fps=12, t_end=T_END):
    '''rho, v, p against x over the rollout of case c to t_end: every arm, the FV data (dots, while it lasts) and
    the exact solution. The axes turn grey in the held-out time band.'''
    t = sp.t[sp.n0:]
    W_fv, W_ex = sp.W[c, sp.n0:], sp.W_ex[c, sp.n0:]
    dt = t[-1] - t[-2]
    extra = max(0, int(round((t_end - t[-1]) / dt)))                       # steps past the data
    if extra:
        t_more = t[-1] + dt * np.arange(1, extra + 1)
        t = np.concatenate([t, t_more])
        W_fv = np.concatenate([W_fv, np.full((extra,) + W_fv.shape[1:], np.nan)])
        W_ex = np.concatenate([W_ex, np.stack([exact_profile(sp.r[c], tt) for tt in t_more])])
    arms = {a: np.concatenate([W_fv[:1], rollout(m, sp, cases=[c], n_steps=len(t) - 1)['W'][0]])
            for a, m in models.items()}                                    # t0: the shared input

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    lines = []
    for row, (ax, var) in enumerate(zip(axes, VARS)):
        lines.append((ax.plot(XC, W_fv[0, :, row], '.', ms=2, color=COLORS['FV'], label='FV')[0], W_fv, row))
        lines.append((ax.plot(XC, W_ex[0, :, row], 'k-', lw=0.8, label='exact')[0], W_ex, row))
        for a, W in arms.items():
            lines.append((ax.plot(XC, W[0, :, row], color=COLORS[a], lw=1.2, label=a)[0], W, row))
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
        note = '  (past the data: exact only)' if t[j] > T_END + 1e-9 else '  (held-out time band)' if out else ''
        title.set_text(f'{sp.name}, r = {sp.r[c]:.2f}, t = {t[j]:.3f}' + note)
        return [l for l, _, _ in lines] + [title]

    anim = animation.FuncAnimation(fig, frame, frames=len(t))
    H.OUT.mkdir(parents=True, exist_ok=True)
    path = H.OUT / f'profiles_{sp.name}_{c}' + (f'_t{t_end:g}' if t_end > T_END + 1e-9 else '') + '.mp4'
    anim.save(path, writer=animation.FFMpegWriter(fps=fps), dpi=120)
    plt.close(fig)
    print(f'saved {path}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=SPLITS, help='default: both')
    ap.add_argument('--case', type=int, help='case index in the split (default: mid r)')
    ap.add_argument('--fps', type=int, default=12)
    ap.add_argument('--t_end', type=float, default=T_END, help='roll out to this time (past 0.24: exact only)')
    args = ap.parse_args()

    models = available_models()
    if not models:
        return
    for name in ([args.split] if args.split else SPLITS):
        sp = Split(name)
        animate(models, sp, sp.mid_case() if args.case is None else args.case, args.fps, args.t_end)

if __name__ == '__main__':
    main()