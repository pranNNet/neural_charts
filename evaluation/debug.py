'''
Tier 3 probes ("Diagnostics round 2", section 0), grouped by the symptom they diagnose.
Run a group only when a core or explainability result in diagnostics.py looks off.

    python -m evaluation.debug speed topology     (from the project root; any of the groups below)

    speed     shock speed or Mn wrong: |psi_x| vs local s error (4A.4), front angle vs exact (4A.6),
              phi_hat vs signed distance (4A.5), Mn_raw vs exact Mn (fig. 12)
    topology  wrong or flickering fronts: front and shock counts over (r, t) (fig. 11),
              precision and recall vs t (4A.3), label margin (4A.2)
    rh        RH arm misbehaves: eval-time Mn gate, no overwrite, oracle s (4D)
    blend     hard vs soft gap: soft blend at eval over eps (4D)
    zones     spikes, zone confusion, dead capacity: zone identity vs r (4B.2), usage (4B.3),
              area vs t (4B.4), per-zone error maps (4C.2), disagreement map (4C.3), FiLM and codes (4C.5)
    locate    where the error is: |W - W_ex| maps (fig. 10), front-point traces and sharpness (4C.4)
    latent    swap test and traversals (plan §9.4)

Training curves (loss terms, eps, grad norms, occupancy) are in TensorBoard, not here.
Reuses run_case, evaluate and the front matching from diagnostics.py. Output goes to results/debug/.
'''

import sys
from contextlib import contextmanager
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch

import model.neural_chart as nc
import evaluation.diagnostics as D
from evaluation.diagnostics import (T_TRAIN, T_MIN, DX, TOL, X0, XC, XQ, COLORS, DEVICE,
                                    run_case, rel_l1, picks, all_cases, levelset_arms,
                                    region_ids, exact_region, cons_to_prim)

OUT = D.ROOT / 'results' / 'debug'
CELL = 2 / 512                                                     # one cell in x~ units


# ======================================================================================
# Helpers
# ======================================================================================

def exact_points(case):
    '''[T, 2048, 3] exact point values at the Gauss points.'''
    reg, t = region_ids(case), case['t'][:, None]
    W = np.zeros(reg.shape + (3,))
    for k in range(5):
        W[reg == k] = exact_region(case, k, XQ[None], t)[reg == k]
    return W


@torch.no_grad()
def phi_hat_at(model, encoder, r, xt):
    '''phi_hat [P, K] at points xt [P, 2] = (x~, t~).'''
    z_geo, z_field = encoder(torch.tensor([[r]], device=DEVICE))
    xt = torch.as_tensor(xt, dtype=torch.float32, device=DEVICE)
    out = model(xt, z_geo.expand(len(xt), -1), z_field.expand(len(xt), -1))
    return out['phi_hat'].cpu().numpy()


def split_l1(model, encoder, ds, **kw):
    '''Mean relative L1 against exact over the cases of a split, t <= T_train; kw goes to run_case.'''
    use = ds.t <= 1 + 1e-6
    return np.mean([rel_l1(run_case(model, encoder, float(ds.r[c]), ds.t[use], ds.x_edges, **kw)['W'],
                           cons_to_prim(ds.U_exact[c][use].double().numpy())) for c in range(ds.C)])


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


# ======================================================================================
# speed: shock speed or Mn wrong
# ======================================================================================

def psi_x_at_fronts(model, encoder, case, h=1e-3):
    '''|psi_x| (psi = phi_hat_b - phi_hat_a) at every front of every snapshot, by central difference in x~.'''
    xs, ts, a, b, owner = [], [], [], [], []
    for j, fr in enumerate(case['fronts']):
        xs.append(2 * fr['x'] - 1)
        ts.append(np.full(len(fr['x']), case['t'][j] / T_TRAIN))
        a.append(fr['a']), b.append(fr['b']), owner.append(np.full(len(fr['x']), j))
    xs, ts, a, b = map(np.concatenate, (xs, ts, a, b))
    rows = np.arange(len(xs))
    ph = phi_hat_at(model, encoder, case['r'], np.stack([np.r_[xs + h, xs - h], np.r_[ts, ts]], -1))
    psi = lambda p: p[rows, b] - p[rows, a]
    return np.abs(psi(ph[:len(xs)]) - psi(ph[len(xs):])) / (2 * h), np.concatenate(owner)


def group_speed(res, nets, data):
    arms = levelset_arms(res)
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    for name in arms:
        model, encoder = nets[name]
        for split, mk in [('test_id', 'o'), ('test_ood', 'x')]:
            px_all, px_s, err_s = [], [], []
            for case in res[name][split]:
                px, owner = psi_x_at_fronts(model, encoder, case)
                px_all.append(px)
                S, idx = case['speeds'][3], case['match']['idx'][:, 1]
                start = np.searchsorted(owner, np.arange(len(case['t'])))          # first front of each snapshot
                for j in np.flatnonzero((case['t'] >= T_MIN) & (idx >= 0)):
                    px_s.append(px[start[j] + idx[j]])
                    err_s.append(abs(case['fronts'][j]['speed'][idx[j]] - S) / S)
            # 4A.4: |psi_x| histogram (all fronts) and against the local s error (matched shocks)
            ax[0].hist(np.concatenate(px_all), bins=np.geomspace(1e-3, 1e3, 60), histtype='step',
                       color=COLORS[name], ls='-' if split == 'test_id' else '--', label=f'{name}, {split}')
            ax[1].loglog(px_s, err_s, mk, ms=2, color=COLORS[name])

            # fig. 12: Mn_raw of detected shocks vs exact Mn, by the exact wave they sit at
            for case in res[name][split]:
                for j, fr in enumerate(case['fronts']):
                    sh = fr['is_shock']
                    at_shock = np.abs(fr['x'][sh] - case['x_ex'][j, 1]) < np.abs(fr['x'][sh] - case['x_ex'][j, 0])
                    ax[3].scatter(np.full(sh.sum(), case['Mn_ex']), fr['Mn_raw'][sh], s=2,
                                  c=np.where(at_shock, 'C3', 'C0'))

    # 4A.6: front angle in (x~, t~) minus the exact wave's, along the matched shock and contact
    for name in arms:
        for (split, c, title), ls in zip(picks(res), ['-', '--', ':']):
            case = res[name][split][c]
            for w, col in [(0, 'C0'), (1, 'C3')]:
                idx = case['match']['idx'][:, w]
                ok = np.flatnonzero(idx >= 0)
                d = [case['fronts'][j]['dxdt'][idx[j]] for j in ok]
                exact = case['speeds'][2 + w] * 2 * T_TRAIN                         # dx~/dt~
                ax[2].plot(case['t'][ok], np.degrees(np.arctan(d) - np.arctan(exact)), ls, color=col, lw=0.8)
    ax[0].set_xscale('log'), ax[0].set_xlabel('|ψ_x| at fronts'), ax[0].legend(fontsize=7)
    ax[1].set_xlabel('|ψ_x| at the matched shock'), ax[1].set_ylabel('local |s − S| / S')
    ax[2].set_xlabel('t'), ax[2].set_ylabel('front angle − exact (deg)')
    ax[2].set_title('contact (blue), shock (red); low, mid, OOD-far', fontsize=9)
    ax[3].plot([1.05, 2.0], [1.05, 2.0], 'k--', lw=1)
    ax[3].axhline(1, color='k', lw=0.5)
    ax[3].set_xlabel('exact shock Mn'), ax[3].set_ylabel('Mn_raw of detected shocks')
    ax[3].set_title('at exact shock (red) / contact (blue)', fontsize=9)
    save(fig, 'speed.png')

    # 4A.5: psi = phi_hat_b - phi_hat_a against the signed distance to the exact shock (mid r case)
    split, c, _ = picks(res)[1]
    fig, ax = plt.subplots(1, len(arms), figsize=(5 * len(arms), 4), squeeze=False)
    for a, name in zip(ax[0], arms):
        case = res[name][split][c]
        for tt in (0.04, 0.10, 0.16):
            j = int(np.argmin(np.abs(case['t'] - tt)))
            i = case['match']['idx'][j, 1]
            if i < 0:
                continue
            fr = case['fronts'][j]
            sd = 2 * (XQ - case['x_ex'][j, 1])                                     # x~ units
            near = np.abs(sd) < 10 * CELL
            psi = case['phi_hat'][j][:, fr['b'][i]] - case['phi_hat'][j][:, fr['a'][i]]
            a.plot(sd[near] / CELL, psi[near] / CELL, label=f't = {case["t"][j]:.2f}')
        a.plot([-10, 10], [-10, 10], 'k--', lw=1)
        a.set_xlabel('signed distance to exact shock (cells)'), a.set_ylabel('ψ (cells)')
        a.set_title(f'{name}, r = {case["r"]:.2f}', fontsize=10)
        a.legend(fontsize=7)
    save(fig, 'speed_phi_vs_distance.png')


# ======================================================================================
# topology: wrong or flickering fronts
# ======================================================================================

def group_topology(res, nets, data):
    # fig. 11: fronts - 2 and shocks - 1 per snapshot, cases sorted by r
    fig, ax = plt.subplots(len(res), 2, figsize=(11, 3.5 * len(res)), squeeze=False)
    for i, name in enumerate(res):
        cases = sorted(all_cases(res, name), key=lambda c: c['r'])
        t, r = cases[0]['t'], [c['r'] for c in cases]
        n_f = np.array([[len(f['x']) for f in c['fronts']] for c in cases]) - 2
        ext = [t[0], t[-1], 0, len(cases)]
        ax[i, 0].imshow(n_f, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-3, vmax=3, extent=ext)
        ax[i, 0].set_title(f'{name}: fronts − 2', fontsize=10)
        if cases[0]['levelset']:
            n_s = np.array([[f['is_shock'].sum() for f in c['fronts']] for c in cases]) - 1
            im = ax[i, 1].imshow(n_s, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-3, vmax=3, extent=ext)
            ax[i, 1].set_title(f'{name}: shocks − 1', fontsize=10)
            fig.colorbar(im, ax=ax[i, 1])
        for a in ax[i]:
            a.set_yticks(np.arange(0, len(r), 4) + 0.5, [f'{x:.1f}' for x in r[::4]])
            a.set_ylabel('r'), a.set_xlabel('t')
            a.axvline(T_TRAIN, color='k', ls=':', lw=0.8)
    save(fig, 'topology_counts.png')

    # 4A.3: precision and recall vs t (hit = within TOL cells of an exact contact or shock)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for name in res:
        for split, ls in [('test_id', '-'), ('test_ood', '--')]:
            prec, rec = [], []
            for case in res[name][split]:
                p, q = np.full(len(case['t']), np.nan), np.full(len(case['t']), np.nan)
                for j, fr in enumerate(case['fronts']):
                    d = np.abs(fr['x'][:, None] - case['x_ex'][j][None]) / DX < TOL
                    if len(fr['x']):
                        p[j] = d.any(1).mean()
                    q[j] = d.any(0).mean()
                prec.append(p), rec.append(q)
            t = res[name][split][0]['t']
            ax[0].plot(t, np.nanmean(prec, 0), ls, color=COLORS[name], label=f'{name}, {split}')
            ax[1].plot(t, np.nanmean(rec, 0), ls, color=COLORS[name])
    for a, title in zip(ax, ['precision', 'recall']):
        a.axvline(T_TRAIN, color='k', ls=':')
        a.set_xlabel('t'), a.set_title(title)
    ax[0].legend(fontsize=7)
    save(fig, 'topology_precision_recall.png')

    # 4A.2: label margin, top-1 minus top-2 phi_hat, along x (mid r and OOD-far)
    arms = levelset_arms(res)
    cases = picks(res)[1:]
    fig, ax = plt.subplots(len(arms), 2, figsize=(11, 3.5 * len(arms)), squeeze=False)
    for i, name in enumerate(arms):
        for a, (split, c, title) in zip(ax[i], cases):
            case = res[name][split][c]
            for tt in (0.04, 0.10, 0.16, 0.22):
                j = int(np.argmin(np.abs(case['t'] - tt)))
                top = np.sort(case['phi_hat'][j], axis=-1)
                a.semilogy(XQ, (top[:, -1] - top[:, -2]) / CELL, lw=0.8, label=f't = {case["t"][j]:.2f}')
            a.axhline(1, color='k', lw=0.5)
            a.set_xlabel('x'), a.set_ylabel('margin (cells)')
            a.set_title(f'{name}, {title} (r = {case["r"]:.2f})', fontsize=10)
        ax[i, 0].legend(fontsize=7)
    save(fig, 'topology_label_margin.png')


# ======================================================================================
# rh and blend: eval-time swaps (4D)
# ======================================================================================

@contextmanager
def oracle_s(S_box):
    '''Inside: fronts within TOL cells of the exact shock get s = n S_exact (field units) before
    detection and the RH map. S_box[0] holds the current case's exact shock speed.'''
    orig = nc.find_fronts

    def patched(phi_fn, z_geo, t, *args, **kw):
        fr = orig(phi_fn, z_geo, t, *args, **kw)
        x_s = X0 + S_box[0] * t[fr['g']] * T_TRAIN
        near = ((fr['x'] + 1) / 2 - x_s).abs() < TOL * DX
        fr['s'] = torch.where(near, fr['n'] * S_box[0], fr['s'])
        return fr

    nc.find_fronts = patched
    try:
        yield
    finally:
        nc.find_fronts = orig


def group_rh(res, nets, data):
    test_id, test_ood, _ = data
    lines = [f'{"arm":<12}{"variant":<22}{"ID L1":>10}{"OOD L1":>10}']
    for name in levelset_arms(res):
        model, encoder = nets[name]
        mode, gate = model.jump_mode, model.mn_gate
        variants = [('as trained', {}), ('Mn gate 1.0', {'mn_gate': 1.0}), ('Mn gate 1.2', {'mn_gate': 1.2})]
        if mode == 'rh':
            variants.append(('no overwrite', {'jump_mode': 'free'}))
        for label, attrs in variants + [('oracle s', None)]:
            for k, v in (attrs or {}).items():
                setattr(model, k, v)
            if attrs is None:                           # oracle s: one case at a time, its own S
                S_box, l1 = [0.0], []
                with oracle_s(S_box):
                    for ds in (test_id, test_ood):
                        use = ds.t <= 1 + 1e-6
                        vals = []
                        for c in range(ds.C):
                            S_box[0] = D.exact_waves(float(ds.r[c]))[0][3]
                            vals.append(rel_l1(run_case(model, encoder, float(ds.r[c]), ds.t[use], ds.x_edges)['W'],
                                               cons_to_prim(ds.U_exact[c][use].double().numpy())))
                        l1.append(np.mean(vals))
            else:
                l1 = [split_l1(model, encoder, ds) for ds in (test_id, test_ood)]
            model.jump_mode, model.mn_gate = mode, gate
            lines.append(f'{name:<12}{label:<22}{l1[0]:>10.4f}{l1[1]:>10.4f}')
    text = '\n'.join(lines)
    print(text)
    (OUT / 'rh_swaps.txt').write_text(text + '\n')


def group_blend(res, nets, data, eps_list=(0.002, 0.004, 0.01, 0.03, 0.1)):
    test_id, test_ood, _ = data
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in levelset_arms(res):
        model, encoder = nets[name]
        for ds, ls, split in [(test_id, '-', 'test_id'), (test_ood, '--', 'test_ood')]:
            soft = [split_l1(model, encoder, ds, blend='soft', eps=e) for e in eps_list]
            ax.plot(eps_list, soft, ls, marker='o', color=COLORS[name], label=f'{name}, {split}')
            ax.axhline(split_l1(model, encoder, ds), ls=ls, color=COLORS[name], lw=0.6)
    ax.set_xscale('log'), ax.set_yscale('log')
    ax.set_xlabel('ε (x~ units; training uses 0.004)'), ax.set_ylabel('relative L1 vs exact')
    ax.set_title('soft blend (markers) vs hard (thin lines)', fontsize=10)
    ax.legend(fontsize=7)
    save(fig, 'blend.png')


# ======================================================================================
# zones: spikes, zone confusion, dead capacity
# ======================================================================================

@torch.no_grad()
def film_stats(model, encoder, r):
    '''Region codes: pairwise distances. FiLM: mean and std of gamma, beta per zone and layer.'''
    fn = model.FieldNet
    e = fn.embed().cpu()                                                            # [K, N_e]
    lines = ['region-code distances:', np.array2string(torch.cdist(e, e).numpy(), precision=3)]
    z_field = encoder(torch.tensor([[r]], device=DEVICE))[1]
    for li, layer in enumerate(l for l in fn.layers if hasattr(l, 'hyper')):
        cond = torch.cat([z_field.expand(len(e), -1), e.to(DEVICE)], dim=-1)
        gamma, beta = layer.hyper(cond).chunk(2, dim=-1)
        for k in range(len(e)):
            lines.append(f'layer {li} zone {k}: gamma {gamma[k].mean():+.3f} ± {gamma[k].std():.3f}, '
                         f'beta {beta[k].mean():+.3f} ± {beta[k].std():.3f}')
    return lines


def group_zones(res, nets, data):
    arms = levelset_arms(res)
    fig, ax = plt.subplots(len(arms), 3, figsize=(16, 3.8 * len(arms)), squeeze=False)
    text = []
    for i, name in enumerate(arms):
        cases = sorted(all_cases(res, name), key=lambda c: c['r'])
        r = [c['r'] for c in cases]

        # 4B.2: dominant zone of each exact region against r
        for k, reg in enumerate(D.REGIONS):
            ax[i, 0].plot(r, [c['conf'][k].argmax() + 0.08 * (k - 2) for c in cases], 'o-', ms=3, lw=0.8, label=reg)
        ax[i, 0].set_xscale('log'), ax[i, 0].set_xlabel('r'), ax[i, 0].set_ylabel('dominant zone id')
        ax[i, 0].set_title(f'{name}: zone identity across r', fontsize=10)
        ax[i, 0].legend(fontsize=7)

        # 4B.3: share of all (x, t) points each zone is the label of
        use = sum(c['conf'].sum(0) for c in cases)
        ax[i, 1].bar(range(len(use)), use / use.sum())
        ax[i, 1].set_xlabel('zone id'), ax[i, 1].set_title(f'{name}: zone usage', fontsize=10)

        # 4B.4: zone area against t, low r / mid r / OOD-far
        for (split, c, title), ls in zip(picks(res), ['-', '--', ':']):
            lab = res[name][split][c]['labels']
            for k in range(len(use)):
                ax[i, 2].plot(res[name][split][c]['t'], (lab == k).mean(1), ls, color=f'C{k}', lw=0.8)
        ax[i, 2].set_xlabel('t'), ax[i, 2].set_ylabel('area fraction')
        ax[i, 2].set_title(f'{name}: zone area vs t (low, mid, OOD-far)', fontsize=10)

        model, encoder = nets[name]
        text += [f'== {name}'] + film_stats(model, encoder, res[name]['test_id'][picks(res)[1][1]]['r'])
    save(fig, 'zones.png')
    (OUT / 'zones_film.txt').write_text('\n'.join(text) + '\n')

    # 4C.2 per-zone |W_k - W_exact| in (x, t) with the zone's support outlined; 4C.3 disagreement map
    split, c, title = picks(res)[1]
    for name in arms:
        case = res[name][split][c]
        Wr, lab, t = case['W_regions'], case['labels'], case['t']
        W_ex = exact_points(case)
        K = Wr.shape[2]
        fig, ax = plt.subplots(1, K + 1, figsize=(4 * (K + 1), 4), sharey=True)
        ext = [0, 1, t[0], t[-1]]
        for k in range(K):
            ax[k].imshow(np.log10(np.abs(Wr[:, :, k] - W_ex).sum(-1) + 1e-6), origin='lower', aspect='auto',
                         extent=ext, vmin=-4, vmax=0)
            ax[k].contour(XQ, t, (lab == k).astype(float), levels=[0.5], colors='w', linewidths=0.8)
            ax[k].set_title(f'zone {k}: log10 |W_k − W_ex|', fontsize=9)
        present = np.stack([np.isin(np.arange(K), np.unique(l)) for l in lab])     # [T, K] zones used at each t
        Wm = np.where(present[:, None, :, None], Wr, np.nan)
        dis = (np.nanmax(Wm, 2) - np.nanmin(Wm, 2)).sum(-1)                        # max_k,l |W_k - W_l|, summed
        im = ax[K].imshow(np.log10(dis + 1e-6), origin='lower', aspect='auto', extent=ext, vmin=-4, vmax=0)
        ax[K].set_title('disagreement (log10)', fontsize=9)
        fig.colorbar(im, ax=ax[K])
        for a in ax:
            a.set_xlabel('x')
        ax[0].set_ylabel('t')
        fig.suptitle(f'{name}, {title} (r = {case["r"]:.2f})')
        save(fig, f'zones_error_maps_{name}.png')


# ======================================================================================
# locate: where the error is
# ======================================================================================

def group_locate(res, nets, data):
    # fig. 10: |W - W_ex| in (x, t) per arm, model fronts (dots) and exact waves (dashed)
    cases = picks(res)
    fig, ax = plt.subplots(len(cases), len(res), figsize=(5 * len(res), 4 * len(cases)), squeeze=False)
    for col, name in enumerate(res):
        for row, (split, c, title) in enumerate(cases):
            case, a = res[name][split][c], ax[row, col]
            t = case['t']
            a.imshow(np.log10(np.abs(case['W'] - case['W_ex']).sum(-1) + 1e-6), origin='lower', aspect='auto',
                     extent=[0, 1, t[0], t[-1]], vmin=-4, vmax=0)
            for j, fr in enumerate(case['fronts']):
                a.plot(fr['x'], np.full(len(fr['x']), t[j]), 'w.', ms=0.8)
            for s in case['speeds']:
                a.plot(X0 + s * t, t, 'k--', lw=0.8)
            a.set_xlim(0, 1)
            a.set_title(f'{name}, {title} (r = {case["r"]:.2f})', fontsize=9)
    save(fig, 'locate_error_maps.png')

    # 4C.4: W_a, W_b at the matched contact (rho) and shock (p) vs the exact states either side (mid r);
    # sharpness = recovered fraction of the exact jump, median over snapshots, per arm and split
    arms = levelset_arms(res)
    split, c, title = picks(res)[1]
    fig, ax = plt.subplots(len(arms), 2, figsize=(11, 3.5 * len(arms)), squeeze=False)
    lines = [f'{"arm":<12}{"split":<10}{"contact":>10}{"shock":>10}']
    for i, name in enumerate(arms):
        for sp in ('test_id', 'test_ood'):
            score = {0: [], 1: []}
            for case in res[name][sp]:
                sol = case['sol']
                exact = {0: (sol.rho_star_L, sol.rho_star_R, 0), 1: (sol.P_star, D.RIGHT[2], 2)}   # (left, right, var)
                for w in (0, 1):
                    left, right, v = exact[w]
                    for j in D.snapshots(case):
                        fr, f = case['fronts'][j], case['match']['idx'][j, w]
                        W_a = np.where(fr['up_is_a'][f], fr['W_up'][f], fr['W_dn'][f])
                        W_b = np.where(fr['up_is_a'][f], fr['W_dn'][f], fr['W_up'][f])
                        score[w].append(abs(W_a[v] - W_b[v]) / abs(left - right))
                        if sp == split and case is res[name][split][c]:
                            ax[i, w].plot(case['t'][j], W_a[v], 'o', ms=2, color='C0')
                            ax[i, w].plot(case['t'][j], W_b[v], 'o', ms=2, color='C1')
                    if sp == split and case is res[name][split][c]:
                        ax[i, w].axhline(left, color='C0', ls='--', lw=0.8)
                        ax[i, w].axhline(right, color='C1', ls='--', lw=0.8)
            lines.append(f'{name:<12}{sp:<10}{np.median(score[0]):>10.3f}{np.median(score[1]):>10.3f}')
        ax[i, 0].set_title(f'{name}: ρ at contact, W_a (blue) W_b (orange), exact dashed', fontsize=9)
        ax[i, 1].set_title(f'{name}: p at shock', fontsize=9)
        for a in ax[i]:
            a.set_xlabel('t')
    save(fig, 'locate_front_traces.png')
    text = '\n'.join(['fraction of the exact jump recovered at the front (median)'] + lines)
    print(text)
    (OUT / 'locate_sharpness.txt').write_text(text + '\n')


# ======================================================================================
# latent: swap test and traversals
# ======================================================================================

@torch.no_grad()
def group_latent(res, nets, data, n_pairs=10, n_t=30):
    test_id = data[0]
    rng = np.random.default_rng(0)
    use = test_id.t <= 1 + 1e-6
    lines = [f'{"arm":<12}{"front err vs A (cells)":>24}{"star-state err vs B":>22}']
    for name in levelset_arms(res):
        model, encoder = nets[name]
        z_geo, z_field = encoder(test_id.r[:, None].to(DEVICE))
        cases = res[name]['test_id']

        # swap test: decode (z_geo of A, z_field of B); fronts vs A's, star states vs B's
        f_err, s_err = [], []
        for A, B in rng.choice(test_id.C, size=(n_pairs, 2), replace=True):
            if A == B:
                continue
            out = run_case(model, encoder, 0.0, test_id.t[use], test_id.x_edges, z=(z_geo[A:A + 1], z_field[B:B + 1]))
            cA, cB = cases[A], cases[B]
            for j, fr in enumerate(out['fronts']):
                if cA['t'][j] >= T_MIN and len(fr['x']):
                    f_err.append(np.abs(fr['x'][:, None] - cA['x_ex'][j][None]).min(0).mean() / DX)
                edges = X0 + cA['speeds'] * cA['t'][j]                              # A's regions
                for lo, hi, W_star in [(edges[1], edges[2], (cB['sol'].rho_star_L, cB['sol'].u_star, cB['sol'].P_star)),
                                       (edges[2], edges[3], cB['W_star_R'])]:
                    inside = (XC > lo + 2 * DX) & (XC < hi - 2 * DX)
                    if inside.any():
                        s_err.append((np.abs(out['W'][j, inside] - W_star).sum(-1) / np.abs(W_star).sum()).mean())
        lines.append(f'{name:<12}{np.mean(f_err):>24.2f}{np.mean(s_err):>22.4f}')

        # traversals: each z_geo coordinate over +-2 std around the mid-r case, region maps
        mid = picks(res)[1][1]
        std = z_geo.std(0)
        t_sub = test_id.t[np.linspace(0, len(test_id.t) - 1, n_t).astype(int)]
        steps = (-2, -1, 0, 1, 2)
        fig, ax = plt.subplots(z_geo.shape[1], len(steps), figsize=(3 * len(steps), 2.5 * z_geo.shape[1]),
                               squeeze=False, sharex=True, sharey=True)
        for k in range(z_geo.shape[1]):
            for col, s in enumerate(steps):
                zg = z_geo[mid:mid + 1].clone()
                zg[0, k] += s * std[k]
                lab = run_case(model, encoder, 0.0, t_sub, test_id.x_edges, z=(zg, z_field[mid:mid + 1]))['labels']
                ax[k, col].imshow(lab, origin='lower', aspect='auto', cmap='tab10', vmin=-0.5, vmax=9.5,
                                  extent=[0, 1, float(t_sub[0]) * T_TRAIN, float(t_sub[-1]) * T_TRAIN])
                ax[k, col].set_title(f'z_geo[{k}] {s:+d} std', fontsize=8)
        fig.suptitle(f'{name}: z_geo traversals around r = {cases[mid]["r"]:.2f}')
        save(fig, f'latent_traversals_{name}.png')

    text = '\n'.join(lines)
    print(text)
    (OUT / 'latent_swap.txt').write_text(text + '\n')


# ======================================================================================
# Master
# ======================================================================================

GROUPS = {'speed': group_speed, 'topology': group_topology, 'rh': group_rh, 'blend': group_blend,
          'zones': group_zones, 'locate': group_locate, 'latent': group_latent}


def main(groups):
    unknown = [g for g in groups if g not in GROUPS]
    if not groups or unknown:
        print(__doc__)
        return
    os.makedirs(OUT, exist_ok=True)
    res, nets, data = D.load_all()
    for g in groups:
        print(f'== {g}')
        GROUPS[g](res, nets, data)


if __name__ == '__main__':
    main(sys.argv[1:])
