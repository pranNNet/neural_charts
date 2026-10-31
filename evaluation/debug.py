'''
Tier 3 probes ("Diagnostics round 2", section 0), grouped by the symptom they diagnose.
Run a group only when a core or explainability result in diagnostics.py looks off.

    python -m evaluation.debug speed topology     (from the project root; any of the groups below)

    speed     shock speed or Mn wrong: |psi_x| vs advection s error (4A.4), advection speed error vs t (4A.6),
              psi / |psi_x| vs signed distance (4A.5), gated Mn vs exact Mn (fig. 12)
    topology  wrong or flickering fronts: front, distinct-front and shock counts over (r, t) (fig. 11),
              precision and recall vs t (4A.3), label margin (4A.2)
    rh        RH arm misbehaves: eval-time Mn gate, no constraint, oracle s (4D)
    zones     spikes, zone confusion, dead capacity: zone identity vs r (4B.2), usage (4B.3),
              area vs t (4B.4), per-zone error maps (4C.2), disagreement map (4C.3), FiLM and codes (4C.5)
    locate    where the error is: |W - W_ex| maps (fig. 10), front-point traces and sharpness (4C.4)
    latent    swap test and traversals, one step from the FV data (plan §9.4)

Training curves (loss terms, eps, grad norms, occupancy) are in TensorBoard, not here.
Reuses rollout, one_step, evaluate and the front matching from diagnostics.py. Everything works on the
rollouts from t0, except the latent group. Output goes to results/debug/.
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
from model.region import build_cond
from evaluation.diagnostics import (T_TRAIN, T_MIN, DT, DX, CELL, TOL, X0, XC, COLORS, DEVICE,
                                    rollout, one_step, start_index, rel_l1, picks, all_cases, levelset_arms,
                                    region_ids, exact_region, cons_to_prim)

OUT = D.ROOT / 'results' / 'debug'

# ======================================================================================
# Helpers
# ======================================================================================

def exact_points(case):
    '''[T, 512, 3] exact point values at the cell centres.'''
    reg, t = region_ids(case), case['t'][:, None]
    W = np.zeros(reg.shape + (3,))
    for k in range(5):
        W[reg == k] = exact_region(case, k, XC[None], t)[reg == k]
    return W

def train_band(ds):
    '''(n0, n_steps): rollout input index and the steps from t0 to T_train.'''
    n0 = start_index(ds)
    return n0, int((ds.t[n0 + 1:] <= 1 + 1e-6).sum())

def split_l1(model, encoder, ds, cases=None):
    '''Mean relative L1 against exact of the rollouts t0 -> T_train.
    cases: case indices to run one at a time (oracle s), else all cases at once.'''
    n0, n = train_band(ds)
    x_cell = ds.x_centers.to(DEVICE)
    W_ex = lambda c: cons_to_prim(ds.U_exact[c, n0 + 1:n0 + 1 + n].double().numpy())
    if cases is None:
        return np.mean([rel_l1(run['W'], W_ex(c)) for c, run in
                        enumerate(rollout(model, encoder, ds.W[:, n0], n, x_cell))])
    return np.mean([rel_l1(rollout(model, encoder, ds.W[c:c + 1, n0], n, x_cell)[0]['W'], W_ex(c))
                    for c in cases])

def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)

# ======================================================================================
# speed: shock speed or Mn wrong
# ======================================================================================

def psi_x_at_fronts(case):
    '''|psi_x| (psi = phi_hat_b - phi_hat_a, x~ units) at every front of every snapshot, by central
    difference of the rollout's phi_hat across the two cells around the front.'''
    px, owner = [], []
    for j, fr in enumerate(case['fronts']):
        i = np.clip(np.rint(fr['x'] / DX - 0.5).astype(int), 1, 510)              # cell nearest the front
        ph = case['phi_hat'][j]
        psi = lambda m: ph[m, fr['b']] - ph[m, fr['a']]
        px.append(np.abs(psi(i + 1) - psi(i - 1)) / (2 * CELL))
        owner.append(np.full(len(fr['x']), j))
    return np.concatenate(px), np.concatenate(owner)

def group_speed(res, nets, data):
    arms = levelset_arms(res)
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    for name in arms:
        for split, mk in [('test_id', 'o'), ('test_ood', 'x')]:
            px_all, px_s, err_s = [], [], []
            for case in res[name][split]:
                px, owner = psi_x_at_fronts(case)
                px_all.append(px)
                S, idx = case['speeds'][3], case['match']['idx'][:, 1]
                start = np.searchsorted(owner, np.arange(len(case['t'])))          # first front of each snapshot
                for j in np.flatnonzero((case['t'] <= T_TRAIN + 1e-9) & (idx >= 0)):
                    px_s.append(px[start[j] + idx[j]])
                    err_s.append(abs(case['fronts'][j]['speed'][idx[j]] - S) / S)
            # 4A.4: |psi_x| histogram (all fronts) and against the local s error (matched shocks)
            ax[0].hist(np.concatenate(px_all), bins=np.geomspace(1e-3, 1e3, 60), histtype='step',
                       color=COLORS[name], ls='-' if split == 'test_id' else '--', label=f'{name}, {split}')
            ax[1].loglog(px_s, err_s, mk, ms=2, color=COLORS[name])

            # fig. 12: gated Mn of detected shocks vs exact Mn, by the exact wave they sit at
            for case in res[name][split]:
                for j, fr in enumerate(case['fronts']):
                    sh = fr['is_shock']
                    at_shock = np.abs(fr['x'][sh] - case['x_ex'][j, 1]) < np.abs(fr['x'][sh] - case['x_ex'][j, 0])
                    ax[3].scatter(np.full(sh.sum(), case['Mn_ex']), fr['Mn'][sh], s=2,
                                  c=np.where(at_shock, 'C3', 'C0'))

    # 4A.6: advection speed minus the exact wave's, along the matched contact and shock (a rollout drifts if it is off)
    for name in arms:
        for (split, c, title), ls in zip(picks(res), ['-', '--', ':']):
            case = res[name][split][c]
            for w, col in [(0, 'C0'), (1, 'C3')]:
                idx = case['match']['idx'][:, w]
                ok = np.flatnonzero(idx >= 0)
                d = np.array([case['fronts'][j]['speed'][idx[j]] for j in ok])
                exact = case['speeds'][2 + w]
                ax[2].plot(case['t'][ok], 100 * (d - exact) / exact, ls, color=col, lw=0.8)
    ax[0].set_xscale('log'), ax[0].set_xlabel('|ψ_x| at fronts'), ax[0].legend(fontsize=7)
    ax[1].set_xlabel('|ψ_x| at the matched shock (t ≤ T_train)'), ax[1].set_ylabel('advection |s − S| / S')
    ax[2].set_xlabel('t'), ax[2].set_ylabel('advection speed s − exact (%)')
    ax[2].axvline(T_TRAIN, color='k', ls=':', lw=0.8)
    ax[2].set_title('contact (blue), shock (red); low, mid, OOD-far', fontsize=9)
    ax[3].plot([1.05, 2.0], [1.05, 2.0], 'k--', lw=1)
    ax[3].axhline(1, color='k', lw=0.5)
    ax[3].set_xlabel('exact shock Mn'), ax[3].set_ylabel('gated Mn of detected shocks')
    ax[3].set_title('at exact shock (red) / contact (blue)', fontsize=9)
    save(fig, 'speed.png')

    # 4A.5: psi = phi_b - phi_a, divided by its slope at the matched front (phi is raw, not a distance),
    # against the signed distance to the exact shock (mid r case): linear across the band if on the diagonal
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
            sd = 2 * (XC - case['x_ex'][j, 1])                                     # x~ units
            near = np.abs(sd) < 10 * CELL
            psi = case['phi_hat'][j][:, fr['b'][i]] - case['phi_hat'][j][:, fr['a'][i]]
            m = np.clip(int(np.rint(fr['x'][i] / DX - 0.5)), 1, 510)                # cell nearest the front
            slope = max(abs(psi[m + 1] - psi[m - 1]) / (2 * CELL), 1e-12)
            a.plot(sd[near] / CELL, psi[near] / slope / CELL, label=f't = {case["t"][j]:.2f}')
        a.plot([-10, 10], [-10, 10], 'k--', lw=1)
        a.set_xlabel('signed distance to exact shock (cells)'), a.set_ylabel('ψ / |ψ_x| at the front (cells)')
        a.set_title(f'{name}, r = {case["r"]:.2f}', fontsize=10)
        a.legend(fontsize=7)
    save(fig, 'speed_phi_vs_distance.png')

# ======================================================================================
# topology: wrong or flickering fronts
# ======================================================================================

def distinct(x):
    '''Number of fronts after merging fronts closer than TOL cells to their neighbour (slivers count once).'''
    return int((np.diff(np.sort(x)) >= TOL * DX).sum() + 1) if len(x) else 0

def group_topology(res, nets, data):
    # fig. 11: fronts - 2, distinct fronts - 2 (after merging within TOL) and shocks - 1 per snapshot,
    # cases sorted by r. fronts - distinct = sliver fronts sitting next to another front.
    fig, ax = plt.subplots(len(res), 3, figsize=(16, 3.5 * len(res)), squeeze=False)
    lines = [f'{"arm":<12}{"split":<10}{"fronts/snap":>13}{"distinct/snap":>15}{"slivers/snap":>14}']
    for i, name in enumerate(res):
        cases = sorted(all_cases(res, name), key=lambda c: c['r'])
        t, r = cases[0]['t'], [c['r'] for c in cases]
        n_f = np.array([[len(f['x']) for f in c['fronts']] for c in cases]) - 2
        n_d = np.array([[distinct(f['x']) for f in c['fronts']] for c in cases]) - 2
        ext = [t[0], t[-1], 0, len(cases)]
        ax[i, 0].imshow(n_f, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-3, vmax=3, extent=ext)
        ax[i, 0].set_title(f'{name}: fronts − 2', fontsize=10)
        ax[i, 1].imshow(n_d, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-3, vmax=3, extent=ext)
        ax[i, 1].set_title(f'{name}: distinct fronts (merged within {TOL} cells) − 2', fontsize=10)
        if cases[0]['levelset']:
            n_s = np.array([[f['is_shock'].sum() for f in c['fronts']] for c in cases]) - 1
            im = ax[i, 2].imshow(n_s, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-3, vmax=3, extent=ext)
            ax[i, 2].set_title(f'{name}: shocks − 1', fontsize=10)
            fig.colorbar(im, ax=ax[i, 2])
        for split in ('test_id', 'test_ood'):
            sel = [c for c in res[name][split]]
            nf = np.mean([len(f['x']) for c in sel for f, tj in zip(c['fronts'], c['t']) if tj >= T_MIN])
            nd = np.mean([distinct(f['x']) for c in sel for f, tj in zip(c['fronts'], c['t']) if tj >= T_MIN])
            lines.append(f'{name:<12}{split:<10}{nf:>13.2f}{nd:>15.2f}{nf - nd:>14.2f}')
        for a in ax[i]:
            a.set_yticks(np.arange(0, len(r), 4) + 0.5, [f'{x:.1f}' for x in r[::4]])
            a.set_ylabel('r'), a.set_xlabel('t')
            a.axvline(T_TRAIN, color='k', ls=':', lw=0.8)
    save(fig, 'topology_counts.png')
    text = '\n'.join([f'fronts per snapshot, t >= {T_MIN} (slivers = fronts within {TOL} cells of another)'] + lines)
    print(text)
    (OUT / 'topology_counts.txt').write_text(text + '\n')

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

    # 4A.2: label margin m = top-1 minus top-2 phi along x, as a distance m / |dm/dx| (phi is raw), mid r and OOD-far
    arms = levelset_arms(res)
    cases = picks(res)[1:]
    fig, ax = plt.subplots(len(arms), 2, figsize=(11, 3.5 * len(arms)), squeeze=False)
    for i, name in enumerate(arms):
        for a, (split, c, title) in zip(ax[i], cases):
            case = res[name][split][c]
            for tt in (0.04, 0.10, 0.16, 0.22):
                j = int(np.argmin(np.abs(case['t'] - tt)))
                top = np.sort(case['phi_hat'][j], axis=-1)
                m = top[:, -1] - top[:, -2]
                dist = m / np.maximum(np.abs(np.gradient(m, CELL)), 1e-12)
                a.semilogy(XC, dist / CELL, lw=0.8, label=f't = {case["t"][j]:.2f}')
            a.axhline(1, color='k', lw=0.5)
            a.set_xlabel('x'), a.set_ylabel('margin (cells)')
            a.set_title(f'{name}, {title} (r = {case["r"]:.2f})', fontsize=10)
        ax[i, 0].legend(fontsize=7)
    save(fig, 'topology_label_margin.png')

# ======================================================================================
# rh: eval-time swaps (4D)
# ======================================================================================

@contextmanager
def oracle_s(box):
    '''Inside: band points whose time-n front lies within TOL cells of the exact shock get s = S_exact n
    (field units, along n) before advection and the shock test. RH no longer uses s (its strength comes from
    the pin pressure), so this isolates the advection speed. box = {'S': the case's exact
    shock speed, 't': t_n of the next step}; each model call advances box['t'] by DT, so run one case at a
    time and reset box['t'] to T_MIN before each rollout.'''
    orig = nc.advect

    def patched(frame, d, j, s, dt_x, reach):
        x_s = X0 + box['S'] * box['t']
        near = ((frame['x_foot'][:, 0] + 1) / 2 - x_s).abs() < TOL * DX
        s = torch.where(near, box['S'] * frame['n'][:, 0], s)
        box['t'] += DT
        return orig(frame, d, j, s, dt_x, reach)

    nc.advect = patched
    try:
        yield
    finally:
        nc.advect = orig

def group_rh(res, nets, data):
    test_id, test_ood, _ = data
    lines = [f'{"arm":<12}{"variant":<22}{"ID L1":>10}{"OOD L1":>10}']
    for name in levelset_arms(res):
        model, encoder = nets[name]
        mode, cfg = model.jump_mode, model.shock_cfg
        variants = [('as trained', mode, {}), ('Mn gate 1.0', mode, {'Mn_gate': 1.0}),
                    ('Mn gate 1.2', mode, {'Mn_gate': 1.2})]
        if mode == 'rh':
            variants.append(('no constraint', 'free', {}))
        for label, jump_mode, gate in variants + [('oracle s', mode, None)]:
            model.jump_mode, model.shock_cfg = jump_mode, {**cfg, **(gate or {})}
            if gate is None:                            # oracle s: one case at a time, its own S
                box, l1 = {}, []
                with oracle_s(box):
                    for ds in (test_id, test_ood):
                        vals = []
                        for c in range(ds.C):
                            box.update(S=D.exact_waves(float(ds.r[c]))[0][3], t=T_MIN)
                            vals.append(split_l1(model, encoder, ds, cases=[c]))
                        l1.append(np.mean(vals))
            else:
                l1 = [split_l1(model, encoder, ds) for ds in (test_id, test_ood)]
            model.jump_mode, model.shock_cfg = mode, cfg
            lines.append(f'{name:<12}{label:<22}{l1[0]:>10.4f}{l1[1]:>10.4f}')
    text = '\n'.join(lines)
    print(text)
    (OUT / 'rh_swaps.txt').write_text(text + '\n')

# ======================================================================================
# zones: spikes, zone confusion, dead capacity
# ======================================================================================

@torch.no_grad()
def film_stats(model, encoder, W):
    '''Region codes: pairwise distances. FiLM (trunk, z_field only): mean and std of gamma, beta per layer.
    Coefficients a_k from [z_field; e_k]: mean and std per zone. z_field from the field W [512, 3].'''
    fn = model.FieldNet
    e = fn.embed()                                                                  # [K, N_e]
    lines = ['region-code distances:', np.array2string(torch.cdist(e, e).cpu().numpy(), precision=3)]
    z_field = encoder(torch.as_tensor(W, dtype=torch.float32, device=DEVICE).T[None])[1]   # [1, z]
    for li, layer in enumerate(l for l in fn.layers if hasattr(l, 'hyper')):
        gamma, beta = layer.hyper(z_field).chunk(2, dim=-1)
        lines.append(f'layer {li}: gamma {gamma.mean():+.3f} ± {gamma.std():.3f}, '
                     f'beta {beta.mean():+.3f} ± {beta.std():.3f}')
    a = fn.coeff(build_cond(z_field, e))[0]                                         # [K, N_bases * F]
    for k in range(len(e)):
        lines.append(f'zone {k}: coefficients {a[k].mean():+.3f} ± {a[k].std():.3f}')
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

        # 4B.3: share of all (x, t) cells each zone is the label of
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
        mid = res[name]['test_id'][picks(res)[1][1]]
        j = int(np.argmin(np.abs(mid['t'] - T_TRAIN)))
        text += [f'== {name}, FV state at t = {mid["t"][j]:.2f}'] + film_stats(model, encoder, mid['W_fv'][j])
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
            ax[k].contour(XC, t, (lab == k).astype(float), levels=[0.5], colors='w', linewidths=0.8)
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
    lines = [f'{"arm":<12}{"split":<10}{"contact":>10}{"shock":>10}']             # t0 <= t <= T_train
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
                        W_a, W_b = fr['W_a'][f], fr['W_b'][f]
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
    '''One step from the FV data at t_n (t0 <= t_n < T_train), not a rollout: z depends on the field, so
    a swap or a traversal is only defined for one step from a given W_grid.'''
    test_id = data[0]
    rng = np.random.default_rng(0)
    n0, n = train_band(test_id)
    x_cell = test_id.x_centers.to(DEVICE)
    W_in = test_id.W[:, n0:n0 + n]                                                 # [C, n, 512, 3]: inputs at t_n
    sub = np.linspace(0, n - 1, n_t).astype(int)                                   # traversal snapshots
    lines = [f'{"arm":<12}{"front err vs A (cells)":>24}{"star-state err vs B":>22}']
    for name in levelset_arms(res):
        model, encoder = nets[name]
        z_geo, z_field = encoder(W_in.flatten(0, 1).to(DEVICE).transpose(1, 2))
        z_geo, z_field = z_geo.view(test_id.C, n, -1), z_field.view(test_id.C, n, -1)
        cases = res[name]['test_id']

        # swap test: A's field and z_geo, B's z_field (same t_n); fronts at t_n+1 vs A's, star states vs B's
        f_err, s_err = [], []
        for A, B in rng.choice(test_id.C, size=(n_pairs, 2), replace=True):
            if A == B:
                continue
            out = one_step(model, encoder, W_in[A], x_cell, z=(z_geo[A], z_field[B]))
            cA, cB = cases[A], cases[B]
            for j, fr in enumerate(out['fronts']):                                 # cA['t'][j] = t_n+1
                if len(fr['x']):
                    f_err.append(np.abs(fr['x'][:, None] - cA['x_ex'][j][None]).min(0).mean() / DX)
                edges = X0 + cA['speeds'] * cA['t'][j]                              # A's regions
                for lo, hi, W_star in [(edges[1], edges[2], (cB['sol'].rho_star_L, cB['sol'].u_star, cB['sol'].P_star)),
                                       (edges[2], edges[3], cB['W_star_R'])]:
                    inside = (XC > lo + 2 * DX) & (XC < hi - 2 * DX)
                    if inside.any():
                        s_err.append((np.abs(out['W'][j, inside] - W_star).sum(-1) / np.abs(W_star).sum()).mean())
        lines.append(f'{name:<12}{np.mean(f_err):>24.2f}{np.mean(s_err):>22.4f}')

        # traversals: each z_geo coordinate over +-2 std (over cases and t) around the mid-r case, region maps at t_n+1
        mid = picks(res)[1][1]
        std = z_geo.flatten(0, 1).std(0)
        t_sub = cases[mid]['t'][sub]
        steps = (-2, -1, 0, 1, 2)
        fig, ax = plt.subplots(z_geo.shape[-1], len(steps), figsize=(3 * len(steps), 2.5 * z_geo.shape[-1]),
                               squeeze=False, sharex=True, sharey=True)
        for k in range(z_geo.shape[-1]):
            for col, s in enumerate(steps):
                zg = z_geo[mid, sub].clone()
                zg[:, k] += s * std[k]
                lab = one_step(model, encoder, W_in[mid, sub], x_cell, fronts=False,
                               z=(zg, z_field[mid, sub]))['labels']
                ax[k, col].imshow(lab, origin='lower', aspect='auto', cmap='tab10', vmin=-0.5, vmax=9.5,
                                  extent=[0, 1, t_sub[0], t_sub[-1]])
                ax[k, col].set_title(f'z_geo[{k}] {s:+d} std', fontsize=8)
        fig.suptitle(f'{name}: z_geo traversals around r = {cases[mid]["r"]:.2f} (one step from FV)')
        save(fig, f'latent_traversals_{name}.png')

    text = '\n'.join(lines)
    print(text)
    (OUT / 'latent_swap.txt').write_text(text + '\n')

# ======================================================================================
# Master
# ======================================================================================

GROUPS = {'speed': group_speed, 'topology': group_topology, 'rh': group_rh,
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