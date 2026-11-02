'''
Distance target for the LevelSetsNet (replaces CE + boundary loss in every stage).

    phi*_k(x) = kappa * clamp(signed distance from x to zone k's region, -clip, +clip)      (cells)

positive inside zone k, negative outside, zones not present at -kappa*clip. argmax phi* is the zone, and
psi* = phi*_a - phi*_b is a straight line through each front out to +-clip cells, with its zero at the
sub-cell front position. Validated in compare_1d (target_builder.py, stage1_trial.py): labels 100%, measured
fronts -0.015 +- 0.06 (contact) and +0.007 +- 0.063 (shock) cells from exact.

Front positions come from the FV solution, not the labels: at each flagged face, zones a | b are the nearest
labelled cells either side, and the front is where rho crosses half-way between its values at those two
cells, linear between cell centres.

    fronts_row(W, cls, flags)             numpy, one snapshot -> [(x_front, a, b)] in data x [0, 1]
    fronts_array(W, cls, flags, F_MAX)    numpy, [..., 512, 3] -> [..., F_MAX, 3], nan padded (for the datasets)
    target_scores(u, fronts, K)           torch, batched: phi* [B, P, K] at u [B, P] (data x)

Data x and cells: one cell = 1/512. The model's x~ in [-1, 1] maps to u = (x~ + 1) / 2.
'''

import numpy as np
import torch

N = 512
DX = 1.0 / N
XC = (np.arange(N) + 0.5) * DX
KAPPA, CLIP = 4.0, 3.0          # score per cell, cells; the values the compare_1d checks validated

def fronts_row(W, cls, flags, merge=3, var=0):
    '''Fronts of one snapshot. W [512, 3] primitives (raw W_fv), cls [512] (-1 = ignored), flags [511] (1 = jump
    at the face between cells i and i+1). var: the primitive that locates the front (0 = rho, which jumps at
    contacts and shocks). Returns [(x_front, a, b)] sorted by x (data x), a = zone left, b = zone right.
    Flagged faces closer than `merge` cells with the same (a, b) count once.'''
    lab = np.flatnonzero(cls >= 0)
    out = []
    for i in np.flatnonzero(flags == 1):
        left, right = lab[lab <= i], lab[lab > i]
        if not len(left) or not len(right):
            continue
        iL, iR = left[-1], right[0]
        a, b = int(cls[iL]), int(cls[iR])
        if a == b:
            continue
        mid = 0.5 * (W[iL, var] + W[iR, var])
        f = W[iL:iR + 1, var] - mid
        k = np.flatnonzero(np.sign(f[:-1]) != np.sign(f[1:]))
        face = XC[i] + DX / 2
        if len(k):
            xs = XC[iL + k] + DX * f[k] / (f[k] - f[k + 1])        # linear between centres
            x = xs[np.argmin(np.abs(xs - face))]
        else:
            x = face
        if out and out[-1][1:] == (a, b) and abs(out[-1][0] - x) < merge * DX:
            continue
        out.append((float(x), a, b))
    return sorted(out)

def fronts_array(W, cls, flags, F_MAX=4):
    '''fronts_row over leading dims: W [..., 512, 3], cls [..., 512], flags [..., 511] (numpy)
    -> [..., F_MAX, 3] float32, rows (x, a, b) sorted by x, nan padded.'''
    lead = W.shape[:-2]
    W, cls, flags = W.reshape(-1, *W.shape[-2:]), cls.reshape(-1, cls.shape[-1]), flags.reshape(-1, flags.shape[-1])
    out = np.full((len(W), F_MAX, 3), np.nan, np.float32)
    for m in range(len(W)):
        for i, f in enumerate(fronts_row(W[m], cls[m], flags[m])[:F_MAX]):
            out[m, i] = f
    return out.reshape(*lead, F_MAX, 3)

def target_scores(u, fronts, K, kappa=KAPPA, clip=CLIP, dx=DX):
    '''phi* [B, P, K] at u [B, P] (data x) from fronts [B, F, 3] (x, a, b), sorted by x, nan padded.
    A snapshot needs at least one front (every training snapshot, t >= t0, has two).'''
    B, P = u.shape
    xf = fronts[..., 0].nan_to_num(float('inf'))                          # pads at +inf: empty intervals
    a = fronts[..., 1].nan_to_num(0).long()
    b = fronts[..., 2].nan_to_num(0).long()
    inf = torch.full_like(xf[:, :1], float('inf'))
    lo, hi = torch.cat([-inf, xf], 1), torch.cat([xf, inf], 1)           # [B, F+1] interval bounds
    zone = torch.cat([a[:, :1], b], 1)                                    # zone of each interval
    inside = torch.minimum(u[..., None] - lo[:, None], hi[:, None] - u[..., None])   # [B, P, F+1], > 0 inside
    d = (inside / dx).clamp(-clip, clip)
    phi = u.new_full((B, P, K), -clip)
    phi.scatter_reduce_(2, zone[:, None].expand(-1, P, -1), d, reduce='amax')
    return kappa * phi