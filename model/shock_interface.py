import torch


# ======================================================================================
# RH map (shock physics only; any number of space dimensions)
#   W = (rho, v_1..v_D, p): rho = W[:, 0], v = W[:, 1:-1], p = W[:, -1]
# ======================================================================================

def normal_mach(W_up: torch.Tensor, n: torch.Tensor, s: torch.Tensor, gamma: float = 1.4) -> torch.Tensor:
    """Mn [M] = (v_up . n - s) / c_up, in the shock frame. n [M, D] points upstream -> downstream,
    s [M] is the normal speed along n, in field units."""
    rho, v, p = W_up[:, 0], W_up[:, 1:-1], W_up[:, -1]
    return ((v * n).sum(-1) - s) / torch.sqrt(gamma * p / rho)


def rankine_hugoniot(W_up: torch.Tensor, n: torch.Tensor, s: torch.Tensor, gamma: float = 1.4) -> torch.Tensor:
    """(W_up [M, 2+D], n [M, D], s [M]) -> W_RH [M, 2+D], the state just behind the shock.
    Only call it where Mn > 1 (the shock test guarantees it). Only the normal velocity changes."""
    rho_m, v_m, p_m = W_up[:, 0], W_up[:, 1:-1], W_up[:, -1]
    g = gamma
    M2 = normal_mach(W_up, n, s, gamma) ** 2
    vn_m = (v_m * n).sum(-1)

    rho_p = rho_m * (g + 1) * M2 / ((g - 1) * M2 + 2)
    p_p = p_m * (1 + 2 * g / (g + 1) * (M2 - 1))
    # Mass flux rho (v.n - s) is continuous
    vn_p = s + (vn_m - s) * rho_m / rho_p
    v_p = v_m + (vn_p - vn_m)[:, None] * n                 # tangential part kept (2D)
    return torch.cat([rho_p[:, None], v_p, p_p[:, None]], dim=-1)


# ======================================================================================
# Shock constraint
# ======================================================================================

def taper_weight(dist: torch.Tensor, width: float, taper: float) -> torch.Tensor:
    """1 for dist <= width, cosine down to 0 over the next `taper`, 0 beyond."""
    r = ((dist - width) / taper).clamp(0, 1)
    return 0.5 * (1 + torch.cos(torch.pi * r))


def constrain_shocks(W_regions, x, idx, d, j, frame, fields_at, labels_at, cfg, apply: bool = True):
    """Pinned RH constraint on the zone behind each shock, decided per point.

    For each band point i = idx[q] (own zone d[i], neighbour zone j[i] across its nearest front,
    whose frame is frame[...][q]):
      1. Read zone d's field at the pin point x_foot - side_offset*n (own side) and zone j's at
         x_foot + side_offset*n (other side): both plateaus, not the smeared ramp.
      2. Shock test (no gradient). The point is behind a shock if its own side is downstream:
         p_pin > (1 + tau_p) p_up, compression (v_pin - v_up).n_ud < 0, Mn > Mn_gate, t >= t_min,
         and the two read points are labelled d and j. Upstream points fail the first test.
      3. At shock points: W_RH = RH(W_up, n_ud, s_ud), and zone d's field at the point is constrained
         by the offset that makes it W_RH at the pin point, weighted by taper_weight(dist):
             rho *= (rho_RH / rho_pin)^w,   v += (v_RH - v_pin) w,   p *= (p_RH / p_pin)^w
         So the composed field equals W_RH exactly at the pin (width >= side_offset), and the
         zone's own field is used, and trained, everywhere else.

    fields_at(u, rows) -> [len(u), K, F] zone fields at points u with the latents of rows;
    labels_at(u, rows) -> [len(u)] owners. cfg: gamma, k, tau_p, Mn_gate, side_offset, t_min, width, taper.
    apply=False (free-jump arm) runs steps 1-3 for the hooks but returns W_regions unchanged.
    Returns (W_regions, jump); jump holds per band point values (W_RH NaN and w 0 where not a shock).
    """
    P = len(idx)
    q = torch.arange(P, device=idx.device)
    dq, jq = d[idx], j[idx]
    n_ud = -frame['n']                                     # upstream (j) -> downstream (d)
    s_ud = -frame['s'] * cfg['k']                          # normal speed along n_ud, field units

    # 1. plateau reads on both sides of the front
    step = cfg['side_offset'] * torch.cat([frame['n'], frame['n'].new_zeros(P, 1)], dim=-1)
    u = torch.cat([frame['x_foot'] - step, frame['x_foot'] + step])           # [2P, D+1]
    u = torch.cat([u[:, :-1].clamp(-1, 1), u[:, -1:]], dim=-1)
    W_side = fields_at(u, idx.repeat(2))                                       # [2P, K, F]
    W_pin, W_up = W_side[:P][q, dq], W_side[P:][q, jq]                         # [P, F] each
    lab = labels_at(u, idx.repeat(2))

    # 2. shock test
    with torch.no_grad():
        Mn = normal_mach(W_up, n_ud, s_ud, cfg['gamma'])
        shock = ((W_pin[:, -1] > (1 + cfg['tau_p']) * W_up[:, -1])
                 & (((W_pin[:, 1:-1] - W_up[:, 1:-1]) * n_ud).sum(-1) < 0)
                 & (Mn > cfg['Mn_gate'])
                 & (x[idx, -1] >= cfg['t_min'])
                 & (lab[:P] == dq) & (lab[P:] == jq))

    # 3. RH state and the constraint, at shock points only
    sh = torch.nonzero(shock).squeeze(1)
    W_RH = rankine_hugoniot(W_up[sh], n_ud[sh], s_ud[sh], cfg['gamma'])        # [S, F]
    w = taper_weight(frame['dist'][sh], cfg['width'], cfg['taper'])             # [S]

    if apply and len(sh) > 0:
        W_p = W_pin[sh]
        delta = torch.cat([torch.log(W_RH[:, :1] / W_p[:, :1]),
                           W_RH[:, 1:-1] - W_p[:, 1:-1],
                           torch.log(W_RH[:, -1:] / W_p[:, -1:])], dim=-1)      # (log rho, v, log p)
        corr = W_regions.new_zeros(W_regions.shape).index_put((idx[sh], dq[sh]), delta * w[:, None])
        W_regions = torch.cat([W_regions[..., :1] * torch.exp(corr[..., :1]),
                               W_regions[..., 1:-1] + corr[..., 1:-1],
                               W_regions[..., -1:] * torch.exp(corr[..., -1:])], dim=-1)

    jump = {'idx': idx, 'zone': dq, 'neighbour': jq,
            'dist': frame['dist'], 'n': frame['n'], 's': frame['s'] * cfg['k'], 'x_foot': frame['x_foot'],
            'shock': shock, 'Mn': Mn, 'W_up': W_up, 'W_pin': W_pin,
            'W_RH': W_up.new_full(W_up.shape, float('nan')).index_put((sh,), W_RH),
            'w': w.new_zeros(P).index_put((sh,), w)}
    return W_regions, jump
