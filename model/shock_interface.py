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

def interface_speed(W_d, W_j, n, rel_tol: float = 0.01) -> torch.Tensor:
    """s [M]: normal speed of the front along n (from d to j), field units, from the states on its two sides.
    Mass conservation across the front: s = [rho v.n] / [rho]. Shock: the RH shock speed;
    contact or material interface: v.n. Where |[rho]| < rel_tol * mean rho, s = mean v.n instead."""
    rho_d, rho_j = W_d[:, 0], W_j[:, 0]
    vn_d, vn_j = (W_d[:, 1:-1] * n).sum(-1), (W_j[:, 1:-1] * n).sum(-1)
    d_rho = rho_j - rho_d
    small = d_rho.abs() < rel_tol * 0.5 * (rho_d + rho_j)
    s = (rho_j * vn_j - rho_d * vn_d) / torch.where(small, torch.ones_like(d_rho), d_rho)   # safe divide
    return torch.where(small, 0.5 * (vn_d + vn_j), s)

# ======================================================================================
# Shock constraint
# ======================================================================================

def taper_weight(dist: torch.Tensor, width: float, taper: float) -> torch.Tensor:
    """1 for dist <= width, cosine down to 0 over the next `taper`, 0 beyond."""
    r = ((dist - width) / taper).clamp(0, 1)
    return 0.5 * (1 + torch.cos(torch.pi * r))

def constrain_shocks(W_regions, idx, d, j, frame, s, test_states, fields_at, cfg, apply: bool = True):
    """Pinned RH constraint on the zone behind each shock and the contact condition on both sides of each contact,
    decided per band point at t_n+1.

    Band point i = idx[q] has own zone d[q] and neighbour zone j[q] across its nearest front, with the
    advected frame frame[...][q] and normal front speed s[q] (field units, along n).
      1. Read zone d's field at the pin point x_foot - side_offset*n (own side) and zone j's at
         x_foot + side_offset*n (other side): both plateaus at t_n+1, not the smeared ramp.
      2. Shock test (no gradient), on the input states at t_n, test_states = (W_in_d, W_in_j):
         p_d > (1 + tau_p) p_j, compression (v_d - v_j).n_ud < 0, Mn > Mn_gate, and the downstream state
         subsonic in the shock frame (Lax: Mn_d < 1). RH also needs the
         predicted upstream state to be supersonic in the shock frame (Mn_pred > 1).
      3. At shock points: W_RH = RH(W_up, n_ud, s_p), s_p the shock speed whose RH pressure is zone d's at the pin
         (so RH sets rho and v behind from FieldNet's p, not from s), and zone d's field at the point is constrained
         by the offset that makes it W_RH at the pin point, weighted by taper_weight(dist):
             rho *= (rho_RH / rho_pin)^w,   v += (v_RH - v_pin) w,   p *= (p_RH / p_pin)^w

      4. At contact points (no shock, |[p]| <= tau_p max p on the input states): zone d's field is shifted so its
         p and v.n at the pin equal the shared values (geometric mean p, mean v.n of both sides), same taper.
    fields_at(u, rows) -> [len(u), K, F] zone fields at points u with the latents of rows.
    cfg: gamma, tau_p, Mn_gate, side_offset, width, taper.
    apply=False (free-jump arm) runs steps 1-3 for the hooks but returns W_regions unchanged.
    Returns (W_regions, jump); jump holds per band point values (W_RH NaN and w 0 where not a shock).
    """
    P = len(idx)
    q = torch.arange(P, device=idx.device)
    n_ud, s_ud = -frame['n'], -s                           # upstream (j) -> downstream (d)

    # 1. plateau reads on both sides of the front, at t_n+1
    step = cfg['side_offset'] * frame['n']
    u = torch.cat([frame['x_foot'] - step, frame['x_foot'] + step]).clamp(-1, 1)   # [2P, D]
    W_side = fields_at(u, idx.repeat(2))                                       # [2P, K, F]
    W_pin, W_up = W_side[:P][q, d], W_side[P:][q, j]                           # [P, F] each

    # 2. shock test, on the input states at t_n
    W_in_d, W_in_j = test_states
    with torch.no_grad():
        Mn = normal_mach(W_in_j, n_ud, s_ud, cfg['gamma'])
        shock = ((W_in_d[:, -1] > (1 + cfg['tau_p']) * W_in_j[:, -1])
                 & (((W_in_d[:, 1:-1] - W_in_j[:, 1:-1]) * n_ud).sum(-1) < 0)
                 & (Mn > cfg['Mn_gate'])
                 & (normal_mach(W_in_d, n_ud, s_ud, cfg['gamma']) < 1)        # AUDIT fix: Lax, subsonic behind; drops a
                 & (normal_mach(W_up, n_ud, s_ud, cfg['gamma']) > 1))          # wild s from a drifted rollout field. RH defined (p_RH > 0)
        # AUDIT fix: contact test, the same input states: no shock and pressure continuous to within tau_p
        contact = ~shock & ((W_in_d[:, -1] - W_in_j[:, -1]).abs() <= cfg['tau_p'] * torch.maximum(W_in_d[:, -1], W_in_j[:, -1]))

    # 3. RH state and the constraint, at shock points only
    sh = torch.nonzero(shock).squeeze(1)
    # AUDIT fix: the shock strength comes from zone d's own pressure at the pin (FieldNet, trained on the data), not
    # from s. s is read off the last step's output, i.e. off the RH band, and RH(s) gives back a state whose speed
    # law is s again: a one-off bad s (foot a few cells off) was written into the band and never washed out
    g, W_u = cfg['gamma'], W_up[sh]
    Mn_p = torch.sqrt((1 + (g + 1) / (2 * g) * (W_pin[sh, -1] / W_u[:, -1] - 1)).clamp(min=1))   # Mn from p_d / p_up
    s_p = (W_u[:, 1:-1] * n_ud[sh]).sum(-1) - Mn_p * torch.sqrt(g * W_u[:, -1] / W_u[:, 0])
    W_RH = rankine_hugoniot(W_u, n_ud[sh], s_p, g)                                 # [S, F], p_RH = p_pin
    w = taper_weight(frame['dist'][sh], cfg['width'], cfg['taper'])             # [S]

    # 4. AUDIT fix: contact condition. No mass crosses a contact, so p and v.n are continuous ([p] = [v.n] = 0; rho and
    # tangential v are free). Each side's field is shifted to the shared values at its pin (mean v.n, geometric mean
    # p of the two zone fields at their pins), with the same taper. Both sides are band points, so both move; the
    # shared values come from the fields at t_n+1, not from the last step's output, so there is no loop
    ct = torch.nonzero(contact).squeeze(1)
    n_c, W_c, W_o = frame['n'][ct], W_pin[ct], W_up[ct]
    w_c = taper_weight(frame['dist'][ct], cfg['width'], cfg['taper'])

    corr = W_regions.new_zeros(W_regions.shape)                                 # (log rho, v, log p) per point, zone d
    if apply and len(sh) > 0:
        W_p = W_pin[sh]
        delta = torch.cat([torch.log(W_RH[:, :1] / W_p[:, :1]),
                           W_RH[:, 1:-1] - W_p[:, 1:-1],
                           torch.log(W_RH[:, -1:] / W_p[:, -1:])], dim=-1)
        corr = corr.index_put((idx[sh], d[sh]), delta * w[:, None])
    if apply and len(ct) > 0:
        dvn = 0.5 * ((W_o[:, 1:-1] - W_c[:, 1:-1]) * n_c).sum(-1)
        delta = torch.cat([torch.zeros_like(W_c[:, :1]),
                           dvn[:, None] * n_c,
                           0.5 * torch.log(W_o[:, -1:] / W_c[:, -1:])], dim=-1)
        corr = corr.index_put((idx[ct], d[ct]), delta * w_c[:, None])
    if apply and len(sh) + len(ct) > 0:
        W_regions = torch.cat([W_regions[..., :1] * torch.exp(corr[..., :1]),
                               W_regions[..., 1:-1] + corr[..., 1:-1],
                               W_regions[..., -1:] * torch.exp(corr[..., -1:])], dim=-1)

    jump = {'idx': idx, 'zone': d, 'neighbour': j,
            'dist': frame['dist'], 'n': frame['n'], 's': s, 'x_foot': frame['x_foot'],
            'shock': shock, 'contact': contact, 'Mn': Mn, 'W_up': W_up, 'W_pin': W_pin,
            'W_RH': W_up.new_full(W_up.shape, float('nan')).index_put((sh,), W_RH),
            'w': w.new_zeros(P).index_put((sh,), w)}
    return W_regions, jump