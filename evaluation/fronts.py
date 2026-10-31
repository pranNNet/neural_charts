import torch


# ======================================================================================
# Front lists (evaluation only; the model works per band point and does not use these)
#   Built from the jump hooks of one step, i.e. the advected frame at t_n+1. There is no
#   time column any more: each call is one step of the step map. In 2D this becomes contouring.
# ======================================================================================

def front_lists(jump: dict, group: torch.Tensor, G: int, gap: float, rh: bool) -> list:
    """The fronts of G groups (batch rows) at t_n+1, from out['jump'] of one forward.

    Each band point sees its nearest front from its own zone d, with n pointing from d toward the
    neighbour zone j, so the points on both sides of a front describe the same front. Points are sorted
    by group, then x_foot; a new front starts where the group or the zone pair changes, or where x_foot
    jumps by more than gap (x~).
    group [N]: each query point's row. rh: the arm applies the constraints, so the composed state on the
    constrained side is W_RH behind a shock, and the shared p (geometric mean) and v.n (mean) at a contact
    (w = 1 at the pin point, which lies inside width).
    1D only from here on (n and the speeds are read along x).

    Returns a list of G dicts of 1-D tensors, F fronts left to right (F may be 0), medians over the cluster:
        x         front position x~ (x_foot)
        S         lab-frame advection speed s * n (the speed law on the input field), field units
        S_RH      lab-frame speed implied by the RH state (mass flux across W_up | W_RH): the shock speed RH
                  actually uses, set by zone d's pressure at the pin; nan where not a shock
        a, b      zone left and right of the front
        is_shock  shock test passed by most points on one side (only the downstream side can pass it)
        up_is_a   upstream side is a, by the shock points (False where not a shock)
        Mn        Mn of the shock test (input states at t_n, s at t_n+1); nan where not a shock
        W_a, W_b  plateau states left and right at t_n+1, side_offset off the front, as composed, each read
                  by the band points on that side (their own zone at the pin)
        W_RH      RH state; nan where not a shock
        n_pts     band points in the cluster
    """
    idx, d, j = jump['idx'], jump['zone'], jump['neighbour']
    g, x = group[idx], jump['x_foot'][:, 0]
    right = jump['n'][:, 0] > 0                                     # j lies to the right of the point
    shock, contact = jump['shock'], jump.get('contact', torch.zeros_like(jump['shock']))
    a, b = torch.where(right, d, j), torch.where(right, j, d)
    W_pin, W_up, n = jump['W_pin'], jump['W_up'], jump['n']

    def shared(W, W_o):
        '''W with the contact condition: v.n moved to the mean of both sides, p to their geometric mean.'''
        dvn = 0.5 * ((W_o[:, 1:-1] - W[:, 1:-1]) * n).sum(-1, keepdim=True)
        return torch.cat([W[:, :1], W[:, 1:-1] + dvn * n, torch.sqrt(W[:, -1:] * W_o[:, -1:])], -1)

    ct = (contact & rh)[:, None]                         # both sides of a contact are constrained
    own = torch.where((shock & rh)[:, None], jump['W_RH'], torch.where(ct, shared(W_pin, W_up), W_pin))
    other = torch.where(ct, shared(W_up, W_pin), W_up)   # behind a shock, the other side is the free upstream
    S = jump['s'] * n[:, 0]
    m_RH, m_up = jump['W_RH'][:, 0] * jump['W_RH'][:, 1], W_up[:, 0] * W_up[:, 1]
    S_RH = (m_RH - m_up) / (jump['W_RH'][:, 0] - W_up[:, 0])                      # nan where W_RH is

    order = torch.argsort(4 * g.double() + x.double())               # by group, then x (x~ in [-1, 1])
    g, x, a, b, right, shock = g[order], x[order], a[order], b[order], right[order], shock[order]
    own, other, S, Mn, W_RH = own[order], other[order], S[order], jump['Mn'][order], jump['W_RH'][order]
    S_RH = S_RH[order]

    new = torch.ones_like(g, dtype=torch.bool)
    new[1:] = (g[1:] != g[:-1]) | (a[1:] != a[:-1]) | (b[1:] != b[:-1]) | (x[1:] - x[:-1] > gap)
    starts = torch.nonzero(new).squeeze(1).tolist() + [len(g)]

    nan = x.new_tensor(float('nan'))
    fronts = [[] for _ in range(G)]
    for lo, hi in zip(starts[:-1], starts[1:]):
        c = slice(lo, hi)
        sh, side = shock[c], right[c]
        frac = [sh[side == k].float().mean() if (side == k).any() else 0.0 for k in (True, False)]
        is_shock = bool(max(frac) > 0.5)
        # each side's state as its own band points read it (other side's reads only if a side has no points)
        W_a = own[c][side] if side.any() else other[c][~side]
        W_b = own[c][~side] if (~side).any() else other[c][side]
        fronts[int(g[lo])].append({
            'x': x[c].median(), 'S': S[c].median(), 'a': a[lo], 'b': b[lo],
            'S_RH': S_RH[c][sh].median() if is_shock else nan,
            'is_shock': torch.tensor(is_shock, device=x.device),
            'up_is_a': torch.tensor(is_shock and bool((~side[sh]).float().mean() > 0.5), device=x.device),
            'Mn': Mn[c][sh].median() if is_shock else nan,
            'W_a': W_a.median(0).values, 'W_b': W_b.median(0).values,
            'W_RH': W_RH[c][sh].median(0).values if is_shock else W_RH.new_full((W_RH.shape[1],), float('nan')),
            'n_pts': torch.tensor(hi - lo, device=x.device)})

    keys = ['x', 'S', 'S_RH', 'a', 'b', 'is_shock', 'up_is_a', 'Mn', 'W_a', 'W_b', 'W_RH', 'n_pts']
    empty = {'x': x.new_zeros(0), 'S': x.new_zeros(0), 'S_RH': x.new_zeros(0), 'a': d.new_zeros(0), 'b': d.new_zeros(0),
             'is_shock': shock.new_zeros(0), 'up_is_a': shock.new_zeros(0), 'Mn': x.new_zeros(0),
             'W_a': own.new_zeros(0, own.shape[1]), 'W_b': own.new_zeros(0, own.shape[1]),
             'W_RH': own.new_zeros(0, own.shape[1]), 'n_pts': d.new_zeros(0)}
    return [{k: torch.stack([f[k] for f in fr]) for k in keys} if fr else empty for fr in fronts]
