import torch
import torch.nn as nn

# ======================================================================================
# Geometry Net
# ======================================================================================

class LevelSetsNet(nn.Module):
    '''
    Predicts Level-Sets parameters and region splits
        - input
            - (x, t) input (or fourier features)

        - z[x: ] -> RegionFieldNet
    '''

    def __init__(self, N_in, z_geo, N_out,
                 hidden_layers = [128, 128, 128, 128],
                 act = nn.SiLU):
        
        super().__init__()

        dims = [N_in + z_geo] + hidden_layers + [N_out]

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))

            if i < len(dims) - 2:
                layers.append(act())

        self.net = nn.Sequential(*layers)

    def forward(self, x, z_geo):
        return self.net(torch.cat((x, z_geo),dim=-1).to(torch.float32))


# ======================================================================================
# Front frame (physics-agnostic: depends only on the level sets)
# ======================================================================================

def newton_step(phi_fn, u, d, j, create_graph: bool = False):                   # AUDIT 1 fix
    """One Newton step on psi = phi_d - phi_j from u [P, D] toward psi = 0. Returns (foot, n, |grad psi|),
    n the unit normal from d toward j at u. Detached unless create_graph."""
    with torch.enable_grad():
        u = u.detach().requires_grad_(True)
        phi = phi_fn(u)
        psi = (phi.gather(1, d[:, None]) - phi.gather(1, j[:, None])).squeeze(1)
        grad = torch.autograd.grad(psi.sum(), u, create_graph=create_graph)[0]     # [P, D]
    norm = grad.norm(dim=-1).clamp_min(1e-12)
    n = -grad / norm[:, None]
    foot = u.detach() + (psi / norm)[:, None] * n
    return (foot, n, norm) if create_graph else (foot.detach(), n.detach(), norm.detach())


def front_frame(phi_fn, x, d, j, create_graph: bool = False, steps: int = 8) -> dict:   # AUDIT 3: no delta. AUDIT fix: 8 steps
    """Frame of each point's nearest front: the front between its own zone d and zone j.

    x [P, D] (space only); d, j [P]. psi = phi_d - phi_j is >= 0 in d and 0 on the front.
    The foot is found by `steps` Newton steps on psi from x (AUDIT 1 fix: one step is crude away from
    the front with raw phi); only the last carries the graph. 3 steps left the foot ~3 cells off from band points ~11 cells out (bad s there); 8 removes it.
    Returns per point:
        dist    distance to the front along n, (x_foot - x).n                    [P]
        n       unit normal from d toward j at the foot                          [P, D]
        norm    psi(x) / dist, so a change of psi at x converts to a change of dist (|grad psi| on the front)  [P]
        x_foot  front point                                                      [P, D]
    The only place distances come from: the raw scores phi are not distances.
    Written for any number of space dimensions D. create_graph=True when the geometry trains;
    otherwise the outputs are detached.
    """
    x = x.detach()
    foot = x
    for _ in range(steps - 1):
        foot = newton_step(phi_fn, foot, d, j)[0]
    foot, n, slope = newton_step(phi_fn, foot, d, j, create_graph)
    dist = ((foot - x) * n).sum(-1)
    with torch.set_grad_enabled(create_graph):
        phi = phi_fn(x)
    psi = (phi.gather(1, d[:, None]) - phi.gather(1, j[:, None])).squeeze(1)
    # detached: only converts a shift to a psi change; its gradient has 1/dist^2 and blows up near the front
    norm = torch.where(dist.abs() > 1e-9, psi / torch.where(dist.abs() > 1e-9, dist, 1.0), slope).detach()
    frame = {'dist': dist, 'n': n, 'norm': norm, 'x_foot': foot}
    if not create_graph:
        frame = {k: v.detach() for k, v in frame.items()}
    return frame


def front_distances(phi_fn, x, z, d, offset: float, steps: int = 8) -> torch.Tensor:   # AUDIT 1 fix
    """Distance from each point to the front between its own zone d and every other zone k, from
    front_frame. A front counts only if it is there: zone d wins at x_foot - offset*n and zone k at
    x_foot + offset*n (the estimate is unreliable for zones that win nowhere near).
    phi_fn(u, z) -> phi; x [P, D], z [P, .], d [P] -> [P, K], inf at k = d and where no front. Detached.
    Costs K front_frame calls on all points; any D."""
    P, K = len(d), phi_fn(x[:1], z[:1]).shape[-1]
    dist = x.new_full((P, K), float('inf'))
    for k in range(K):
        fr = front_frame(lambda u: phi_fn(u, z), x, d, torch.full_like(d, k), steps=steps)
        v = torch.cat([fr['x_foot'] - offset * fr['n'], fr['x_foot'] + offset * fr['n']]).clamp(-1, 1)
        with torch.no_grad():
            lab = phi_fn(v, z.detach().repeat(2, 1)).argmax(-1)
        there = (lab[:P] == d) & (lab[P:] == k) & (d != k)
        dist[:, k] = torch.where(there, fr['dist'], dist[:, k])
    return dist


def advect(frame, d, j, s, dt_x: float, reach: float):
    """Move each band point's front one time step: by s * dt_x along n (x~), s [P] in field units along n.

    The point stays put; its distance to the front grows by the shift. A point the front passed over
    (new dist < 0) changes owner: d and j swap, and n, dist and s flip sign, so the frame is again seen
    from the owner. Returns (frame, d, j, s, flipped); frame['shift'] is the shift along the time-n normal.
    """
    s = s.clamp(-reach / dt_x, reach / dt_x)     # ⚑ front moves at most width + taper per step (bounds s in rollouts)
    shift = s * dt_x
    dist = frame['dist'] + shift
    flipped = dist < 0
    sign = 1 - 2 * flipped.to(dist)               # -1 where the owner changed
    frame = {'dist': sign * dist, 'n': sign[:, None] * frame['n'], 'norm': frame['norm'],
             'x_foot': frame['x_foot'] + shift[:, None] * frame['n'], 'shift': shift}
    return frame, torch.where(flipped, j, d), torch.where(flipped, d, j), sign * s, flipped


def advect_scores(phi, idx, d, j, dpsi):
    """phi at t_n+1: at band points idx (time-n zones d, j), psi = phi_d - phi_j grows by dpsi,
    split evenly between the two scores. Use dpsi = shift * norm, so that the new owner matches advect's
    `flipped` exactly. Out of place; other points and zones are unchanged."""
    return phi + torch.zeros_like(phi).index_put((idx, d), dpsi / 2).index_put((idx, j), -dpsi / 2)
