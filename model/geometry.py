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
# Zone scores and front frame (physics-agnostic: depend only on the level sets)
# ======================================================================================

def phi_and_slope(phi_fn, x, create_graph: bool):

    with torch.enable_grad():
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)         # a leaf of our own; caller's tensor untouched
        phi = phi_fn(x)
        # spatial gradient of each zone's phi: [N, K, D] (all columns but the last, t)
        dphi_dx = torch.stack(
            [torch.autograd.grad(phi[:, k].sum(), x, create_graph=create_graph,
                                 retain_graph=True)[0][:, :-1]
             for k in range(phi.shape[1])], dim=1)
    return phi, dphi_dx


def zone_scores(phi_fn, x, delta: float, create_graph: bool) -> torch.Tensor:
    """phi_hat [N, K] = phi / sqrt(|grad_x phi|^2 + delta^2): one distance-like score per zone.
    The owner of a point is phi_hat.argmax(-1). phi_fn(u) -> phi [N, K]."""
    phi, dphi_dx = phi_and_slope(phi_fn, x, create_graph)
    return phi / torch.sqrt((dphi_dx ** 2).sum(-1) + delta ** 2)


def front_frame(phi_fn, x, d, j, delta: float, create_graph: bool = False) -> dict:
    """Frame of each point's nearest front: the front between its own zone d and zone j.

    x [P, D+1] = (space..., t~); d, j [P]. psi = phi_hat_d - phi_hat_j is >= 0 in d and 0 on the front.
    Returns per point:
        dist    distance to the front, psi / |grad_x psi|                          [P]
        n       unit normal from d toward j, -grad_x psi / |grad_x psi|            [P, D]
        s       normal speed of the front along n, psi_t / |grad_x psi| (x~ per t~) [P]
        x_foot  nearest front point, one Newton step on psi (exact if psi is linear)   [P, D+1]
    Written for any number of space dimensions D. create_graph=True when the geometry trains;
    otherwise the outputs are detached.
    """
    with torch.enable_grad():
        u = x.detach().requires_grad_(True)
        ph = zone_scores(phi_fn, u, delta, create_graph=True)                     # graph kept for grad psi
        psi = (ph.gather(1, d[:, None]) - ph.gather(1, j[:, None])).squeeze(1)
        grad = torch.autograd.grad(psi.sum(), u, create_graph=create_graph)[0]   # [P, D+1]

    g_x, psi_t = grad[:, :-1], grad[:, -1]
    norm = g_x.norm(dim=-1).clamp_min(1e-12)
    dist = psi / norm
    n = -g_x / norm[:, None]
    frame = {'dist': dist, 'n': n, 's': psi_t / norm,
             'x_foot': torch.cat([x[:, :-1].detach() + dist[:, None] * n, x[:, -1:].detach()], dim=-1)}
    if not create_graph:
        frame = {k: v.detach() for k, v in frame.items()}
    return frame
