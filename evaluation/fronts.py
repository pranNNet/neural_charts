import torch

from model.geometry import zone_scores


# ======================================================================================
# Front lists (evaluation only; the model works per point and does not use these)
#   Moved from model/helper.py. In 2D this becomes contouring of the labels.
# ======================================================================================

def search_points(n_search: int) -> torch.Tensor:
    """n_search evenly spaced x~ on [-1, 1] (spacing 2 / n_search), where the continuous phi_hat is
    sampled to bracket its fronts. Not a mesh and not tied to the data: only a search resolution."""
    return -1 + (2 * torch.arange(n_search, dtype=torch.float32) + 1) / n_search


def find_fronts(phi_fn, z_geo, t, x_search, k: float, delta: float,
                create_graph: bool = False, search_chunk: int = None) -> dict:
    """The fronts of G (case, time) groups at once. z_geo [G, d], t [G] (t~), x_search [M] (x~).

    A front is bracketed by any pair of neighbouring search points whose hard label flips (no table, D77).
    Returns 1-D tensors of length F (all groups' flips, may be 0), by group, then left to right:
        g     group index in 0..G-1 (row of z_geo and t)
        x     front position x~ (linear zero of psi between the two search points, differentiable)
        dxdt  lab-frame speed dx~/dt~ = -psi_t / psi_x
        n     sign(psi_x): +1 or -1, pointing from zone a to zone b
        s     normal speed n * dxdt * k, in field units (D76)
        a, b  zone left and right of the flip
    psi = phi_hat_b - phi_hat_a. phi_fn(u, z) -> phi. create_graph=True when the geometry trains.
    search_chunk: evaluate the label grid in chunks of this many points. None = one call.
    """
    G, M = t.shape[0], x_search.shape[0]
    x_search = x_search.to(t)
    dx = x_search[1] - x_search[0]
    grad_on = torch.is_grad_enabled()

    def phi_hat_at(u, z, graph):
        return zone_scores(lambda v: phi_fn(v, z), u, delta, graph)

    with torch.enable_grad():
        # 1. Flips: hard labels on the G stacked search grids, no gradient.
        grid = torch.stack([x_search.repeat(G), t.repeat_interleave(M)], dim=-1)   # [G*M, 2]
        z_grid = z_geo.detach().repeat_interleave(M, dim=0)                         # [G*M, d]
        step = len(grid) if search_chunk is None else search_chunk
        lab = torch.cat([phi_hat_at(grid[c:c + step], z_grid[c:c + step], False).argmax(dim=-1)
                         for c in range(0, len(grid), step)]).view(G, M)
        g, i = torch.nonzero(lab[:, 1:] != lab[:, :-1], as_tuple=True)   # flip between i and i+1 of group g
        a, b = lab[g, i], lab[g, i + 1]
        F = len(g)
        t_f, z_f = t[g], z_geo[g]                                          # [F], [F, d]: one row per front

        def psi(u, rep: int = 1):
            """psi at points u [rep*F, 2]: rep stacked copies of the F fronts."""
            ph = phi_hat_at(u, z_f.repeat(rep, 1), True)
            return (ph.gather(1, b.repeat(rep)[:, None]) - ph.gather(1, a.repeat(rep)[:, None])).squeeze(1)

        # 2. Position: zero of psi, linear between x_i (psi <= 0) and x_i+1 (psi >= 0).
        ends = torch.stack([torch.cat([x_search[i], x_search[i + 1]]), t_f.repeat(2)], dim=-1)   # [2F, 2]
        psi_ij = psi(ends, rep=2)
        psi_i, psi_j = psi_ij[:F], psi_ij[F:]
        x = x_search[i] + dx * psi_i / (psi_i - psi_j)

        # 3. Speed: one autograd call for (psi_x, psi_t) at (x, t~), all fronts.
        u = torch.stack([x, t_f], dim=-1)
        if not u.requires_grad:                       # frozen geometry: make it differentiable
            u.requires_grad_(True)
        psi_x, psi_t = torch.autograd.grad(psi(u).sum(), u, create_graph=create_graph)[0].unbind(-1)
        n = torch.sign(psi_x)
        dxdt = -psi_t / psi_x
        s = n * dxdt * k

    fronts = {"g": g, "x": x, "dxdt": dxdt, "n": n, "s": s, "a": a, "b": b}
    if not (grad_on and create_graph):                # evaluation, or frozen geometry: drop the graph
        fronts = {key: v.detach() for key, v in fronts.items()}
    return fronts


def split_fronts(fronts: dict, G: int) -> list:
    """Flat find_fronts dict -> list of G dicts, one per group, without 'g'. Views, no copies."""
    counts = torch.bincount(fronts["g"], minlength=G).tolist()
    parts = {key: v.split(counts) for key, v in fronts.items() if key != "g"}
    return [{key: parts[key][j] for key in parts} for j in range(G)]
