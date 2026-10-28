import torch


# ======================================================================================
# Multi-Scale Fourier Transform
# ======================================================================================


def MSFourier_Features(x: torch.Tensor, B:torch.Tensor) -> torch.Tensor:
    """[N, D+1] normalized (space..., t~) -> [N, D+1 + 2*n_freq]: [xt, sin(2pi xt@B), cos(2pi xt@B)].
 
    Pure function of xt. No detach: autograd still flows to xt (needed for normals).
    """
    proj = 2 * torch.pi * (x @ B.to(x))
    return torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=1)

def make_B(bands, seed: int = 0) -> torch.Tensor:
    """Draw the fixed frequency matrix B [D+1, sum(n_freq)].
 
    bands: list of (n_freq, (sigma_x, [sigma_y,] sigma_t)), one entry per scale.
    Local generator, bands drawn in order: B depends only on (bands, seed),
    so it can be rebuilt from config instead of being checkpointed.
    """
    g = torch.Generator().manual_seed(seed)
    blocks = []
    for n, sigma in bands:
        Bm = torch.randn(len(sigma), n, generator=g)
        # One row per coordinate (x, [y,] t), each scaled by its sigma, per band.
        blocks.append(Bm * torch.tensor(sigma, dtype=Bm.dtype).view(-1, 1))
    return torch.cat(blocks, dim=1)

def build_B(bands, seed, fourier, N_input = 2):
    """Fixed frequencies [N_input, n_freq], rebuilt from config, never a parameter or buffer.
    N_input = D+1 coordinates; each band gives sigma_x, sigma_y (2D only) and sigma_t.
    fourier off -> n_freq = 0, so fourier_features returns just the coordinates."""
    if not fourier:
        return torch.zeros(N_input, 0)
    keys = ["sigma_x", "sigma_y", "sigma_t"] if N_input == 3 else ["sigma_x", "sigma_t"]
    return make_B([(b["n_freq"], tuple(b[k] for k in keys)) for b in bands], seed)
