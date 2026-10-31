import torch


# ======================================================================================
# Multi-Scale Fourier Transform
# ======================================================================================


def MSFourier_Features(x: torch.Tensor, B:torch.Tensor) -> torch.Tensor:
    """[N, D] normalized space -> [N, D + 2*n_freq]: [x, sin(2pi x@B), cos(2pi x@B)].    # AUDIT: docstring
 
    Pure function of x. No detach: autograd still flows to x (needed for normals).
    """
    proj = 2 * torch.pi * (x @ B.to(x))
    return torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=1)

def make_B(bands, seed: int = 0) -> torch.Tensor:
    """Draw the fixed frequency matrix B [D, sum(n_freq)].                    # AUDIT: docstring
 
    bands: list of (n_freq, (sigma_x, [sigma_y])), one entry per scale.
    Local generator, bands drawn in order: B depends only on (bands, seed),
    so it can be rebuilt from config instead of being checkpointed.
    """
    g = torch.Generator().manual_seed(seed)
    blocks = []
    for n, sigma in bands:
        Bm = torch.randn(len(sigma), n, generator=g)
        # One row per coordinate (x, [y]), each scaled by its sigma, per band.
        blocks.append(Bm * torch.tensor(sigma, dtype=Bm.dtype).view(-1, 1))
    return torch.cat(blocks, dim=1)

def build_B(bands, seed, fourier, N_input = 1):
    """Fixed frequencies [N_input, n_freq], rebuilt from config, never a parameter or buffer.
    N_input = D space coordinates (no t); each band gives sigma_x, and sigma_y in 2D.
    fourier off -> n_freq = 0, so fourier_features returns just the coordinates."""
    if not fourier:
        return torch.zeros(N_input, 0)
    keys = ["sigma_x", "sigma_y"][:N_input]
    return make_B([(b["n_freq"], tuple(b[k] for k in keys)) for b in bands], seed)
