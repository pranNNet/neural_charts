import torch


# ======================================================================================
# Multi-Scale Fourier Transform
# ======================================================================================


def MSFourier_Features(x: torch.Tensor, B:torch.Tensor) -> torch.Tensor:
    """[N, 2] normalized (x~, t~) -> [N, 2 + 2*n_freq]: [x~, t~, sin(2pi xt@B), cos(2pi xt@B)].
 
    Pure function of xt. No detach: autograd still flows to xt (needed for normals).
    """
    proj = 2 * torch.pi * (x @ B.to(x))
    return torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=1)

def make_B(bands, seed: int = 0) -> torch.Tensor:
    """Draw the fixed frequency matrix B [2, sum(n_freq)].
 
    bands: list of (n_freq, (sigma_x, sigma_t)), one entry per scale.
    Local generator, bands drawn in order: B depends only on (bands, seed),
    so it can be rebuilt from config instead of being checkpointed.
    """
    g = torch.Generator().manual_seed(seed)
    blocks = []
    for n, sigma in bands:
        Bm = torch.randn(2, n, generator=g)
        # Row 0 scales x frequencies, row 1 scales t frequencies, per band.
        blocks.append(Bm * torch.tensor(sigma, dtype=Bm.dtype).view(2, 1))
    return torch.cat(blocks, dim=1)

def build_B(bands, seed, fourier):
    """Fixed frequencies [2, n_freq], rebuilt from config, never a parameter or buffer.
    fourier off -> n_freq = 0, so fourier_features returns just [x~, t~]."""
    if not fourier:
        return torch.zeros(2, 0)
    return make_B([(n, tuple(sig)) for n, sig in bands], seed)

