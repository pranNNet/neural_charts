import torch
import torch.nn as nn

from model.transforms import make_B, MSFourier_Features
# ======================================================================================
# Region Embedding
# ======================================================================================

class RegionEmbedding(nn.Module):
    '''
    Region id -> e_k [8]. Table [K, 8] = 40 parameters for K = 5.
    '''

    def __init__(self, N_regions: int = 5,
                 dim: int = 8):
        super().__init__()

        self.table = nn.Embedding(N_regions, dim)
        self.register_buffer("ids", torch.arange(N_regions), persistent = False)

    def forward(self) -> torch.Tensor:
        """All K codes at once: [K, 8]. Same as self.table.weight, written as a lookup."""
        return self.table(self.ids)

def build_cond(z_field: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """z_field [N, 32], e [K, 8] -> cond [N, K, 40]. expand() is a view, no copy."""

    N, K = z_field.shape[0], e.shape[0]
    z = z_field.unsqueeze(1).expand(N, K, -1)        # [N, K, 32]: same z for every region
    ek = e.unsqueeze(0).expand(N, K, -1)             # [N, K, 8]:  same e_k for every point
    return torch.cat([z, ek], dim=-1)

# ======================================================================================
# FiLM
# ======================================================================================

class FiLMLayer(nn.Module):

    def __init__(self, N_in, N_features, N_z, N_e):
        super().__init__()

        self.layer = nn.Linear(N_in, N_features)
        self.hyper = nn.Linear(N_z + N_e, 2 * N_features) #creates (gamma, beta) for its own layer, based on encoding

        # initalize FiLM at identity (gamma = beta = 0)
            # but cannot initiate at 0 because then gradient will be zero at beginning
            # so we initialize only part of weights as default 
                # -> the end results at this beginning state will be identity, but gradients are non-zero

        with torch.no_grad():
            self.hyper.weight[:, N_z:].zero_()
            self.hyper.bias.zero_()

    def forward(self, x, cond):
        gamma, beta = self.hyper(cond).chunk(2, dim=-1)

        return nn.functional.silu((1+gamma) * self.layer(x) + beta)
 
# ======================================================================================
# FieldNet
# ======================================================================================

class FieldNet(nn.Module):

    def __init__(self, N_in, N_regions,
                 N_z_field, N_e_field, N_dist,
                 hidden_layers):
        super().__init__()

        self.N_dist = N_dist

        N_in = N_in + N_dist

        self.embed = RegionEmbedding(N_regions, N_e_field)


        self.layers = nn.ModuleList(
            [FiLMLayer(N_in, hidden_layers[0], N_z_field, N_e_field)] + 
            [FiLMLayer(N_features, N_features, N_z_field, N_e_field) for N_features in hidden_layers[1:]] +
            [nn.Linear(hidden_layers[-1], 3)]
        )

    def forward(self, x, z_field, dist = None):

        if dist is None:
            dist = x.new_zeros(x.shape[0], self.N_dist)

        x = torch.cat([x, dist], dim=-1) # [N, 134] w/ fourier features

        e = self.embed() # [K, 8]
        cond = build_cond(z_field, e) # [N, K, 40]

        x = x.unsqueeze(1).expand(-1, e.shape[0], -1) # [N, K, 134] - one copy per zone

        for layer in self.layers[0:-1]:
            x = layer(x, cond)
        return self.layers[-1](x)


def positive_clamp(R):

    rho = nn.functional.softplus(R[..., 0])+1e-6
    u = R[..., 1]
    P = nn.functional.softplus(R[..., 2]) + 1e-6
    return torch.stack([rho, u, P], dim=-1)