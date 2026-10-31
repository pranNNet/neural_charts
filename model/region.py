import torch
import torch.nn as nn

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
    '''
    Shared bases: a trunk conditioned on z_field gives N_bases functions of x per output,
    and each region k mixes them with its own coefficients from [z_field; e_k].
        W_kc(x) = sum_m a_kmc * b_mc(x)
    '''

    def __init__(self, N_in, N_regions,
                 N_z_field, N_e_field,
                 hidden_layers, N_out = 3, N_bases = 16):                 # AUDIT 7: no N_dist
        super().__init__()

        self.N_out, self.N_bases = N_out, N_bases

        self.embed = RegionEmbedding(N_regions, N_e_field)

        # trunk: FiLM on z_field only (N_e = 0), runs once per point
        self.layers = nn.ModuleList(
            [FiLMLayer(N_in, hidden_layers[0], N_z_field, 0)] +
            [FiLMLayer(N_features, N_features, N_z_field, 0) for N_features in hidden_layers[1:]] +
            [nn.Linear(hidden_layers[-1], N_bases * N_out)]       # the bases b_mc(x)
        )

        # per-region coefficients a_kmc from [z_field; e_k], nonlinear in z (AUDIT test: was one Linear)
        self.coeff = nn.Sequential(nn.Linear(N_z_field + N_e_field, hidden_layers[-1]), nn.SiLU(),
                                   nn.Linear(hidden_layers[-1], N_bases * N_out))
        with torch.no_grad():
            self.coeff[-1].bias.fill_(1 / N_bases)       # ⚑ each region starts near the mean of the bases

    def forward(self, x, z_field):                                         # AUDIT 7: no dist input

        for layer in self.layers[0:-1]:
            x = layer(x, z_field)                                          # [N, H]

        N = x.shape[0]
        b = self.layers[-1](x).view(N, self.N_bases, self.N_out)                         # [N, M, F]
        e = self.embed()                                                   # [K, N_e]
        a = self.coeff(build_cond(z_field, e)).view(N, e.shape[0], self.N_bases, self.N_out)  # [N, K, M, F]
        return torch.einsum('nkmf,nmf->nkf', a, b)                         # [N, K, F]

def positive_clamp(R):

    # R = (rho, v_1..v_D, p): softplus on rho and p, velocity left free
    rho = nn.functional.softplus(R[..., :1])+1e-6
    u = R[..., 1:-1]
    P = nn.functional.softplus(R[..., -1:]) + 1e-6
    return torch.cat([rho, u, P], dim=-1)