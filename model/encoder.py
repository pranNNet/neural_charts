import torch
import torch.nn as nn


# ======================================================================================
# LatentEncoder
# ======================================================================================

class LatentEncoder(nn.Module):
    '''
    Encodes PDE/Field/Problem Paramters into a latent vector (z)
        - z[0:x] -> LevelSetsNet
        - z[x: ] -> RegionFieldNet
    '''

    def __init__(self, N_in, z_geo, z_field, 
                 hidden_layers = [64, 64, 64],
                 act = nn.SiLU):
        
        super().__init__()

        dims = [N_in] + hidden_layers + [z_geo + z_field]

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(act())

        self.body = nn.Sequential(*layers)
        self.head_geo = nn.Linear(dims[-1], z_geo)
        self.head_field = nn.Linear(dims[-1], z_field)

    def forward(self, x):
        y = self.body(x.to(torch.float32))
        return self.head_geo(y), self.head_field(y)




# ======================================================================================
# FieldEncoder
# ======================================================================================

class FieldEncoder(nn.Module):
    '''
    Encodes the current field W_grid [B, C, N_grid] into a latent vector (z). Same outputs as LatentEncoder.
        - standardize, then append the x~ coordinate as a channel, so where the fronts are survives
        - strided Conv1d blocks, flatten (no pooling: it would average away where the fronts are)
        - optional case parameters, then LatentEncoder's MLP and its two heads
    '''

    def __init__(self, N_channels, N_grid, z_geo, z_field,
                 conv_channels = [32, 64, 64, 64], kernel = 5,
                 hidden_layers = [64, 64, 64], N_param = 0,
                 act = nn.SiLU):

        super().__init__()

        # set once from the training data (enc.w_mean.copy_(...)); saved with the checkpoint
        self.register_buffer("w_mean", torch.zeros(N_channels, 1))
        self.register_buffer("w_std", torch.ones(N_channels, 1))
        self.register_buffer("coord", (-1 + (2 * torch.arange(N_grid) + 1) / N_grid).view(1, 1, -1))

        dims = [N_channels + 1] + conv_channels

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Conv1d(dims[i], dims[i+1], kernel, stride = 2, padding = kernel // 2))
            layers.append(act())

        self.conv = nn.Sequential(*layers)
        N_flat = conv_channels[-1] * N_grid // 2 ** len(conv_channels)     # each block halves the grid
        self.mlp = LatentEncoder(N_flat + N_param, z_geo, z_field, hidden_layers, act)

        # ⚑ He init, zero bias: with PyTorch's default init the differences between inputs shrink
        # ~10x per layer, so at the start every input gives nearly the same z
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity = 'relu')
                nn.init.zeros_(m.bias)

    def forward(self, W_grid, params = None):
        h = (W_grid - self.w_mean) / self.w_std
        h = torch.cat([h, self.coord.expand(h.shape[0], -1, -1)], dim=1)   # [B, C+1, N_grid]
        h = self.conv(h).flatten(1)
        if params is not None:
            h = torch.cat([h, params], dim=-1)
        return self.mlp(h)
