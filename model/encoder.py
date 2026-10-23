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

            if i < len(dims) - 2:
                layers.append(act())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.to(torch.float32))




