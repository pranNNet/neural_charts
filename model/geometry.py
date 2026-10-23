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









