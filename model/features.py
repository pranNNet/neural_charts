import torch
import torch.nn as nn

# ======================================================================================
# FeatureCNN: local features of the input field, shared design for the geometry and the fields
# ======================================================================================

class FeatureCNN(nn.Module):
    '''
    Standardized W(t_n) [B, C, N] -> feature grid [B, N_feat, N], once per step.          # LOCAL FIELDS
        - len(conv_channels) + 1 Conv1d layers, stride 1, replicate padding, SiLU between, the last linear
        - no coordinate channel: nothing says where, so the same input near x gives the same features at any x
        - cell i sees i +- receptive cells, receptive = (len(conv_channels) + 1) * (kernel // 2)
    The model standardizes once (NeuralChart1DST.w_mean / w_std); both CNNs take the standardized grid.
    '''

    def __init__(self, N_channels, conv_channels = [64, 64, 64], N_feat = 32, kernel = 5, act = nn.SiLU):
        super().__init__()
        dims = [N_channels] + list(conv_channels) + [N_feat]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Conv1d(dims[i], dims[i+1], kernel, stride = 1,
                                    padding = kernel // 2, padding_mode = 'replicate'))
            if i < len(dims) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)
        self.N_feat = N_feat
        self.receptive = (len(dims) - 1) * (kernel // 2)

        # ⚑ He init, zero bias (default init shrinks input differences layer by layer)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity = 'relu')
                nn.init.zeros_(m.bias)

    def forward(self, W_std):
        return self.net(W_std)