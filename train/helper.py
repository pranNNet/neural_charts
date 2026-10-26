import torch
import numpy as np

# ======================================================================================
# Losses
# ======================================================================================

def boundary_loss(model, batch, x_cell, z_geo, z_field, eps):


    B, n = len(batch['t']), len(x_cell)
    x = torch.stack([x_cell.expand(B, n), batch['t'][:, None].expand(B, n)], dim=-1).reshape(-1, 2)

    # w = model(x, z_geo.repeat_interleave(n, 0), z_field.repeat_interleave(n, 0),
    #           blend = 'soft', eps = eps)['weights'].view(B, n, -1) # [B, 512, K]

    # same = (w[:, :-1] * w[:, 1:]).sum(-1).clamp(1e-6, 1-1e-6 ) # [B, 511]
    # flags = batch['flags']

    # return -(same[flags==0].log().mean() + (1 - same[flags==1]).log().mean())


    phi_hat = model(x, z_geo.repeat_interleave(n, 0), z_field.repeat_interleave(n, 0),
              blend = 'soft', eps = eps)['phi_hat'].view(B, n, -1) # [B, 512, K]

    K = phi_hat.shape[-1]

    logw = torch.log_softmax(phi_hat/eps, dim=-1)
    pair = logw[:, :-1, :, None] + logw[:, 1:, None, :]

    log_same = pair.diagonal(dim1=-2, dim2=-1).logsumexp(-1)
    eye = torch.eye(K, dtype=torch.bool, device=pair.device)

    log_diff = pair.masked_fill(eye, float('-inf')).logsumexp((-2, -1))

    f = batch['flags']
    smooth = -log_same[f == 0].mean()
    jump = -log_diff[f == 1].mean()
    return smooth + jump, smooth, jump

    # same = (w[:, :-1] * w[:, 1:]).sum(-1).clamp(1e-6, 1-1e-6 ) # [B, 511]
    # flags = batch['flags']

    # return -(same[flags==0].log().mean() + (1 - same[flags==1]).log().mean())

def cell_L1(model, batch, x_cell, z_geo, z_field, blend, eps=None):
    B, n = len(batch['t']), len(x_cell)
    x = torch.stack([x_cell.expand(B, n), batch['t'][:, None].expand(B, n)], dim=-1).reshape(-1, 2)

    group = torch.arange(B, device = x_cell.device).repeat_interleave(n)

    W = model(x, z_geo[group], z_field[group],
              blend = blend, eps = eps, group = group)['W'].view(B, n, 3)

    return (W-batch['W']).abs().mean()

@torch.no_grad
def val_L1(model, encoder, val_loader, x_cell):
    out = []
    for batch in val_loader:
        batch = {key: val.to(x_cell.device) for key, val in batch.items()}

        z_geo, z_field = encoder(batch['r'])
        out.append(cell_L1(model, batch, x_cell, 
                           z_geo, z_field, 
                           blend='hard').item())
    return np.mean(out)


# ======================================================================================
# One-region support
# ======================================================================================

def geo_net(model):
    '''The geometry net, or None in the one-region arm (architecture.ablation.geometry = none).'''
    return model.LevelSetsNet if model.geo_mode == 'levelset' else None

def set_encoder_trainable(encoder, which):
    '''which: 'none', 'head' (z_field head only) or 'all'. Returns the trainable encoder params.'''
    encoder.requires_grad_(which == 'all')
    if which == 'head':
        encoder.head_field.requires_grad_(True)
    return [p for p in encoder.parameters() if p.requires_grad]
