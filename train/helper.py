import torch
import torch.nn.functional as F
import numpy as np

# ======================================================================================
# Points
# ======================================================================================

def points(x_cell, B):
    '''Query points for B items on the cell centres: x [B*512, 1] and group [B*512], the batch index.'''
    return x_cell.repeat(B)[:, None], torch.arange(B, device=x_cell.device).repeat_interleave(len(x_cell))

def z_penalty(z_geo, z_field):
    '''The lambda_z term, now shared by all four stages.'''
    return (z_geo.pow(2).sum(-1) + z_field.pow(2).sum(-1)).mean()

# ======================================================================================
# Losses
# ======================================================================================

def boundary_loss(phi, flags):                                                   # AUDIT 5: no eps
    '''Takes the raw zone scores [B*512, K] and flags [B, 511].'''
    K = phi.shape[-1]
    phi = phi.view(len(flags), -1, K)                                          # [B, 512, K]

    logw = torch.log_softmax(phi, dim=-1)
    pair = logw[:, :-1, :, None] + logw[:, 1:, None, :]

    log_same = pair.diagonal(dim1=-2, dim2=-1).logsumexp(-1)
    eye = torch.eye(K, dtype=torch.bool, device=pair.device)

    log_diff = pair.masked_fill(eye, float('-inf')).logsumexp((-2, -1))

    smooth = -log_same[flags == 0].mean()
    jump = -log_diff[flags == 1].mean()
    return smooth + jump, smooth, jump

def class_ce(phi, classes):                                                      # AUDIT 5: no eps
    '''CE of the raw zone scores against the classes [B, 512] (-1 ignored).'''
    return F.cross_entropy(phi, classes.reshape(-1), ignore_index=-1)

def front_loss(out, classes, flags):                                              # AUDIT fix: boundary loss after stage 1
    '''CE on the classes + boundary loss on the flags, on the scores at t_n and the advected ones at t_n+1.
    classes [B, 2, 512], flags [B, 2, 511] at n, n+1. The classes ignore the cells around a jump, so the
    boundary loss is what holds the front on the flagged face there; without it the front drifts in stage 3.'''
    return sum(class_ce(phi, classes[:, i]) + boundary_loss(phi, flags[:, i])[0]
               for i, phi in enumerate((out['phi_hat_now'], out['phi_hat'])))

def geometry_loss(model, encoder, batch, x_cell, lambda_ce):                     # AUDIT 5: no eps
    '''Stage 1: lambda_ce * CE on the classes + boundary loss on the flags, both at t_n.'''
    z_geo, z_field = encoder(batch['W_in'].transpose(1, 2))
    x, group = points(x_cell, len(z_geo))
    phi = model.geometry(x, z_geo[group])

    ce = class_ce(phi, batch['classes'][:, 0])
    bnd, smooth, jump = boundary_loss(phi, batch['flags'][:, 0])
    return lambda_ce * ce + bnd, ce, smooth, jump, z_geo, z_field

def cell_L1(model, W_in, W_target, x_cell, z_geo, z_field):                     # AUDIT fix: no blend / eps
    '''one step W_in [B, 512, 3] -> t_{n+1}, L1 against W_target. Also returns the output.'''
    B, n = W_in.shape[:2]
    x, group = points(x_cell, B)

    out = model(x, z_geo[group], z_field[group], W_in.transpose(1, 2), group)

    return (out['W'].view(B, n, 3) - W_target).abs().mean(), out

def rollout_L1(model, encoder, batch, x_cell, detach=False, lambda_ce=0.0):
    '''mean cell L1 over the R steps in batch['W_out'], each step fed the model's own output.
    Also returns the mean lambda_z penalty. detach=True cuts the graph between steps (each step trains on
    the model's own input, but no gradient flows back through it).
    lambda_ce > 0 adds the stage 3 front loss (at n and n+1) at every step, so the fronts stay on the data.'''
    W, R = batch['W_in'], batch['W_out'].shape[1]
    loss = pen = 0
    for k in range(R):
        z_geo, z_field = encoder(W.transpose(1, 2))
        l, out = cell_L1(model, W, batch['W_out'][:, k], x_cell, z_geo, z_field)
        if lambda_ce and out['phi_hat'] is not None:                       # AUDIT fix: stage 4 CE + boundary
            l = l + lambda_ce * front_loss(out, batch['classes'][:, k:k + 2], batch['flags'][:, k:k + 2])
        W = out['W'].view_as(W)
        if detach:                                                          # AUDIT: every step, was every other
            W = W.detach()
        loss, pen = loss + l, pen + z_penalty(z_geo, z_field)
    return loss / R, pen / R

@torch.no_grad
def val_L1(model, encoder, val_loader, x_cell):
    '''one-step L1 on an R = 1 loader, full-rollout L1 on an R = None loader.'''
    out = []
    for batch in val_loader:
        batch = {key: val.to(x_cell.device) for key, val in batch.items()}
        out.append(rollout_L1(model, encoder, batch, x_cell)[0].item())
    return np.mean(out)


# ======================================================================================
# One-region support
# ======================================================================================

def geo_net(model):
    '''The geometry net, or None in the one-region arm (architecture.ablation.geometry = none).'''
    return model.LevelSetsNet if model.geo_mode == 'levelset' else None

def set_encoder_trainable(encoder, which):
    '''which: 'none', 'head' (z_field head only), 'geo_head' (z_geo head only) or 'all'. Returns the trainable encoder params.'''
    encoder.requires_grad_(which == 'all')
    if which == 'head':
        encoder.mlp.head_field.requires_grad_(True)     # FieldEncoder keeps its heads in .mlp
    if which == 'geo_head':
        encoder.mlp.head_geo.requires_grad_(True)
    return [p for p in encoder.parameters() if p.requires_grad]
