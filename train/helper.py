import torch
import torch.nn.functional as F
import numpy as np

from train.fronts import target_scores

# ======================================================================================
# Points
# ======================================================================================

def points(x_cell, B):
    '''Query points for B items on the cell centres: x [B*512, 1] and group [B*512], the batch index.'''
    return x_cell.repeat(B)[:, None], torch.arange(B, device=x_cell.device).repeat_interleave(len(x_cell))

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

def distance_loss(phi, u, fronts):
    '''Huber of the raw zone scores phi [B*P, K] against the distance target phi* at the points u [B, P] (data x),
    fronts [B, F, 3] (train/fronts.py).'''
    K = phi.shape[-1]
    return F.huber_loss(phi, target_scores(u, fronts, K).reshape(-1, K))

def front_loss(out, fronts, x_cell):                                              # DISTANCE TARGET
    '''Distance loss on the scores at t_n (phi_hat_now) and the advected ones at t_n+1 (phi_hat), on the cell
    centres. fronts [B, 2, F, 3] at n, n+1. Replaces CE + boundary loss, which pulled psi back to a step (zero on a
    cell centre, so a contact moving 0.5 cells per step never flipped a label).'''
    u = ((x_cell + 1) / 2).expand(len(fronts), -1)                             # [B, 512] data x
    return sum(distance_loss(phi, u, fronts[:, i])
               for i, phi in enumerate((out['phi_hat_now'], out['phi_hat'])))

def geometry_loss(model, batch, x_cell):                                         # DISTANCE TARGET
    '''Stage 1: distance loss at t_n, at the cell centres and one random point inside every cell (the zero lies
    between centres, and Newton reads it there). The whole stage 1 loss: replaces CE + boundary loss.
    LOCAL GEOMETRY: no encoder; the geometry reads W^n through LevelSetsNet's conv features.'''
    W_grid = batch['W_in'].transpose(1, 2)                                     # [B, 3, 512]
    B, dx = len(W_grid), x_cell[1] - x_cell[0]                                 # dx in x~
    xc = x_cell.expand(B, -1)
    x = torch.cat([xc, xc + (torch.rand_like(xc) - 0.5) * dx], 1)              # [B, 1024] x~
    group = torch.arange(B, device=x.device).repeat_interleave(x.shape[1])
    phi = model.geometry(x.reshape(-1, 1), W_grid, group)
    return distance_loss(phi, (x + 1) / 2, batch['fronts'][:, 0])

def cell_L1(model, W_in, W_target, x_cell):                                     # LOCAL FIELDS: no latents
    '''one step W_in [B, 512, 3] -> t_{n+1}, L1 against W_target. Also returns the output.'''
    B, n = W_in.shape[:2]
    x, group = points(x_cell, B)

    out = model(x, W_in.transpose(1, 2), group)

    return (out['W'].view(B, n, 3) - W_target).abs().mean(), out

def rollout_L1(model, batch, x_cell, detach=False, lambda_ce=0.0):
    '''mean cell L1 over the R steps in batch['W_out'], each step fed the model's own output.
    detach=True cuts the graph between steps (each step trains on
    the model's own input, but no gradient flows back through it).
    lambda_ce > 0 adds the stage 3 front loss (distance target at n and n+1) at every step, so the fronts stay on
    the data.'''
    W, R = batch['W_in'], batch['W_out'].shape[1]
    loss = 0
    for k in range(R):
        l, out = cell_L1(model, W, batch['W_out'][:, k], x_cell)
        if lambda_ce and out['phi_hat'] is not None:                       # stage 4 distance-target front loss
            l = l + lambda_ce * front_loss(out, batch['fronts'][:, k:k + 2], x_cell)
        W = out['W'].view_as(W)
        if detach:                                                          # AUDIT: every step, was every other
            W = W.detach()
        loss = loss + l
    return loss / R

@torch.no_grad
def val_L1(model, val_loader, x_cell):
    '''one-step L1 on an R = 1 loader, full-rollout L1 on an R = None loader.'''
    out = []
    for batch in val_loader:
        batch = {key: val.to(x_cell.device) for key, val in batch.items()}
        out.append(rollout_L1(model, batch, x_cell).item())
    return np.mean(out)

# ======================================================================================
# One-region support
# ======================================================================================

def geo_net(model):
    '''The geometry net, or None in the one-region arm (architecture.ablation.geometry = none).'''
    return model.LevelSetsNet if model.geo_mode == 'levelset' else None