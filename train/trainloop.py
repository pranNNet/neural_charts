from pathlib import Path
import os
import yaml
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from model.neural_chart import NeuralChart1DST
from train.dataset import load_splits, RolloutDataset
from train.helper import geometry_loss, cell_L1, front_loss, rollout_L1, val_L1
from train.helper import geo_net

torch.backends.cuda.matmul.allow_tf32 = True

# ======================================================================================
# NaN guard
# ======================================================================================

NAN_PATIENCE = 20       # consecutive non-finite steps before a stage gives up

def guarded_step(loss, params, optimizer, clip, state):
    '''Clip, then step only if the loss and the gradient norm are finite. A non-finite step is skipped (no weight
    update, so Adam's moments stay clean); NAN_PATIENCE in a row raises FloatingPointError, which stops this
    model's run. state: {'bad': consecutive, 'total': all skipped}, one per stage.'''
    gn = torch.nn.utils.clip_grad_norm_(params, clip)
    if torch.isfinite(loss) and torch.isfinite(gn):
        optimizer.step()
        state['bad'] = 0
        return True
    optimizer.zero_grad(set_to_none=True)
    state['bad'] += 1
    state['total'] += 1
    print(f'NaN GUARD: non-finite loss or gradient, step skipped ({state["total"]} so far, {state["bad"]} in a row)',
          flush=True)
    if state['bad'] >= NAN_PATIENCE:
        raise FloatingPointError(f'{NAN_PATIENCE} non-finite steps in a row; stopping this run')
    return False

# ======================================================================================
# Stage 1: geometry (segmentation of W^n)
# ======================================================================================

def train_geometry(model,
                   train_loader, val_loader,
                   x_cell, writer,
                   config = Path(__file__).parent.parent / 'configs' / 'train_config.yaml',
                   checkpoint_dir = Path(__file__).parent.parent / 'results' / 'stage1.pt',
                   tag = 'stage1'):                 # tensorboard prefix

    os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)

    # enable training on LevelSetsNet only (its CNN + INR head)
    model.requires_grad_(False)
    model.LevelSetsNet.requires_grad_(True)

    # set up training components
    params = list(model.LevelSetsNet.parameters())
    optimizer = torch.optim.Adam(params, lr=config['geometry']['lr'])

    step_tot = config['geometry']['epochs'] * len(train_loader)
    scheduler = SequentialLR(optimizer,
                             [LinearLR(optimizer, 0.01, 1.0, config['geometry']['warmup_steps']),
                              CosineAnnealingLR(optimizer, step_tot - config['geometry']['warmup_steps'],
                                                eta_min = config['geometry']['lr_min'])],
                                                [config['geometry']['warmup_steps']])

    #validation/checkpointing purposes
    step, best = 0, torch.inf
    nan_guard = {'bad': 0, 'total': 0}

    torch.save({'model': model.state_dict()}, checkpoint_dir)   # AUDIT 9: a checkpoint exists even if every val is NaN
    print(f"Beginning {tag} Training...")
    for epoch in range(config['geometry']['epochs']):
        print(f"-------------------------------\nEpoch {epoch+1}\n-------------------------------")
        tot_loss = 0

        for batch in train_loader:
            batch = {key: val.to(x_cell.device) for key, val in batch.items()}

            # DISTANCE TARGET: Huber of the zone scores vs phi* at t_n (centres + 1 random point per cell), replaces
            # CE + boundary loss. LOCAL GEOMETRY: no latent
            loss = geometry_loss(model, batch, x_cell)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            nets = {'geo_conv': model.LevelSetsNet.features, 'geo_inr': model.LevelSetsNet.net}
            for name, net in nets.items():
                g = [p.grad for p in net.parameters() if p.grad is not None]
                if g:   # frozen nets have no grads, skip them
                    writer.add_scalar(f'{tag}/grad_norm/{name}', torch.linalg.vector_norm(torch.stack([x.norm() for x in g])).item(), step)

            guarded_step(loss, params, optimizer, config['misc']['grad_clip'], nan_guard)   # NaN GUARD
            scheduler.step()

            writer.add_scalar(f'{tag}/train_loss', loss.item(), step)
            writer.add_scalar(f'{tag}/schedule/lr', scheduler.get_last_lr()[0], step)

            step += 1
            tot_loss += loss.item() if torch.isfinite(loss) else 0.0

        print('TRAIN:')
        print(f"loss: {tot_loss / len(train_loader):>7f}")
        #validation evaluation
        if (epoch+1)%25 == 0:
            print(f'VALIDATION [Epoch {epoch+1}]:')
            vals= []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {key: val.to(x_cell.device) for key, val in batch.items()}
                    vals.append(geometry_loss(model, batch, x_cell).item())

                val = np.mean(vals)
                writer.add_scalar(f'{tag}/val_loss', val, step)
                if val < best:
                    best = val
                    torch.save({'model': model.state_dict()},
                                checkpoint_dir)

                print(f"loss: {val:>7f}")

    optimizer.zero_grad(set_to_none=True)
    print(f"{tag} Training Complete!")
    print('\n')
    state = torch.load(checkpoint_dir)
    model.load_state_dict(state['model'])

# ======================================================================================
# Stage 2: field (one step, cut blend)
# ======================================================================================

def train_field(model,
                train_loader, val_loader,
                x_cell, writer,
                config = Path(__file__).parent.parent / 'configs' / 'train_config.yaml',
                checkpoint_dir = Path(__file__).parent.parent / 'results' / 'stage2.pt'):

    os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)

    # enable training on FieldNet only (LOCAL FIELDS: its FieldFeatureCNN included; no encoder)
    model.requires_grad_(False)
    model.FieldNet.requires_grad_(True)

    # set up training components
    params = list(model.FieldNet.parameters())
    optimizer = torch.optim.Adam(params, lr=config['field']['lr'])

    step_tot = config['field']['epochs']*len(train_loader)
    scheduler = SequentialLR(optimizer,
                             [LinearLR(optimizer, 0.01, 1.0, config['field']['warmup_steps']),
                              CosineAnnealingLR(optimizer, step_tot - config['field']['warmup_steps'],
                                                eta_min = config['field']['lr_min'])],
                                                [config['field']['warmup_steps']])

    #cvalidation/heckpointing purposes
    step, best = 0, torch.inf
    nan_guard = {'bad': 0, 'total': 0}
    noise = config['misc']['noise_std'] * model.w_std.view(-1)                 # [3], per variable

    torch.save({'model': model.state_dict()}, checkpoint_dir)   # AUDIT 9: a checkpoint exists even if every val is NaN
    print("Beginning Stage 2 Training...")
    for epoch in range(config['field']['epochs']):
        print(f"-------------------------------\nEpoch {epoch+1}\n-------------------------------")
        tot_loss = 0

        for batch in train_loader:
            batch = {key: val.to(x_cell.device) for key, val in batch.items()}

            # noisy W^n in, L1 at t_{n+1}
            W_in = batch['W_in'] + noise * torch.randn_like(batch['W_in'])
            loss, _ = cell_L1(model, W_in, batch['W_out'][:, 0], x_cell)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            nets = {'geo': geo_net(model), 'field': model.FieldNet}
            for name, net in nets.items():
                if net is None:              # one-region baseline has no geometry net
                    continue
                g = [p.grad for p in net.parameters() if p.grad is not None]
                if g and (step % 10 == 0):   # frozen nets have no grads, skip them
                    writer.add_scalar(f'stage2/grad_norm/{name}', torch.linalg.vector_norm(torch.stack([x.norm() for x in g])).item(), step)

            guarded_step(loss, params, optimizer, config['misc']['grad_clip'], nan_guard)   # NaN GUARD
            scheduler.step()

            writer.add_scalar('stage2/train_loss', loss.item(), step)
            writer.add_scalar('stage2/schedule/lr', scheduler.get_last_lr()[0], step)

            step += 1
            tot_loss += loss.item() if torch.isfinite(loss) else 0.0

        print('TRAIN:')
        print(f"loss: {tot_loss / len(train_loader):>7f}")
        #validation evaluation
        if (epoch+1)%25 == 0:

            val = val_L1(model, val_loader, x_cell)
            writer.add_scalar('stage2/val_loss', val, step)
            if val < best:
                best = val
                torch.save({'model': model.state_dict()},
                            checkpoint_dir)

            print(f'VALIDATION [Epoch {epoch+1}]:')
            print(f"loss: {val:>7f}")

    optimizer.zero_grad(set_to_none=True)
    print("Stage 2 Training Complete!")
    print('\n')
    state = torch.load(checkpoint_dir)
    model.load_state_dict(state['model'])

# ======================================================================================
# Stage 3: fine-tuning (one step, + front loss at n and n+1)
# ======================================================================================

def finetune(model,
             train_loader, val_loader,
             x_cell, writer,
             config = Path(__file__).parent.parent / 'configs' / 'train_config.yaml',
             checkpoint_dir = Path(__file__).parent.parent / 'results' / 'stage3.pt'):

    os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)

    # enable training on everything for fine-tuning
    model.requires_grad_(True)

    # set up training components (LOCAL FIELDS: each module brings its own CNN, no encoder)
    geo_params = list(geo_net(model).parameters()) if geo_net(model) is not None else []
    field_encode_params = list(model.FieldNet.parameters())
    optimizer = torch.optim.Adam([{'params': geo_params, 'lr': config['finetune']['geo_lr']},
                                  {'params': field_encode_params, 'lr': config['finetune']['field_lr']}])

    step_tot = config['finetune']['epochs']*len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, step_tot,
                                  eta_min = config['finetune']['lr_min']
                                  )

    #cvalidation/heckpointing purposes
    step, best = 0, torch.inf
    nan_guard = {'bad': 0, 'total': 0}
    noise = config['misc']['noise_std'] * model.w_std.view(-1)

    torch.save({'model': model.state_dict()}, checkpoint_dir)   # AUDIT 9: a checkpoint exists even if every val is NaN
    print("Beginning Stage 3 Training...")
    for epoch in range(config['finetune']['epochs']):
        print(f"-------------------------------\nEpoch {epoch+1}\n-------------------------------")
        tot_loss = 0

        for batch in train_loader:
            batch = {key: val.to(x_cell.device) for key, val in batch.items()}

            # noisy W^n in, L1 at t_{n+1}, plus CE at n and on the advected scores at n+1
            W_in = batch['W_in'] + noise * torch.randn_like(batch['W_in'])
            loss, out = cell_L1(model, W_in, batch['W_out'][:, 0], x_cell)     # cut blend (forward)

            if out['phi_hat'] is not None:       # one-region arm has no geometry
                ce = front_loss(out, batch['fronts'][:, :2], x_cell)     # DISTANCE TARGET at n and n+1
                loss = loss + config['finetune']['lambda_ce'] * ce
                writer.add_scalar('stage3/loss/front', ce, step)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            nets = {'geo': geo_net(model), 'field': model.FieldNet}
            for name, net in nets.items():
                if net is None:
                    continue
                g = [p.grad for p in net.parameters() if p.grad is not None]
                if g and (step % 10 == 0):   # frozen nets have no grads, skip them
                    writer.add_scalar(f'stage3/grad_norm/{name}', torch.linalg.vector_norm(torch.stack([x.norm() for x in g])).item(), step)

            guarded_step(loss, geo_params + field_encode_params, optimizer, config['misc']['grad_clip'], nan_guard)   # NaN GUARD
            scheduler.step()

            writer.add_scalar('stage3/train_loss', loss.item(), step)
            writer.add_scalar('stage3/schedule/lr_geo', scheduler.get_last_lr()[0], step)
            writer.add_scalar('stage3/schedule/lr_field_enc', scheduler.get_last_lr()[-1], step)

            step += 1
            tot_loss += loss.item() if torch.isfinite(loss) else 0.0

        print('TRAIN:')
        print(f"loss: {tot_loss / len(train_loader):>7f}")
        #validation evaluation
        if (epoch+1)%25 == 0:

            val = val_L1(model, val_loader, x_cell)
            writer.add_scalar('stage3/val_loss', val, step)
            if val < best:
                best = val
                torch.save({'model': model.state_dict()},
                        checkpoint_dir)

            print(f'VALIDATION [Epoch {epoch+1}]:')
            print(f"loss: {val:>7f}")

    optimizer.zero_grad(set_to_none=True)
    print("Stage 3 Training Complete!")
    print('\n')
    state = torch.load(checkpoint_dir)
    model.load_state_dict(state['model'])

# ======================================================================================
# Stage 4: rollout
# ======================================================================================

def train_rollout(model,
                  train_loaders, val_loader,
                  x_cell, writer,
                  config = Path(__file__).parent.parent / 'configs' / 'train_config.yaml',
                  checkpoint_dir = Path(__file__).parent.parent / 'results' / 'stage4.pt'):
    '''train_loaders: one R-step loader per entry of rollout.R. val_loader: full rollouts (R = None).'''

    os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)
    cfg = config['rollout']

    # everything trains, at one lower lr (LOCAL FIELDS: no encoder)
    model.requires_grad_(True)
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg['lr'])

    step_tot = cfg['epochs_per_R'] * sum(len(loader) for loader in train_loaders)
    scheduler = CosineAnnealingLR(optimizer, step_tot, eta_min = cfg['lr_min'])

    step = 0
    nan_guard = {'bad': 0, 'total': 0}
    best = val_L1(model, val_loader, x_cell)        # AUDIT fix: stage 4 keeps the incoming model unless it beats it

    torch.save({'model': model.state_dict()}, checkpoint_dir)   # AUDIT 9: a checkpoint exists even if every val is NaN
    print("Beginning Stage 4 Training...")
    for R, train_loader in zip(cfg['R'], train_loaders):
        for epoch in range(cfg['epochs_per_R']):
            print(f"-------------------------------\nR = {R}, Epoch {epoch+1}\n-------------------------------")
            tot_loss = 0

            for batch in train_loader:
                batch = {key: val.to(x_cell.device) for key, val in batch.items()}

                loss = rollout_L1(model, batch, x_cell, cfg.get('detach', False),
                                      cfg['lambda_ce'])                  # distance-target front loss keeps the fronts on the data

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # AUDIT 9: grad-norm logging, same block as stage 3
                nets = {'geo': geo_net(model), 'field': model.FieldNet}
                for name, net in nets.items():
                    if net is None:
                        continue
                    g = [p.grad for p in net.parameters() if p.grad is not None]
                    if g and (step % 10 == 0):
                        writer.add_scalar(f'stage4/grad_norm/{name}', torch.linalg.vector_norm(torch.stack([x.norm() for x in g])).item(), step)

                guarded_step(loss, params, optimizer, config['misc']['grad_clip'], nan_guard)   # NaN GUARD
                scheduler.step()

                writer.add_scalar('stage4/train_loss', loss.item(), step)
                writer.add_scalar('stage4/schedule/lr', scheduler.get_last_lr()[0], step)
                writer.add_scalar('stage4/schedule/R', R, step)

                step += 1
                tot_loss += loss.item() if torch.isfinite(loss) else 0.0

            print('TRAIN:')
            print(f"loss: {tot_loss / len(train_loader):>7f}")
            #validation evaluation: full rollout t0 -> T_train, comparable across R
            if (epoch+1)%25 == 0 or epoch+1 == cfg['epochs_per_R']:

                val = val_L1(model, val_loader, x_cell)
                writer.add_scalar('stage4/val_loss', val, step)
                if val < best:
                    best = val
                    torch.save({'model': model.state_dict()},
                               checkpoint_dir)

                print(f'VALIDATION [R = {R}, Epoch {epoch+1}]:')
                print(f"loss: {val:>7f}")

    optimizer.zero_grad(set_to_none=True)
    print("Stage 4 Training Complete!")
    print('\n')
    state = torch.load(checkpoint_dir)
    model.load_state_dict(state['model'])

# ======================================================================================
# Master
# ======================================================================================

def train_NC1DST(model_cfg = Path(__file__).parent.parent / 'configs' / 'model_config.yaml',
                 train_cfg = Path(__file__).parent.parent / 'configs' / 'train_config.yaml',
                 log_dir = Path(__file__).parent.parent / 'logs',
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 baseline = False):

        print(f'Using {device}\n')

        with open(model_cfg, 'r') as file:
            model_cfg = yaml.safe_load(file)
        with open(train_cfg, 'r') as file:
            train_cfg = yaml.safe_load(file)

        savepth = os.path.join(train_cfg['path']['save_path'], model_cfg['name'])
        checkpth = os.path.join(savepth, 'checkpoint')
        os.makedirs(log_dir, exist_ok=True)      # not dirname: a bare name like 'logs' gives '' and makedirs('') raises
        os.makedirs(checkpth, exist_ok=True)     # also creates save_path and savepth

        #data: one-step pairs for stages 1-3, R-step windows and full val rollouts for stage 4
        path, t0 = train_cfg['path']['data_path'], train_cfg['data']['t0']
        B_train = train_cfg['split']['train']['batch_size']
        B_val = train_cfg['split']['validate']['batch_size']

        train_ds, val_ds, ___,___ = load_splits(path, cls=RolloutDataset, t0=t0)
        train_loader = DataLoader(train_ds, batch_size=B_train, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=B_val)
        # stage 4 is cut: rollout loaders off
        # rollout_loaders = [DataLoader(RolloutDataset(path, 'train', R=R, t0=t0), batch_size=B_train, shuffle=True)
        #                    for R in train_cfg['rollout']['R']]
        # val_rollout = DataLoader(RolloutDataset(path, 'val', R=None, t0=t0), batch_size=B_val)
        x_cell = train_ds.x_centers.to(device)

        #initialize model
        model = NeuralChart1DST(config = model_cfg,
                                N_input = model_cfg['architecture']['coordinate_input']
                                ).to(device)
        # LOCAL FIELDS: no encoder. Per-variable stats over the training split, used by both CNNs; buffers, so
        # every checkpoint keeps them
        model.w_mean.copy_(train_ds.W.mean((0, 1, 2)).view_as(model.w_mean))
        model.w_std.copy_(train_ds.W.std((0, 1, 2)).view_as(model.w_std))

        writer = SummaryWriter(os.path.join(log_dir, model_cfg['name']))

        if not baseline:
            train_geometry(model = model,
                        train_loader = train_loader, val_loader = val_loader,
                        x_cell = x_cell, writer = writer,
                        config = train_cfg,
                        checkpoint_dir = os.path.join(checkpth, 'stage1.pt'))
        train_field(model = model,
                       train_loader = train_loader, val_loader = val_loader,
                       x_cell = x_cell, writer = writer,
                       config = train_cfg,
                       checkpoint_dir = os.path.join(checkpth, 'stage2.pt'))
        # no geometry refit: nothing is shared between the geometry and the fields, so stage 2 cannot move the geometry
        finetune(model = model,
                       train_loader = train_loader, val_loader = val_loader,
                       x_cell = x_cell, writer = writer,
                       config = train_cfg,
                       checkpoint_dir = os.path.join(checkpth, 'stage3.pt'))
        # stage 4 (rollout) is cut
        # train_rollout(model = model,
        #                train_loaders = rollout_loaders, val_loader = val_rollout,
        #                x_cell = x_cell, writer = writer,
        #                config = train_cfg,
        #                checkpoint_dir = os.path.join(checkpth, 'stage4.pt'))

        writer.close()

        torch.save({'model': model.state_dict()},
                   f=os.path.join(savepth, f'trained_model.pt'))
        return model

if __name__ == '__main__':
    train_NC1DST(model_cfg = Path(__file__).parent.parent / 'configs' / 'models_cfg' / 'model_one' / 'model_config.yaml',
                 train_cfg = Path(__file__).parent.parent / 'configs' / 'models_cfg' / 'model_one' / 'train_config.yaml')

    # train_NC1DST(model_cfg = Path(__file__).parent.parent / 'configs' / 'models_cfg' / 'baseline_one' / 'model_config.yaml',
    #              train_cfg = Path(__file__).parent.parent / 'configs' / 'models_cfg' / 'baseline_one' / 'train_config.yaml',
    #              baseline=True)

    # train_NC1DST(model_cfg = Path(__file__).parent.parent / 'configs' / 'models_cfg' / 'model_one_free' / 'model_config.yaml',
    #              train_cfg = Path(__file__).parent.parent / 'configs' / 'models_cfg' / 'model_one_free' / 'train_config.yaml')