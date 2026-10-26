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
from model.encoder import LatentEncoder
from train.dataset import load_splits
from train.helper import boundary_loss, cell_L1, val_L1
from train.helper import geo_net, set_encoder_trainable


torch.backends.cuda.matmul.allow_tf32 = True



# ======================================================================================
# Stage 1: geometry
# ======================================================================================

def train_geometry(model, encoder, 
                   train_loader, val_loader, 
                   x_cell, writer,
                   config = Path(__file__).parent.parent / 'configs' / 'train_config.yaml',
                   checkpoint_dir = Path(__file__).parent.parent / 'results' / 'stage1.pt'):
    
    os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)

    # enable training on LevelSetNet only
    model.requires_grad_(False)
    model.LevelSetsNet.requires_grad_(True)

    # set up training components
    params = list(model.LevelSetsNet.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=config['geometry']['lr'])

    step_tot = config['geometry']['epochs'] * len(train_loader)
    scheduler = SequentialLR(optimizer, 
                             [LinearLR(optimizer, 0.01, 1.0, config['geometry']['warmup_steps']),
                              CosineAnnealingLR(optimizer, step_tot - config['geometry']['warmup_steps'],
                                                eta_min = config['geometry']['lr_min'])], 
                                                [config['geometry']['warmup_steps']])

    #validation/checkpointing purposes
    step, best = 0, torch.inf

    print("Beginning Stage 1 Training...")
    for epoch in range(config['geometry']['epochs']):
        print(f"-------------------------------\nEpoch {epoch+1}\n-------------------------------")
        tot_loss = 0

        for batch in train_loader:
            batch = {key: val.to(x_cell.device) for key, val in batch.items()}

            # eps = 0.1 * ((config['geometry']['eps_min']/config['geometry']['eps0']) **
            #               min(1.0, step / (0.7*step_tot)))

            # [FIX 5] geometric eps0 -> eps_min over the first 70%, then held
            eps = config['geometry']['eps0'] * (config['geometry']['eps_min']/config['geometry']['eps0']) ** \
                  min(1.0, step / (0.7*step_tot))
 

            z_geo, z_field = encoder(batch['r'])
            loss, smooth, jump = boundary_loss(model=model,
                                                batch=batch,
                                                x_cell=x_cell,
                                                z_geo=z_geo,
                                                z_field=z_field,
                                                eps=eps)

            optimizer.zero_grad(set_to_none=True)
            (loss + config['misc']['lambdaZ'] * 
             (z_geo.pow(2).sum(-1)+z_field.pow(2).sum(-1)).mean()).backward()

            nets = {'geo': model.LevelSetsNet, 'field': model.FieldNet, 'enc': encoder}
            for name, net in nets.items():
                g = [p.grad for p in net.parameters() if p.grad is not None]
                if g:   # frozen nets have no grads, skip them
                    writer.add_scalar(f'stage1/grad_norm/{name}', torch.linalg.vector_norm(torch.stack([x.norm() for x in g])).item(), step)


            torch.nn.utils.clip_grad_norm_(params, config['misc']['grad_clip'])
            optimizer.step()
            scheduler.step()

            writer.add_scalar('stage1/train_loss', loss.item(), step)
            writer.add_scalar('stage1/schedule/eps', eps, step)
            writer.add_scalar('stage1/schedule/lr', scheduler.get_last_lr()[0], step)
            writer.add_scalar('stage1/latent/z_geo norm', z_geo.norm(dim=-1).mean(), step)
            writer.add_scalar('stage1/latent/z_field norm', z_field.norm(dim=-1).mean(), step)
            writer.add_scalar('stage1/loss/jump loss', jump, step)
            writer.add_scalar('stage1/loss/smooth loss', smooth, step)


            step += 1
            tot_loss += loss.item()

        print('TRAIN:')
        print(f"loss: {tot_loss / len(train_loader):>7f}")
        #validation evaluation
        if (epoch+1)%25 == 0:
            print(f'VALIDATION [Epoch {epoch+1}]:')
            vals= []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {key: val.to(x_cell.device) for key, val in batch.items()}
                    z_geo, z_field = encoder(batch['r'])

                    vals.append(boundary_loss(model, batch, x_cell, 
                                              z_geo, z_field,
                                               config['geometry']['eps_min'])[0].item())
     
                val = np.mean(vals)
                writer.add_scalar('stage1/val_loss', val, step)
                if val < best:
                    best = val
                    torch.save({'model': model.state_dict(), 'encoder': encoder.state_dict()}, 
                                checkpoint_dir)
                    
                
                print(f"loss: {val:>7f}")

    optimizer.zero_grad(set_to_none=True)
    print("Stage 1 Training Complete!")
    print('\n')
    state = torch.load(checkpoint_dir)
    model.load_state_dict(state['model'])
    encoder.load_state_dict(state['encoder'])


# ======================================================================================
# Stage 2: field
# ======================================================================================

def train_field(model, encoder, 
                train_loader, val_loader, 
                x_cell, writer,
                config = Path(__file__).parent.parent / 'configs' / 'train_config.yaml',
                checkpoint_dir = Path(__file__).parent.parent / 'results' / 'stage2.pt'):
    
    os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)

    # enable training on FieldNet and z_field encoderhead only
    model.requires_grad_(False)
    model.FieldNet.requires_grad_(True)

    # encoder.requires_grad_(False)
    # encoder.head_field.requires_grad_(True)

    if config['architecture']['ablation']['geometry'] == 'levelset':
        enc_params = set_encoder_trainable(encoder, config['field'].get('encoder', 'head'))
    else: 
        enc_params = set_encoder_trainable(encoder, config['field'].get('encoder', 'all'))
    # set up training components
    params = list(model.FieldNet.parameters()) + enc_params#list(encoder.head_field.parameters())
    optimizer = torch.optim.Adam(params, lr=config['field']['lr'])

    step_tot = config['field']['epochs']*len(train_loader)
    scheduler = SequentialLR(optimizer, 
                             [LinearLR(optimizer, 0.01, 1.0, config['field']['warmup_steps']),
                              CosineAnnealingLR(optimizer, step_tot - config['field']['warmup_steps'],
                                                eta_min = config['field']['lr_min'])], 
                                                [config['field']['warmup_steps']])

    #cvalidation/heckpointing purposes
    step, best = 0, torch.inf

    print("Beginning Stage 2 Training...")
    for epoch in range(config['field']['epochs']):
        print(f"-------------------------------\nEpoch {epoch+1}\n-------------------------------")
        tot_loss = 0

        for batch in train_loader:
            batch = {key: val.to(x_cell.device) for key, val in batch.items()}

            z_geo, z_field = encoder(batch['r'])
            loss = cell_L1(model=model,
                        batch=batch,
                        x_cell=x_cell,
                        z_geo=z_geo,
                        z_field=z_field,
                        blend='hard'
                        )

            optimizer.zero_grad(set_to_none=True)
            (loss + config['misc']['lambdaZ'] * 
            (z_geo.pow(2).sum(-1)+z_field.pow(2).sum(-1)).mean()).backward()


            nets = {'geo': geo_net(model), 'field': model.FieldNet, 'enc': encoder}
            for name, net in nets.items():
                if net is None:              # one-region baseline has no geometry net
                    continue
                g = [p.grad for p in net.parameters() if p.grad is not None]
                if g and (step % 10 == 0):   # frozen nets have no grads, skip them
                    writer.add_scalar(f'stage2/grad_norm/{name}', torch.linalg.vector_norm(torch.stack([x.norm() for x in g])).item(), step)
                        
            torch.nn.utils.clip_grad_norm_(params, config['misc']['grad_clip'])     
            optimizer.step()
            scheduler.step()

            writer.add_scalar('stage2/train_loss', loss.item(), step)
            writer.add_scalar('stage2/schedule/lr', scheduler.get_last_lr()[0], step)
            writer.add_scalar('stage2/latent/z_geo norm', z_geo.norm(dim=-1).mean(), step)
            writer.add_scalar('stage2/latent/z_field norm', z_field.norm(dim=-1).mean(), step)

            step += 1
            tot_loss += loss.item()


        print('TRAIN:')
        print(f"loss: {tot_loss / len(train_loader):>7f}")
        #validation evaluation
        if (epoch+1)%25 == 0:

            val = val_L1(model, encoder, val_loader, x_cell)
            writer.add_scalar('stage2/val_loss', val, step)
            if val < best:
                best = val
                torch.save({'model': model.state_dict(), 'encoder': encoder.state_dict()}, 
                            checkpoint_dir)

            print(f'VALIDATION [Epoch {epoch+1}]:')
            print(f"loss: {val:>7f}")

    optimizer.zero_grad(set_to_none=True)
    print("Stage 2 Training Complete!")
    print('\n')
    state = torch.load(checkpoint_dir)
    model.load_state_dict(state['model'])
    encoder.load_state_dict(state['encoder'])

# ======================================================================================
# Stage 3: fine-tuning
# ======================================================================================

def finetune(model, encoder, 
             train_loader, val_loader, 
             x_cell, writer,
             config = Path(__file__).parent.parent / 'configs' / 'train_config.yaml',
             checkpoint_dir = Path(__file__).parent.parent / 'results' / 'stage3.pt'):

    os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)

    # enable training on everything for fine-tuning
    model.requires_grad_(True)
    # encoder.requires_grad_(True)

    enc_params = set_encoder_trainable(encoder, config['finetune'].get('encoder', 'none'))

    # set up training components
    geo_params = list(geo_net(model).parameters()) if geo_net(model) is not None else []
    field_encode_params = list(model.FieldNet.parameters()) + enc_params #list(encoder.parameters())
    optimizer = torch.optim.Adam([{'params': geo_params, 'lr': config['finetune']['geo_lr']},
                                  {'params': field_encode_params, 'lr': config['finetune']['field_lr']}])

    step_tot = config['finetune']['epochs']*len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, step_tot,
                                  eta_min = config['finetune']['lr_min']
                                  )

    #cvalidation/heckpointing purposes
    step, best = 0, torch.inf

    print("Beginning Stage 3 Training...")
    for epoch in range(config['finetune']['epochs']):
        print(f"-------------------------------\nEpoch {epoch+1}\n-------------------------------")
        tot_loss = 0

        for batch in train_loader:
            batch = {key: val.to(x_cell.device) for key, val in batch.items()}
            z_geo, z_field = encoder(batch['r'])

            loss = cell_L1(model=model,
                           batch=batch,
                           x_cell=x_cell,
                           z_geo=z_geo,
                           z_field=z_field,
                           blend='soft',
                           eps = config['finetune']['eps']
                           )

            optimizer.zero_grad(set_to_none=True)
            (loss + config['misc']['lambdaZ'] * 
             (z_geo.pow(2).sum(-1)+z_field.pow(2).sum(-1)).mean()).backward()


            nets = {'geo': geo_net(model), 'field': model.FieldNet, 'enc': encoder}
            for name, net in nets.items():
                if net is None:
                    continue
                g = [p.grad for p in net.parameters() if p.grad is not None]
                if g and (step % 10 == 0):   # frozen nets have no grads, skip them
                    writer.add_scalar(f'stage3/grad_norm/{name}', torch.linalg.vector_norm(torch.stack([x.norm() for x in g])).item(), step)
                        
            torch.nn.utils.clip_grad_norm_(geo_params + field_encode_params, config['misc']['grad_clip'])     
            optimizer.step()
            scheduler.step()

            writer.add_scalar('stage3/train_loss', loss.item(), step)
            writer.add_scalar('stage3/schedule/lr_geo', scheduler.get_last_lr()[0], step)
            writer.add_scalar('stage3/schedule/lr_field_enc', scheduler.get_last_lr()[-1], step)
            writer.add_scalar('stage3/latent/z_geo norm', z_geo.norm(dim=-1).mean(), step)
            writer.add_scalar('stage3/latent/z_field norm', z_field.norm(dim=-1).mean(), step)

            step += 1
            tot_loss += loss.item()

        print('TRAIN:')
        print(f"loss: {tot_loss / len(train_loader):>7f}")
        #validation evaluation
        if (epoch+1)%25 == 0:

            val = val_L1(model, encoder, val_loader, x_cell)
            writer.add_scalar('stage3/val_loss', val, step)
            if val < best:
                best = val
                torch.save({'model': model.state_dict(), 'encoder': encoder.state_dict()}, 
                        checkpoint_dir)
                
            print(f'VALIDATION [Epoch {epoch+1}]:')
            print(f"loss: {val:>7f}")


    optimizer.zero_grad(set_to_none=True)
    print("Stage 3 Training Complete!")
    print('\n')
    state = torch.load(checkpoint_dir)
    model.load_state_dict(state['model'])
    encoder.load_state_dict(state['encoder'])

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
        os.makedirs(os.path.dirname(log_dir), exist_ok=True)
        os.makedirs(os.path.dirname(train_cfg['path']['save_path']), exist_ok=True)
        os.makedirs(os.path.dirname(checkpth), exist_ok=True)

        #data
        train_ds, val_ds, ___,___ = load_splits(train_cfg['path']['data_path'])
        train_loader = DataLoader(train_ds, 
                                  batch_size=train_cfg['split']['train']['batch_size'],
                                  shuffle=True)
        val_loader = DataLoader(val_ds, 
                                batch_size=train_cfg['split']['validate']['batch_size'])
        x_cell = train_ds.x_centers.to(device)

        #initialize model
        model = NeuralChart1DST(config = model_cfg,
                                N_input = model_cfg['architecture']['coordinate_input']
                                ).to(device)
        encoder = LatentEncoder(N_in = model_cfg['encoder']['encodernet']['input'],
                                z_geo = model_cfg['architecture']['latent']['z_geo'],
                                z_field = model_cfg['architecture']['latent']['z_field'],
                                hidden_layers = model_cfg['encoder']['encodernet']['hidden_layers']
                                ).to(device)

        writer = SummaryWriter(os.path.join(log_dir, model_cfg['name']))

        if not baseline:
            train_geometry(model = model, encoder = encoder, 
                        train_loader = train_loader, val_loader = val_loader, 
                        x_cell = x_cell, writer = writer, 
                        config = train_cfg,
                        checkpoint_dir = os.path.join(checkpth, 'stage1.pt'))
        train_field(model = model, encoder = encoder, 
                       train_loader = train_loader, val_loader = val_loader, 
                       x_cell = x_cell, writer = writer, 
                       config = train_cfg,
                       checkpoint_dir = os.path.join(checkpth, 'stage2.pt'))
        finetune(model = model, encoder = encoder, 
                       train_loader = train_loader, val_loader = val_loader, 
                       x_cell = x_cell, writer = writer, 
                       config = train_cfg,
                       checkpoint_dir = os.path.join(checkpth, 'stage3.pt'))

        writer.close()

        torch.save({'model': model.state_dict(), 'encoder': encoder.state_dict()}, 
                   f=os.path.join(savepth, f'trained_model.pt'))
        return model, encoder

if __name__ == '__main__':
    train_NC1DST(model_cfg = Path(__file__).parent.parent / 'configs' / 'models_cfg' / 'model_one' / 'model_config.yaml',
                 train_cfg = Path(__file__).parent.parent / 'configs' / 'models_cfg' / 'model_one' / 'train_config.yaml')
    
    # train_NC1DST(model_cfg = Path(__file__).parent.parent / 'configs' / 'models_cfg' / 'baseline_one' / 'model_config.yaml',
    #              train_cfg = Path(__file__).parent.parent / 'configs' / 'models_cfg' / 'baseline_one' / 'train_config.yaml',
    #              baseline=True)

    # train_NC1DST(model_cfg = Path(__file__).parent.parent / 'configs' / 'models_cfg' / 'model_one_free' / 'model_config.yaml',
    #              train_cfg = Path(__file__).parent.parent / 'configs' / 'models_cfg' / 'model_one_free' / 'train_config.yaml')








         

     





