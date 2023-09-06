import os
import math
import time
import torch
import numpy as np
import random
import yaml
from tqdm import tqdm
import wandb
from Dataset import Combplex_Dataset
from models.model import Model
from util.losses import LossG
from util.util import *

start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(callback=None):
    # load config file
    with open("config/config_train.yaml", "r") as f:
        config = yaml.safe_load(f)
    epsilon = 1e-7

    # wandb initializations
    model_name = config['model_name']
    if model_name == 'delete':
        wandb.init(project='Combplex',name=model_name, entity=config['W&B_entity'], config=config)
    else:    
        wandb.init(project='Combplex',name=model_name, id=model_name, entity=config['W&B_entity'], config=config, resume="allow")
    cfg = wandb.config
    outputs_dir = os.path.join('pretrained_models', wandb.run.name)
    os.makedirs(outputs_dir, exist_ok=True)

    # set seed
    seed = np.random.randint(2 ** 32 - 1, dtype=np.int64) if cfg['seed'] == None else cfg['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f'running with seed: {seed}.')

    # get variables from compression matrices form
    _, singles_names, multis_names, _, _, _ = read_compression_matrix_form(cfg['compression_matrix_form_path'])

    # create datasets
    train_dataset = Combplex_Dataset(cfg['training_dataset_path'], cfg, device)
    validation_dataset = Combplex_Dataset(cfg['validation_dataset_path'], cfg, device)

    # define model, loss function, optimizer and scheduler
    model = Model(in_channels=len(multis_names), n_classes=len(singles_names), filters=cfg['model_features'])
    criterion = LossG(cfg)
    optimizer = get_optimizer(cfg, model.parameters())
    scheduler = get_scheduler(optimizer, cfg)

    # resume exists checkpoint, if needed
    resume_path = f"pretrained_models/{cfg['model_name']}/{cfg['model_name']}-latest.pt"
    if os.path.exists(resume_path) and cfg['model_name'] != 'delete':
        checkpoint = torch.load(resume_path,  map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        current_epoch = checkpoint['step']
        optimal_validation_loss = checkpoint['optimal_validation_loss']
        print('loading ckpt from {}'.format(resume_path))
    else:
        optimal_validation_loss = math.inf
        current_epoch = 0
        
    # training process
    with tqdm(range(1, cfg['num_epochs'] + 1)) as tepoch:
        for epoch in tepoch:
            current_time = time.time()
            if current_time - start_time > cfg['running_time'] * 60 * 60:
                save_checkpoint(model, optimizer, current_epoch, optimal_validation_loss, outputs_dir, model_name, save_name='latest')
                break
            current_epoch += 1
            inputs = train_dataset[0]
            for key in inputs:
                inputs[key] = inputs[key].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            losses = criterion(outputs, inputs)
            loss_G = losses['loss']
            log_data = {**losses, 'epoch': current_epoch}

            # validation set
            if current_epoch % cfg['validation_frequency'] == 0:
                with torch.no_grad():
                    validation_inputs = validation_dataset[0]
                    for key in validation_inputs:
                        validation_inputs[key] = validation_inputs[key].to(device)
                    validation_outputs = model(validation_inputs)
                    validation_loss = criterion(validation_outputs, validation_inputs)
                    log_data['Validation loss'] = validation_loss['loss']
                if (validation_loss['loss'] < optimal_validation_loss) and current_epoch > 10000:
                    save_checkpoint(model, optimizer, current_epoch, optimal_validation_loss, outputs_dir, model_name, save_name='optimal_{}'.format(current_epoch))
                    optimal_validation_loss = validation_loss['loss']

            # reshape and create binary masks
            b, c = inputs['GT_singles'].shape[:2]
            target = (inputs['GT_singles'] > 0).float().reshape(b, c, -1)
            pred = (outputs['preds'].detach() > 0.5).float().reshape(b, c, -1)
            if current_epoch % cfg['validation_frequency'] == 0:
                validation_target = (validation_inputs['GT_singles'] > 0).float().reshape(b, c, -1)
                validation_pred = (validation_outputs['preds'].detach() > 0.5).float().reshape(b, c, -1)
            # calculate F1 scores for evaluation
            rand_idx = np.random.randint(0, cfg['batch_size'])
            for i, channel in enumerate(singles_names):
                target_i = target[rand_idx, i]
                pred_i = pred[rand_idx, i]
                f1 = (2 * (target_i * pred_i).sum() + epsilon) / (target_i.sum() + pred_i.sum() + epsilon)
                log_data[f'{channel}_f1'] = f1
                if current_epoch % cfg['validation_frequency'] == 0:
                    validation_target_i = validation_target[rand_idx, i]
                    validation_pred_i = validation_pred[rand_idx, i]
                    f1 = (2 * (validation_target_i * validation_pred_i).sum() + epsilon) / (validation_target_i.sum() + validation_pred_i.sum() + epsilon)
                    log_data[f'{channel}_validation_f1'] = f1
                    
            # update learning rate
            lr = optimizer.param_groups[0]['lr']
            log_data["lr"] = lr
            tepoch.set_description(f"Epoch {log_data['epoch']}")
            tepoch.set_postfix(loss=log_data["loss"].item())

            # save checkpoint
            if current_epoch % cfg['save_checkpoint_freq'] == 0:
                save_checkpoint(model, optimizer, current_epoch, optimal_validation_loss, outputs_dir, model_name, save_name='latest')
                save_checkpoint(model, optimizer, current_epoch, optimal_validation_loss, outputs_dir, model_name, save_name=f'{current_epoch}')

            wandb.log(log_data)
            loss_G.backward()
            optimizer.step()
            scheduler.step()


if __name__ == '__main__':
    train_model()
