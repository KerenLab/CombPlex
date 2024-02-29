import sys
import os
import math
import time
import torch
from tqdm import tqdm
import wandb
from Dataset import Combplex_Dataset
from models.model import Model
from util.losses import Loss
from util.util import *

model_type = sys.argv[1]
start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model():
    # load config file
    config = get_config(model_type)
    scale_MIBI = True if ((config['model_type'] == 'Value reconstruction network') and (config['imaging_platform'] == 'MIBI')) else False 

    # wandb initializations
    model_description = 'binary' if model_type == 'Decompression masking network' else 'values'
    model_name = f'{config["model_name"]}_{model_description}'
    if model_name.startswith('delete'):
        wandb.init(project='Combplex',name=model_name, entity=config['W&B_entity'], config=config)
    else:    
        wandb.init(project='Combplex',name=model_name, id=model_name, entity=config['W&B_entity'], config=config, resume="allow")
    wandb.config
    outputs_dir = os.path.join('pretrained_models', wandb.run.name)
    os.makedirs(outputs_dir, exist_ok=True)

    # set seed
    set_seed(config)

    # get variables from compression matrices form
    _, singles_names, multis_names, _, _, _ = read_compression_matrix_form(config['compression_matrix_form_path'])

    # create datasets
    train_dataset = Combplex_Dataset(config['training_dataset_path'], False, config, device)
    validation_dataset = Combplex_Dataset(config['validation_dataset_path'], True, config, device)

    # define model, loss function, optimizer and scheduler
    model = Model(in_channels=len(multis_names), n_classes=len(singles_names), model_features=config['model_features'], dropout_rate=config['dropout_rate'])
    criterion = Loss(config)
    loss_name = criterion.get_loss_name()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # resume exists checkpoint, if needed
    resume_path = f"pretrained_models/{model_name}/{model_name}-latest.pt"
    if os.path.exists(resume_path) and config['model_name'] != 'delete':
        checkpoint = torch.load(resume_path,  map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        current_batch = checkpoint['step']
        optimal_validation_loss = checkpoint['optimal_validation_loss']
        print('loading ckpt from {}'.format(resume_path))
    else:
        optimal_validation_loss = math.inf
        current_batch = 0
        
    # training process
    with tqdm(range(current_batch, config['num_batches'] + 1)) as tbatch:
        for batch in tbatch:
            current_batch += 1
            # save checkpoint and finish if exceeds max running time
            current_running_time = (time.time() - start_time) / 3600
            if  current_running_time > config['running_time']:
                save_checkpoint(model, optimizer, current_batch, optimal_validation_loss, outputs_dir, model_name,
                                config['model_features'], save_name='latest')
                break
            # load batch for training
            inputs = train_dataset[0]
            # infer current model and calculate loss
            optimizer.zero_grad()
            outputs = model(inputs)
            # scale in case MIBI value reconstruction
            if scale_MIBI:
                outputs['preds'] = outputs['preds'] * (2 ** 8 - 1)
            losses = criterion(outputs, inputs)
            log_data = {**losses, 'batch': current_batch}

            # validation set
            if current_batch % config['validation_frequency'] == 0:
                with torch.no_grad():
                    validation_inputs = validation_dataset[0]
                    for key in validation_inputs:
                        validation_inputs[key] = validation_inputs[key].to(device)
                    validation_outputs = model(validation_inputs)
                    if scale_MIBI:
                        validation_outputs['preds'] = validation_outputs['preds'] * (2 ** 8 - 1)
                    validation_loss = criterion(validation_outputs, validation_inputs)
                    log_data[f'validation_loss_{loss_name}'] = validation_loss[f'loss_{loss_name}']
                if (validation_loss[f'loss_{loss_name}'] < optimal_validation_loss) and current_batch > 10000:
                    save_checkpoint(model, optimizer, current_batch, optimal_validation_loss, outputs_dir, model_name,
                                    config['model_features'], save_name=f'optimal_{current_batch}')
                    optimal_validation_loss = validation_loss[f'loss_{loss_name}']

            # calculate F1 scores for evaluation
            if model_type == 'Decompression masking network':
                log_data = F1_scores(inputs['GT_singles'], outputs['preds'], singles_names, log_data, is_validation=False)
                if current_batch % config['validation_frequency'] == 0:
                    log_data = F1_scores(validation_inputs['GT_singles'], validation_outputs['preds'], singles_names, log_data, is_validation=True)
                    
            # update tqdm details
            tbatch.set_description(f"Batch {log_data['batch']}")
            tbatch.set_postfix(loss=log_data[f"loss_{loss_name}"].item())

            # save checkpoint
            if current_batch % config['save_checkpoint_frequency'] == 0:
                save_checkpoint(model, optimizer, current_batch, optimal_validation_loss, outputs_dir, model_name,
                                config['model_features'], save_name='latest')
                save_checkpoint(model, optimizer, current_batch, optimal_validation_loss, outputs_dir, model_name,
                                config['model_features'], save_name=f'{current_batch}')

            wandb.log(log_data)
            losses[f'loss_{loss_name}'].backward()
            optimizer.step()


if __name__ == '__main__':
    train_model()
