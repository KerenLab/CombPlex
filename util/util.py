import torch
import os
import yaml
import numpy as np
import random
import pandas as pd

# This function creates relevant config acccording to model type and the user's config
def get_config(model_type):
    # read user's config
    with open("config/config_train.yaml", "r") as f:
        user_config = yaml.safe_load(f)

    # initialize new_config with model_type
    config = {'model_type': model_type}

    # Exclude keys "Decompression masking network" and "Value reconstruction network" and add the others to new_config
    excluded_keys = ["Decompression masking network", "Value reconstruction network"]
    config.update({key: value for key, value in user_config.items() if key not in excluded_keys})

    # add relevant keys according to model_type
    config.update(user_config[model_type])
    
    return config

# This function sets a seed number according to user's input in config
def set_seed(config):
    seed = np.random.randint(2 ** 32 - 1, dtype=np.int64) if config['seed'] == None else config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f'running with seed: {seed}.')

#This function reads all the needed information from the user's compression matrix form
def read_compression_matrix_form(compression_matrix_form_path):
    # Read the Excel or csv file
    settings_form_df = pd.read_csv(compression_matrix_form_path, index_col=0, header=None)

    # Identify row indices and their separators
    seperators = ['Reconstruction matrix A (note: protein and channel names can be represented by nicknames instead of filenames):',
                  'Training compression matrix (note: the columns and rows headers must be filenames. Also, keep the same order)',
                  'Test compression matrix (note: the columns and rows headers must be filenames. Also, keep the same order)',
                  'GT filename for each protein (note: keep the same order):']
    iloc_values = [settings_form_df.index.get_loc(seperator) for seperator in seperators]
    iloc_values = [x + 1 for x in iloc_values]

    # Extract data for reconstruction matrix A
    A_df = pd.read_csv(compression_matrix_form_path, skiprows=iloc_values[0], nrows=(iloc_values[1] - iloc_values[0] - 2), index_col=0).dropna(axis=1).astype('float32')
    singles_nicknames = A_df.columns.tolist()
    multis_nicknames = A_df.index.tolist()
    
    # Extract data for training and test compression matrices
    train_compression_matrix = pd.read_csv(compression_matrix_form_path, skiprows=iloc_values[1], nrows=(iloc_values[2] - iloc_values[1] - 2), index_col=0).dropna(axis=1).astype('float32')
    test_compression_matrix = pd.read_csv(compression_matrix_form_path, skiprows=iloc_values[2], nrows=(iloc_values[3] - iloc_values[2] - 2), index_col=0).dropna(axis=1).astype('float32')
    
    # get list of the GT filenames
    GT_filenames = settings_form_df.loc['GT filenames'].dropna().tolist()

    return A_df, singles_nicknames, multis_nicknames, train_compression_matrix, test_compression_matrix, GT_filenames

# Pytorch version of F1 score
def F1_scores(GT_singles, preds, singles_names, log_data, is_validation):
    # calculate F1 score for two  Pytorch tensors
    epsilon = 1e-7
    b, c = GT_singles.shape[:2]
    target = (GT_singles > 0).float().reshape(b, c, -1)
    pred = (preds.detach() > 0.5).float().reshape(b, c, -1)
    for i, channel in enumerate(singles_names):
        target_i = target[0, i]
        pred_i = pred[0, i]
        f1 = (2 * (target_i * pred_i).sum() + epsilon) / (target_i.sum() + pred_i.sum() + epsilon)
        log_data[f'{channel}{"_validation" if is_validation else ""}_f1'] = f1
    return log_data

# This function saves a checkpoint of the model with all information needed
def save_checkpoint(model, optimizer, current_epoch, optimal_validation_loss, outputs_dir, model_name,
                    model_features, save_name='latest'):
    # Create a dictionary containing the current state of the model, optimizer, and other relevant values
    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": current_epoch,
        "optimal_validation_loss": optimal_validation_loss,
        "model_features": model_features
    }
    
    # Create the path for the checkpoint file using the provided logging directory and save_name
    checkpoint_path = os.path.join(outputs_dir, f"{model_name}-{save_name}.pt")
    
    # Save the checkpoint dictionary to the specified file path
    torch.save(checkpoint_state, checkpoint_path)
