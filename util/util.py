from torch.optim import lr_scheduler
import torch
import os
import pandas as pd

def read_compression_matrix_form(compression_matrix_form_path):
    # Read the Excel or csv file
    settings_form_df = pd.read_csv(compression_matrix_form_path, index_col=0, header=None)

    # Identify row indices and their separators
    seperators = ['Reconstruction matrix A (note: protein and channel names can be represented by nicknames instead of filenames):',
                  'Training compression matrix (note: the columns and rows headers must be filenames. Also, keep the same order)',
                  'Test compression matrix (note: the columns and rows headers be filenames. Also, keep the same order)',
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

def get_scheduler(optimizer, cfg):
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)
    return scheduler


def get_optimizer(cfg, params):
    optimizer = torch.optim.Adam(params, lr=cfg['lr'])
    return optimizer


def save_checkpoint(model, optimizer, current_epoch, optimal_validation_loss, outputs_dir, model_name, save_name='latest'):
    # Create a dictionary containing the current state of the model, optimizer, and other relevant values
    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": current_epoch,
        "optimal_validation_loss": optimal_validation_loss,
    }
    
    # Create the path for the checkpoint file using the provided logging directory and save_name
    checkpoint_path = os.path.join(outputs_dir, "{}-{}.pt".format(model_name, save_name))
    
    # Save the checkpoint dictionary to the specified file path
    torch.save(checkpoint_state, checkpoint_path)
