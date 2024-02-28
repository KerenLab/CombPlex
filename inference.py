import os
import torch
import numpy as np
from models.model import Model
import yaml
import pandas as pd
from torchvision import transforms as T
import torch.nn.functional as F
from scipy import ndimage
import skimage.io as io 
import shutil
from evaluations.F1_box_plot import F1_box_plot
from evaluations.Pearson_correlations_box_plot import Pearson_correlation_box_plot
from sklearn.metrics import f1_score
from util.util import read_compression_matrix_form

# Load inference configuration from YAML file
with open("config/config_inference.yaml", "r") as f:
    config = yaml.safe_load(f)

# Define the output folder for the results based on the configuration
outputs_folder = f"results/{config['results_folder_name']}"

# Determine the device for computation (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data from the compression matrix form
A_df, singles_names, multis_names, _, test_compression_matrix, GT_filenames = read_compression_matrix_form(config['compression_matrix_form_path'])
num_singles = len(singles_names)
num_multis = len(multis_names)

# Define a transformation for center cropping (if specified in the config)
center_crop_transform = T.CenterCrop(config['center_crop']) if config['center_crop'] else T.Compose([])

# create list with all decompression masking models
binary_models_list = []
for filename in os.listdir(config['masking_models_folder']):
    if filename.endswith('.pt'):
        model_path = os.path.join(config['masking_models_folder'], filename)
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        model_features = checkpoint['model_features'] if 'model_features' in checkpoint else [64, 128, 128, 256, 256]
        model = Model(in_channels=num_multis, n_classes=num_singles, model_features=model_features).to(device).eval()    
        model.load_state_dict(checkpoint["model"])
        binary_models_list.append(model)

# create list with all decompression masking models
values_models_list = []
for filename in os.listdir(config['values_reconstruction_models_folder']):
    if filename.endswith('.pt'):
        model_path = os.path.join(config['values_reconstruction_models_folder'], filename)
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        model_features = checkpoint['model_features'] if 'model_features' in checkpoint else [64, 128, 128, 256, 256]
        model = Model(in_channels=num_multis, n_classes=num_singles, model_features=model_features).to(device).eval()    
        model.load_state_dict(checkpoint["model"])
        values_models_list.append(model)

# Create an empty DataFrame to store F1 scores
if config['evaluate_with_GT']:
    f1_df = pd.DataFrame(columns=singles_names, dtype=float)
    dilated_f1_df = pd.DataFrame(columns=singles_names, dtype=float)

# Load FOV data and process it
def load_fov_data(fov_path):
    fov_GT_singles, fov_sim_singles, fov_multis = [], [], []
    simulation_file_names = list(test_compression_matrix.columns)
    simulation_mat = torch.from_numpy(test_compression_matrix.values).unsqueeze(-1).unsqueeze(-1).to(device)
    
    # stack simulation singles
    for file_name in simulation_file_names:
        fov_sim_singles.append(io.imread(os.path.join(fov_path, '{}.tif'.format(file_name))))
    fov_sim_singles = center_crop_transform(torch.from_numpy(np.stack(fov_sim_singles).astype(np.float32))).unsqueeze(0).to(device)

    # create fov_multis according to the simulation matrix
    fov_multis = center_crop_transform(F.conv2d(fov_sim_singles, simulation_mat))
    
    # stack GT singles
    if config['evaluate_with_GT']:
        for file_name in GT_filenames:
            fov_GT_singles.append(io.imread(os.path.join(fov_path, '{}.tif'.format(file_name))))
        fov_GT_singles = center_crop_transform(torch.from_numpy(np.stack(fov_GT_singles).astype(np.float32))).unsqueeze(0)
    else:
        fov_GT_singles = None

    return fov_GT_singles, fov_multis

# Perform binary prediction on the FOV
def fov_prediction(fov_multis):

    # Generate binary predictions for each model
    inputs = {'multis': fov_multis.to(device)}

    with torch.no_grad():
        # perform model ensembling for the decompression masking models
        binary_mask = (binary_models_list[0](inputs)['preds'] > 0.5).detach().float()
        for model in binary_models_list[1:]:
            binary_mask = binary_mask + (model(inputs)['preds'] > 0.5).detach().float()
        binary_mask = ((binary_mask / len(binary_models_list)) >= config['agreement']).float()

        # perform model ensembling for the values reconstruction models
        if config['imaging_platform'] == 'CODEX':
            inputs = {'multis': (fov_multis / (2 ** 16 -1)).to(device)}
        values_prediction = values_models_list[0](inputs)['preds'].detach()
        for model in values_models_list[1:]:
            values_prediction = values_prediction + model(inputs)['preds'].detach()
        if config['imaging_platform'] == 'CODEX':
            values_prediction = (2 ** 16 - 1) * (values_prediction / len(values_models_list))
        else:
            values_prediction = (2 ** 8 - 1) * (values_prediction / len(values_models_list))

    # create msks according to relevant multis of each protein
    compression_matrix = torch.from_numpy(A_df.values).to(device)
    multis_masks = torch.ones_like(binary_mask)
    for i in range(num_multis):
        for j in range(num_singles):
            if compression_matrix[i, j] != 0:
                multis_masks[:, j] = multis_masks[:, j] * (fov_multis[:, i] > 0)
    
    # create outputs after intersection
    fov_binary_pred = (binary_mask * multis_masks).detach().cpu()
    fov_values_pred = (values_prediction * multis_masks).detach().cpu()
    fov_combplex_pred = (fov_binary_pred * fov_values_pred).detach()

    return fov_binary_pred, fov_values_pred, fov_combplex_pred

# Save FOV images and results
def save_fov_images(fov, fov_GT_singles, fov_multis, fov_binary_pred, fov_values_pred, fov_combplex_pred, fov_path):
    # create fov output dir
    fov_dir_path = os.path.join(os.path.join(outputs_folder, fov))
    if not os.path.exists(fov_dir_path):
        os.makedirs(fov_dir_path)
    
    if config['evaluate_with_GT']:
        # save GT images
        for i, name in enumerate(singles_names):
            io.imsave(os.path.join(fov_dir_path, '{}.tif'.format(name)), fov_GT_singles[:, i].numpy(), check_contrast=False)
        # save DAPI or dsDNA if possible
        for filename in ['dsDNA.tif', 'DAPI.tif']:
            src_path = os.path.join(fov_path, filename)
            dest_path = os.path.join(fov_dir_path, filename)
            if os.path.exists(src_path):
                shutil.copy(src_path, dest_path)

    # save multi images
    for i, name in enumerate(multis_names):
        io.imsave(os.path.join(fov_dir_path, '{}.tif'.format(name)), fov_multis[:, i].cpu().numpy(), check_contrast=False)

    # # save binary prediction
    # for i, name in enumerate(singles_names):
    #     io.imsave(os.path.join(fov_dir_path, 'binary_{}.tif'.format(name)), fov_binary_pred[:, i].cpu().numpy(), check_contrast=False)

    # # save values prediction
    # for i, name in enumerate(singles_names):
    #     io.imsave(os.path.join(fov_dir_path, 'values_{}.tif'.format(name)), fov_values_pred[:, i].cpu().numpy(), check_contrast=False)

    # save combplex prediction
    for i, name in enumerate(singles_names):
        io.imsave(os.path.join(fov_dir_path, 'recon_{}.tif'.format(name)), fov_combplex_pred[:, i].cpu().numpy(), check_contrast=False)

# create and save dilated version for GT images, used in case of MIBI data
def smooth_GT_singles(fov, fov_GT_singles, fov_multis):
    fov_dir_path = os.path.join(os.path.join(outputs_folder, fov))

    # create smooth version of GT images
    fov_GT_smooth_singles = torch.nn.AvgPool2d(kernel_size=5, stride=1, padding=2)(fov_GT_singles)

    # intersect it with relevant multis
    A = A_df.values
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != 0:
                fov_GT_smooth_singles[:, j] = fov_GT_smooth_singles[:, j] * (fov_multis[:, i].cpu() > 0)
    for i, name in enumerate(singles_names):
        io.imsave(os.path.join(fov_dir_path, 'AP_{}.tif'.format(name)), fov_GT_smooth_singles[:, i].numpy(), check_contrast=False)

    return fov_GT_smooth_singles

# Analyze FOV results and calculate F1 scores
def analyze_fov(fov, fov_GT_singles, fov_combplex_pred, f1_df, dilated_f1_df):
    fov_dir_path = os.path.join(outputs_folder, fov)
        
    # calculate F1 and dilated F1 scores
    for protein_idx in range(fov_GT_singles.size(1)):
        GT_single = fov_GT_singles[:, protein_idx, :, :].cpu()
        recon_single = fov_combplex_pred[:, protein_idx, :, :].cpu()
        singles_names = f1_df.columns.tolist()

        # flatten GT and recon img
        GT_single_np = (GT_single.numpy().flatten() > 0).astype(float)
        recon_single_np = (recon_single.numpy().flatten() > 0).astype(float)
        # assign F1 score in case of empty GT img
        if GT_single_np.sum() == 0:
            if recon_single_np.sum() == 0:
                f1_df.at[fov, f'{singles_names[protein_idx]}'] = 1
                dilated_f1_df.at[fov, f'{singles_names[protein_idx]}'] = 1
            else:
                f1_df.at[fov, f'{singles_names[protein_idx]}'] = 0
                dilated_f1_df.at[fov, f'{singles_names[protein_idx]}'] = 0
        else:
            # calculate F1 score
            f1_score_result = f1_score(GT_single_np, recon_single_np)
            f1_df.at[fov, f'{singles_names[protein_idx]}'] = f1_score_result

            tp_mask = ((GT_single > 0) * (recon_single > 0)).float()
            fp_mask = ((GT_single == 0) * (recon_single > 0)).int()
            fn_mask = ((GT_single > 0) * (recon_single == 0)).int()
            dilated_tp_mask = (ndimage.uniform_filter(tp_mask, size=5) > 0).astype(int)
            dilated_pred = (recon_single > 0).int() - fp_mask * dilated_tp_mask + fn_mask * dilated_tp_mask
            pred_dilated = dilated_pred.flatten()

            dilated_F1_score_result = f1_score(GT_single_np, pred_dilated)
            dilated_f1_df.at[fov, f'{singles_names[protein_idx]}'] = dilated_F1_score_result

        # rename reconstruction file with its F1 score
        os.rename(os.path.join(fov_dir_path, 'recon_{}.tif'.format(singles_names[protein_idx])), 
                  os.path.join(fov_dir_path, 'recon_{}_{:.3f}_{:.3f}.tif'.format(singles_names[protein_idx], dilated_F1_score_result, f1_score_result)))


# Iterate over the test dataset and perform inference
for fov in os.listdir(config['test_dataset_path']):
    fov_path = os.path.join(config['test_dataset_path'], fov)
    if os.path.isdir(fov_path):
        print(f'FOV {fov}...')
        fov_GT_singles, fov_multis = load_fov_data(fov_path) 
        fov_binary_pred, fov_values_pred, fov_combplex_pred = fov_prediction(fov_multis)
        save_fov_images(fov, fov_GT_singles, fov_multis, fov_binary_pred, fov_values_pred, fov_combplex_pred, fov_path)
        if config['evaluate_with_GT']:
            if config['imaging_platform'] == 'MIBI':
                fov_GT_singles = smooth_GT_singles(fov, fov_GT_singles, fov_multis)
            analyze_fov(fov, fov_GT_singles, fov_combplex_pred, f1_df, dilated_f1_df)   
        print(f'Inference process for FOV {fov} is done!\n')

# print F1 scores and Pearson correlation graphs 
if config['evaluate_with_GT']:
    # Save the calculated F1 scores to a CSV file
    f1_df.to_csv(os.path.join(outputs_folder, 'F1 scores results - {}.csv').format(config['results_folder_name']))
    dilated_f1_df.to_csv(os.path.join(outputs_folder, 'Dilated F1 scores results - {}.csv').format(config['results_folder_name']))

    # create F1 box plots
    print('Creating F1 scores box plots...')
    F1_box_plot(config['results_folder_name'], dilated=False)
    F1_box_plot(config['results_folder_name'], dilated=True)
    print('Done!\n')

    # create pearson correlations box plot
    print('Creating Pearson correlation box plots...')
    Pearson_correlation_box_plot(config['results_folder_name'], config['compression_matrix_form_path'], config['imaging_platform'], window_size= 5, all_pixels=True)
    Pearson_correlation_box_plot(config['results_folder_name'], config['compression_matrix_form_path'], config['imaging_platform'], window_size= 1, all_pixels=True)
    print('Done!\n')

print('Inference process is done! :)\n')