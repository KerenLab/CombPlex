import os
import torch
import numpy as np
from models.model import Model
import yaml
import pandas as pd
from torchvision import transforms as T
import torch.nn.functional as F
import skimage.io as io 
from sklearn.metrics import f1_score
from util.util import read_compression_matrix_form

# Load inference configuration from YAML file
with open("config/config_inference.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Define the output folder for the results based on the configuration
outputs_folder = 'results/{}'.format(cfg['results_folder_name'])

# Determine the device for computation (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data from the compression matrix form
A_df, singles_names, multis_names, _, test_compression_matrix, GT_filenames = read_compression_matrix_form(cfg['compression_matrix_form_path'])

# Define a transformation for center cropping (if specified in the config)
center_crop_transform = T.Compose([T.CenterCrop(cfg['center_crop'])]) if cfg['center_crop'] else T.Compose([])

# Create a list of paths to trained model checkpoints
checkpoint_paths = ['pretrained_models/{}.pt'.format(model_name) for model_name in cfg['model_names']]

# Initialize an empty list to store trained model instances
models_list = []
channels_out = len(singles_names)
channels_in = len(multis_names)

# Load trained models into the list
for checkpoint_path in checkpoint_paths:
    model = Model(in_channels=channels_in, n_classes=channels_out, filters=cfg['model_features']).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["model"])
    models_list.append(model)

# Create an empty DataFrame to store F1 scores
f1_df = pd.DataFrame(columns=singles_names, dtype=float)

# Load FOV data and process it
def load_fov_data(fov_path, GT_filenames, compression_matrix, center_crop_transform, device):
    fov_GT_singles, fov_sim_singles, fov_multis = [], [], []
    simulation_file_names = list(compression_matrix.columns)
    simulation_mat = torch.from_numpy(compression_matrix.values).unsqueeze(-1).unsqueeze(-1).to(device)
    
    # stack simulation singles
    for file_name in simulation_file_names:
        fov_sim_singles.append(io.imread(os.path.join(fov_path, '{}.tif'.format(file_name))))
    fov_sim_singles = center_crop_transform(torch.from_numpy(np.stack(fov_sim_singles).astype(np.float32))).unsqueeze(0).to(device)

    # create fov_multis according to the simulation matrix
    fov_multis = center_crop_transform(F.conv2d(fov_sim_singles, simulation_mat))
    
    # stack GT singles
    for file_name in GT_filenames:
        fov_GT_singles.append(io.imread(os.path.join(fov_path, '{}.tif'.format(file_name))))
    fov_GT_singles = center_crop_transform(torch.from_numpy(np.stack(fov_GT_singles).astype(np.float32))).unsqueeze(0)

    return fov_GT_singles, fov_multis

# Perform binary prediction on the FOV
def fov_binary_prediction(models_list, fov_multis, cfg, device):
    # Set the threshold for ensembling based on agreement percentage
    threshold = round(cfg['agreement'] * len(models_list))
    if threshold == 0:
        threshold = 1
    
    # Generate binary predictions for each model
    models_outputs = []
    inputs = {'multis': fov_multis.to(device)}
    for model in models_list:
        models_outputs.append((model(inputs)['preds'] > 0.5).float())
    
    # Combine model outputs and perform ensembling
    models_outputs = torch.cat(models_outputs, dim=0)
    fov_binary_preds = (torch.sum(models_outputs, 0) >= threshold).float().unsqueeze(0)
    
    # Return the binary predictions after ensembling
    return fov_binary_preds


def create_skew_tensor(fov_binary_preds, A, skew_parameters_csv_path, multi_names):
    skew_tensor = []
    fov_binary_preds = fov_binary_preds[0].cpu().numpy()
    for j in range(A.shape[0]):
        indices = [i for i, value in enumerate(A[j]) if value != 0]
        num_proteins_multi= (fov_binary_preds[indices] > 0).astype(int).sum(axis=0)
        skew_tensor.append(num_proteins_multi)
    skew_tensor = np.stack(skew_tensor).astype(np.float32)
    skew_parameters_df = pd.read_csv(skew_parameters_csv_path, index_col=0)
    
    for i in range(A.shape[0]):
        num_proteins_unique_values = np.unique(num_proteins_multi[i])
        num_proteins_unique_values = num_proteins_unique_values[num_proteins_unique_values != 0]
        for val in num_proteins_unique_values:
            skew_tensor[i][skew_tensor[i] == val] = 1 / (1 + skew_parameters_df.loc['mean', f'{multi_names[i]}_{int(val)}'])
    return torch.from_numpy(skew_tensor).unsqueeze(0).to(device)

# Post-process FOV prediction to recover singles intensities
def fov_post_process_prediction(fov_multis, fov_binary_preds, A, lr, n_iter, device):
    # Get the dimensions of multis and binary predictions
    num_multis = fov_multis.shape[0]
    num_singles = fov_binary_preds.shape[0]
    
    # create skew tensor in case of CODEX exp
    if cfg['skew_parameters_csv_path']:
        skew_tensor = create_skew_tensor(fov_binary_preds, A.values, cfg['skew_parameters_csv_path'], multis_names)

    # Convert the compression matrix to a tensor suitable for convolution
    A = torch.from_numpy(A.values).unsqueeze(-1).unsqueeze(-1).to(device)

    # Intersect binary masks with relevant multis masks
    masks = fov_binary_preds
    for i in range(num_multis):
        for j in range(num_singles):
            if A[i, j] != 0:
                masks[j] = masks[j] * (fov_multis[:, i][0] > 0)

    # Perform optimization to recover singles intensities
    x = (torch.rand(1, num_singles, fov_binary_preds.shape[-2], fov_binary_preds.shape[-1]).to(device) * masks).to(device)
    x = torch.nn.Parameter(x, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=lr)
    loss_history = []
    for i in range(n_iter + 1):
        optimizer.zero_grad()
        multi_recon = F.conv2d(F.relu(x * masks), A)
        if cfg['skew_parameters_csv_path']:
            multi_recon = multi_recon * skew_tensor
        loss = F.mse_loss(multi_recon, fov_multis)
        loss_history.append(loss.detach().cpu())
        if len(loss_history) > 50:
            if loss_history[-50] - loss_history[-1] < 10 ** -1:
                print(f'Iteration {i}: loss={loss.item():.8f}')
                print('Early stopping condition achieved!')
                break
        loss.backward()
        optimizer.step()
        if i % (n_iter / 16) == 0:
            print(f'Iteration {i}: loss={loss.item():.4f}')

    # Generate reconstructed FOV with recovered singles intensities
    fov_reconstructed = torch.round(F.relu(x * masks)).detach()
    return fov_reconstructed


# Save FOV images and results
def save_fov_images(outputs_folder, fov, fov_GT_singles, fov_binary_preds, fov_reconstructed, fov_multis, singles_names, multis_names):
    # create fov dir
    fov_dir_path = os.path.join(os.path.join(outputs_folder, fov))
    if not os.path.exists(fov_dir_path):
        os.makedirs(fov_dir_path)
    # save GT images
    for i, name in enumerate(singles_names):
        io.imsave(os.path.join(fov_dir_path, '{}.tif'.format(name)), fov_GT_singles[:, i].numpy(), check_contrast=False)
    # save binary prediction images
    for i, name in enumerate(singles_names):
        io.imsave(os.path.join(fov_dir_path, 'pred_{}.tif'.format(name)), fov_binary_preds[:, i].cpu().numpy(), check_contrast=False)
    # save post process reconstruction images
    for i, name in enumerate(singles_names):
        io.imsave(os.path.join(fov_dir_path, 'recon_{}.tif'.format(name)), fov_reconstructed[:, i].cpu().numpy(), check_contrast=False)
    # save multi images
    for i, name in enumerate(multis_names):
        io.imsave(os.path.join(fov_dir_path, '{}.tif'.format(name)), fov_multis[:, i].cpu().numpy(), check_contrast=False)
    # save average pooled GT images in case of MIBI platform
    if cfg['data_type'] == 'MIBI':
        fov_GT_smooth_singles = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(fov_GT_singles)
        for i, name in enumerate(singles_names):
            io.imsave(os.path.join(fov_dir_path, 'AP_{}.tif'.format(name)), fov_GT_smooth_singles[:, i].numpy(), check_contrast=False)

# Analyze FOV results and calculate F1 scores
def analyze_fov(outputs_folder, fov, fov_GT_singles, fov_recon_singles, f1_df):
    fov_dir_path = os.path.join(outputs_folder, fov)
    # applying average pooling in case of MIBI data
    if cfg['data_type'] == 'MIBI':
        fov_GT_singles = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(fov_GT_singles)
        fov_recon_singles = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(fov_recon_singles)
    # iterate over all proteins in the FOV
    for protein_idx in range(fov_GT_singles.size(1)):
        GT_single = fov_GT_singles[:, protein_idx, :, :]
        recon_single = fov_recon_singles[:, protein_idx, :, :]
        singles_names = f1_df.columns.tolist()

        # calculate F1 score
        GT_single_np = (GT_single.cpu().numpy().flatten() > 0).astype(float)
        recon_single_np = (recon_single.cpu().numpy().flatten() > 0).astype(float)
        f1_score_result = f1_score(GT_single_np, recon_single_np)
        f1_df.at[fov, f'{singles_names[protein_idx]}'] = f1_score_result

        # rename reconstruction file with its F1 score
        os.rename(os.path.join(fov_dir_path, 'recon_{}.tif'.format(singles_names[protein_idx])), 
                  os.path.join(fov_dir_path, 'recon_{}_{:.3f}.tif'.format(singles_names[protein_idx], f1_score_result)))


# Iterate over the test dataset and perform inference
for fov in os.listdir(cfg['test_dataset_path']):
    fov_path = os.path.join(cfg['test_dataset_path'], fov)
    if os.path.isdir(fov_path):
        print(f'FOV {fov}:')
        fov_GT_singles, fov_multis = load_fov_data(fov_path, GT_filenames, test_compression_matrix, center_crop_transform, device)
        fov_binary_preds = fov_binary_prediction(models_list, fov_multis, cfg, device)
        fov_pp_preds = fov_post_process_prediction(fov_multis, fov_binary_preds, A_df, cfg['lr'], cfg['num_iterations'], device)
        save_fov_images(outputs_folder, fov, fov_GT_singles, fov_binary_preds, fov_pp_preds, fov_multis, singles_names, multis_names)
        analyze_fov(outputs_folder, fov, fov_GT_singles,fov_pp_preds, f1_df)
        print(f'Inference process for FOV {fov} is done!\n')

# Save the calculated F1 scores to a CSV file
f1_df.to_csv(os.path.join(outputs_folder, 'F1 scores results - {}.csv').format(cfg['results_folder_name']))




