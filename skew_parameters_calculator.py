import os
import torch
import numpy as np
import pandas as pd
from torchvision import transforms as T
import torch.nn.functional as F
import skimage.io as io 
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.colors import ListedColormap
from util.util import read_compression_matrix_form

# user inputs
validation_set_path = '../../datasets/your_validation_dataset' # path to the validation set with the real multis
compression_matrix_form_path = 'compression_matrices/your_compression_matrix_form' # path to compression matrix form
center_crop = 2048 # [None or int]
print_analysis = False # [True | False] if True, figures showing the analyses of each multi image will be saved to "skew_parameters_analysis" directory

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
compression_matrix_form_filename = os.path.splitext(os.path.basename(compression_matrix_form_path))[0]

# read compression matrix form
A_df, _, multis_names, train_compression_matrix, _, GT_filenames = read_compression_matrix_form(compression_matrix_form_path)

# set center crop transform
center_crop = T.CenterCrop(center_crop) if center_crop != None else T.Compose([])

# Create empty dataframes for mu and std values
mu_df = pd.DataFrame()
std_df = pd.DataFrame()

def load_fov_data(fov_path, GT_filenames, compression_matrix, center_crop, multis_names, device):
    # stacks fov data as needed from its path
    fov_GT_singles, fov_sim_singles, fov_sim_multis, fov_real_multis = [], [], [], []
    simulation_file_names = list(compression_matrix.columns)
    simulation_mat = torch.from_numpy(compression_matrix.values).unsqueeze(-1).unsqueeze(-1).to(device)
    
    # stack simulation singles
    for file_name in simulation_file_names:
        fov_sim_singles.append(io.imread(os.path.join(fov_path, '{}.tif'.format(file_name))))
    fov_sim_singles = center_crop(torch.from_numpy(np.stack(fov_sim_singles).astype(np.float32))).unsqueeze(0).to(device)
    
    # create fov_multis according to the simulation matrix
    fov_sim_multis = F.conv2d(fov_sim_singles, simulation_mat).squeeze(0).cpu()
    
    # stack fov real multis
    for file_name in multis_names:
        fov_real_multis.append(io.imread(os.path.join(fov_path, '{}.tif'.format(file_name))))
    fov_real_multis = center_crop(torch.from_numpy(np.stack(fov_real_multis).astype(np.float32)))

    # stack GT singles
    for file_name in GT_filenames:
        fov_GT_singles.append(io.imread(os.path.join(fov_path, '{}.tif'.format(file_name))))
    fov_GT_singles = center_crop(torch.from_numpy(np.stack(fov_GT_singles).astype(np.float32)))

    return fov_GT_singles, fov_sim_multis, fov_real_multis



def multi_image_noise_analysis(sim_multi_img, real_multi_img, fov, real_multi, num_proteins_map):
    # Calculate noise values
    noise = (sim_multi_img - real_multi_img) / real_multi_img

    # returning non zero values
    noise = noise[real_multi_img > 0]
    num_proteins_map = num_proteins_map[real_multi_img > 0].int()
    real_multi_img = real_multi_img[real_multi_img > 0]

    unique_values = np.unique(num_proteins_map)
    for i in unique_values:
        # get the data points relevant for specific num of proteins
        data = noise[num_proteins_map == i]

        # Fit a normal distribution to the data 
        mu, std = stats.norm.fit(data)

        # Add mu and std values to dataframes
        mu_df.loc[fov, f'{real_multi}_{i}'] = mu
        std_df.loc[fov, f'{real_multi}_{i}'] = std

    if print_analysis:
        # calculate median intensity value
        median_real_multi = np.median(real_multi_img)

        # Create subplots for scatter plot and histogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Scatter plot
        # Generate a colormap with a unique color for each unique value in num_proteins_map
        num_colors = len(unique_values)
        color_map = ListedColormap(plt.cm.get_cmap('jet', num_colors)(range(num_colors)))

        ax1.scatter(real_multi_img, noise, c=num_proteins_map, cmap=color_map, alpha=0.5)
        ax1.set_xlabel('Real Multi Image Values')
        ax1.set_ylabel('Noise (simulation - real)')
        ax1.set_title('Scatter Plot: Real Multi Image vs Noise')
        y_range = np.max([abs(noise.max()), abs(noise.min())])
        ax1.set_ylim(-y_range, y_range)
        ax1.grid(True)

        # Histogram
        # Generate colors for each unique value in num_proteins_map
        colors = plt.get_cmap('jet', num_colors)(range(num_colors))

        for i in unique_values:
            # get data, mu and std values for plotting
            data = noise[num_proteins_map == i]
            mu = mu_df.loc[fov, f'{real_multi}_{i}']
            std = std_df.loc[fov, f'{real_multi}_{i}']

            # Plot the histogram with fitted normal distribution
            ax2.hist(data, bins='auto', density=True, alpha=0.7, color=colors[i - 1], label=f'{i}')
            x_range = np.max([abs(mu - 5 * std), abs(mu + 5 * std)])
            x = np.linspace(-x_range, x_range, 1000)
            p = stats.norm.pdf(x, mu, std)
            ax2.plot(x, p, color=colors[i - 1], linewidth=2)

        # Create a legend for the color-coded histograms
        legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=f'{i}') for i, color in zip(unique_values, colors)]
        ax2.legend(handles=legend_handles, title='num_proteins_map')
        
        # Add title to the figure
        fig.suptitle(f'FOV {fov}, Multi {real_multi}: (median={int(median_real_multi)})', fontsize=14, fontweight='bold')

        # Adjust spacing between subplots
        plt.xlim(-x_range, x_range)
        plt.tight_layout()

        # Create the output directory for the real_multi
        output_dir = os.path.join('skew_parameters_analysis', compression_matrix_form_filename, real_multi)
        os.makedirs(output_dir, exist_ok=True)

        # Save the figure in the output directory
        filename = f'{fov}_{real_multi}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()


for fov in os.listdir(validation_set_path):
    fov_path = os.path.join(validation_set_path, fov)
    if os.path.isdir(fov_path):
        fov_GT_singles, fov_sim_multis, fov_real_multis = load_fov_data(fov_path, GT_filenames, train_compression_matrix, center_crop, multis_names, device)
        for j, real_multi in enumerate(multis_names):
            indices = [i for i, value in enumerate(A_df.loc[real_multi].tolist()) if value]
            num_proteins_map = (fov_GT_singles[indices] > 0).float().sum(dim=0)
            multi_image_noise_analysis(fov_sim_multis[j], fov_real_multis[j], fov, real_multi, num_proteins_map)

# Save skew parameters df to "compression_matrices" folder
means = mu_df.mean()
skew_parameters_df = pd.DataFrame({'mean': means}).T
skew_parameters_df.to_csv(os.path.join('compression_matrices', f'{compression_matrix_form_filename}_skew_parameters.csv'))