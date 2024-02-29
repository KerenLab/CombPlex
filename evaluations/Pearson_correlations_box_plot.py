import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
from scipy import ndimage
import matplotlib.pyplot as plt
import os
import pandas as pd
import skimage.io
import sys
sys.path.append('.')
from util.util import read_compression_matrix_form
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rc('font', family='arial')
mpl.rc('font', serif='arial')

'''
This script creats Pearson correlation boxplot figure.
'''

# This function calculates Pearson correlation for corresponding pixels in the intersection of GT and prediction
def Pearson_intersection(GT_protein_img, recon_protein_image, window_size):
    # Convert the numpy arrays to binary masks
    GT_protein_mask = GT_protein_img > 0
    recon_protein_mask = recon_protein_image > 0
    # Take the intersection of the two masks and keep the values of each of them
    intersection_mask = GT_protein_mask & recon_protein_mask
    # calculating Pearson correlation
    if np.all(intersection_mask==False):
        # In case of no pixels in the intersection
        correlation = 1
    else:
        # Neglecting pixels that aren't in the intersection
        GT_protein_img = ndimage.uniform_filter(GT_protein_img, size=window_size) * intersection_mask
        recon_protein_image = ndimage.uniform_filter(recon_protein_image, size=window_size) * intersection_mask

        # Flatten the images into 1D arrays
        GT_protein_flat = GT_protein_img[intersection_mask > 0]
        recon_protein_flat = recon_protein_image[intersection_mask > 0]

        # Calculate the Pearson correlation coefficient and p-value
        correlation, _ = pearsonr(GT_protein_flat, recon_protein_flat)
        
    return correlation

# This function calculates Pearson correlation for all corresponding pixels in the GT and prediction images
def Pearson_all_pixels(GT_protein_img, recon_protein_image, window_size):
    # Avg filtering the image given a window size
    GT_protein_img = ndimage.uniform_filter(GT_protein_img, size=window_size)
    recon_protein_image = ndimage.uniform_filter(recon_protein_image, size=window_size)

    # Flatten the images into 1D arrays
    GT_protein_flat = GT_protein_img.flatten()
    recon_protein_flat = recon_protein_image.flatten()

    # Calculate the Spearman correlation coefficient and p-value
    correlation, _ = pearsonr(GT_protein_flat, recon_protein_flat)
        
    return correlation

# This function creats Pearson correlation figure
def Pearson_correlation_box_plot(run_name, compression_matrices_csv_path, imaging_platform, window_size, all_pixels):
    _, singles_names, _, _, _, _ = read_compression_matrix_form(compression_matrices_csv_path)
    dir_path = f'results/{run_name}'
    fov_dirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    correlations_df = pd.DataFrame()
    # Iterate over all single-protein images and FOVs
    for fov in fov_dirs:
        for protein in singles_names:
            GT_protein_img_path = os.path.join(dir_path, fov, f"{'AP_' if imaging_platform == 'MIBI' else ''}{protein}.tif")
            recon_protein_img_path = [os.path.join(dir_path, fov, f) for f in os.listdir(os.path.join(dir_path, fov)) if f.startswith(f"recon_{protein}_")][0]
            GT_protein_img = skimage.io.imread(GT_protein_img_path)
            recon_protein_image = skimage.io.imread(recon_protein_img_path)
            # calculate Pearson correlation for a given protein in FOV
            if all_pixels:
                correlation = Pearson_all_pixels(GT_protein_img, recon_protein_image, window_size)
            else:
                correlation = Pearson_intersection(GT_protein_img, recon_protein_image, window_size)
            correlations_df.loc[fov, protein] = correlation
    # save .csv file with all correlations
    correlations_df.to_csv(os.path.join(dir_path, f'Pearson correlations - {run_name} - window_size={window_size} - all_pixels={all_pixels}.csv'))
    # reorder df columns according to median
    channel_order = correlations_df.median().sort_values(ascending=False).index
    correlations_df = correlations_df[channel_order]
    # initialize style and create figure
    num_channels = len(singles_names)
    plt.figure(figsize=(num_channels * 2 + 2, 30), dpi=300)
    sns.set(style="white")
    PROPS = {
        'boxprops': {'facecolor': 'grey', 'edgecolor': 'black'},
        'medianprops': {'color': 'black'},
        'whiskerprops': {'color': 'black'},
        'capprops': {'color': 'black'}
    }
    ax = sns.boxplot(data=correlations_df, color='w', width=0.6, showfliers=False, linewidth=0.5, **PROPS)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    medians = correlations_df.median().round(2)
    for i, median in enumerate(medians):
        plt.text(i, median + 0.01, f"{median:.2f}", horizontalalignment='center', fontsize=24, color='black')

    plt.xticks(range(0, num_channels), correlations_df.columns, fontsize=30, rotation=90)
    plt.yticks(fontsize=30)

    plt.xlabel("\nProteins", fontsize=24)
    plt.ylabel("Pearson correlation\n", fontsize=24)
    plt.ylim(-1, 1)
    plt.title("Pearson correlations by protein\n", fontsize=30)
    
    plt.savefig(os.path.join(dir_path, f'Box Plot - Pearson correlations - {run_name} - window_size={window_size} - all_pixels={all_pixels}.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
