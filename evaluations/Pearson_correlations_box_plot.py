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

# set your parameters
run_name = 'CODEX7to3_demo_results'
compression_matrices_csv_path = 'compression_matrices/CODEX_7to3_paper_exp.csv'

def calculate_blur_pearson_correlation(GT_protein_img, recon_protein_image, window_size=5):
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
        GT_protein_flat = GT_protein_img[GT_protein_img > 0]
        recon_protein_flat = recon_protein_image[recon_protein_image > 0]

        # Calculate the Spearman correlation coefficient and p-value
        correlation, _ = pearsonr(GT_protein_flat, recon_protein_flat)
        
    return correlation

# Define the directory path, channels list, and measure functions
_, singles_names, _, _, _, _ = read_compression_matrix_form(compression_matrices_csv_path)

# Iterate over all FOVs and channels
dir_path = f'results/{run_name}'
fov_dirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
correlations_df = pd.DataFrame()
for fov in fov_dirs:
    for protein in singles_names:
        # Load the channel and post-ref images
        GT_protein_img_path = os.path.join(dir_path, fov, f"{protein}.tif")
        recon_protein_img_path = [os.path.join(dir_path, fov, f) for f in os.listdir(os.path.join(dir_path, fov)) if f.startswith(f"recon_{protein}_")][0]
        GT_protein_img = skimage.io.imread(GT_protein_img_path)
        recon_protein_image = skimage.io.imread(recon_protein_img_path)
        correlation = calculate_blur_pearson_correlation(GT_protein_img, recon_protein_image)
        correlations_df.loc[fov, protein] = correlation

# save df in your output directory
correlations_df.to_csv(os.path.join(dir_path, f'Pearson correlations - {run_name}.csv'))

# Order the df according to the median pearson corrlation
channel_order = correlations_df.median().sort_values(ascending=False).index
correlations_df = correlations_df[channel_order]
# figure settings
num_channels = len(singles_names)
plt.figure(figsize=(num_channels * 2 + 2, 30), dpi=300)
sns.set(style="white")
PROPS = {
'boxprops':{'facecolor':'grey', 'edgecolor':'black'},
'medianprops':{'color':'black'},
'whiskerprops':{'color':'black'},
'capprops':{'color':'black'}
}
ax = sns.boxplot(data=correlations_df, color='w', width=0.6, showfliers=False, linewidth=0.5, **PROPS)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['left'].set_linewidth(0.5)
# Add median lines to the plot
medians = correlations_df.median().round(2)
for i, median in enumerate(medians):
    plt.text(i, median + 0.01, f"{median:.2f}", horizontalalignment='center', fontsize=24, color='black')
# Set the x-tick labels to channel names
plt.xticks(range(0, num_channels), correlations_df.columns, fontsize=30, rotation=90)
plt.yticks(fontsize=30)
# Add labels and title to the plot
plt.xlabel("\nProteins", fontsize=24)
plt.ylabel("Pearson correlation\n", fontsize=24)
plt.ylim(0, 1)
plt.title("Pearson correlations by protein\n", fontsize=30)
# Show and save the plot
plt.savefig(os.path.join(dir_path, f'Box Plot - Pearson correlations - {run_name}.pdf'), format='pdf',bbox_inches='tight')
plt.show()