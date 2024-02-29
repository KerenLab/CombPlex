import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
import sys
sys.path.append('.')
from util.util import read_compression_matrix_form

'''
This script analyzes the choice of a specific table including sums of rows, columns and hamming distances.
'''
# set parameters
compression_matrices_form_name = 'CODEX_22to5_in-silico_paper_exp' # name of table to analyze
plot = True # plot png file with relevant analysis and diagrams
txt = True # output txt file with relevant analysis

def hamming_dist(A):
    # return hamming distance matrix for the columns of a matrix
    num_cols = A.shape[1]
    hamming_matrix = np.zeros((num_cols, num_cols))
    for i in range(num_cols):
        for j in range(i, num_cols):
            hamming_matrix[i, j] = hamming(A[:, i], A[:, j]) * len(multis_names)
    hamming_matrix = hamming_matrix + hamming_matrix.T
    return hamming_matrix


# read table
A_df, singles_names, multis_names, _, _, _ = read_compression_matrix_form(f'compression_matrices/{compression_matrices_form_name}.csv')
proteins_names = list(A_df.columns)
multis_names = list(A_df.index.values)
A = (A_df.values > 0).astype(int)

# collect statistics
hamming_matrix = hamming_dist(A)
mean_hamming_dist = hamming_matrix.mean()
sums_cols = A.sum(axis=0)
mean_sum_cols = sums_cols.mean()
sums_rows = A.sum(axis=1)
mean_sum_rows = sums_rows.mean()

# plotting
if plot:
    num_cols = len(proteins_names)
    fig = plt.figure(figsize=[2 * num_cols,1.75 * num_cols])
    plt.suptitle(f'Table Analysis - {compression_matrices_form_name}\n')

    ax1 = fig.add_subplot(2,1,1)
    heatmap = ax1.imshow(hamming_matrix)
    ax1.set_xticks(np.arange(len(proteins_names)), labels=proteins_names)
    ax1.set_yticks(np.arange(len(proteins_names)), labels=proteins_names)
    plt.setp(ax1.get_xticklabels(), rotation=90, ha="right",
            rotation_mode="anchor")
    for i in range(len(proteins_names)):
        for j in range(len(proteins_names)):
            text = ax1.text(j, i, '{}'.format(int(hamming_matrix[i, j])),
                        ha="center", va="center", color="w")
    # plt.colorbar(heatmap, ax=ax1, label='Hamming distance', orientation='vertical')
    ax1.set_title(f"Hamming Distance Heatmap\nmean={mean_hamming_dist:.2f}")

    ax2 = fig.add_subplot(2,2,3)
    ax2.bar(proteins_names, sums_cols, color='orange')
    ax2.axline((0, mean_sum_cols), slope=0, linestyle='dashed', color='black')
    plt.setp(ax2.get_xticklabels(), rotation=90, ha="right",
            rotation_mode="anchor")
    plt.xlabel('Target name')
    plt.ylabel('Antibody partition')
    plt.title(f'Sum of rows distribution\nmean={mean_sum_cols:.2f}\n')

    ax3 = fig.add_subplot(2,2,4)
    ax3.bar(multis_names, sums_rows, color='orange')
    ax3.axline((0, mean_sum_rows), slope=0, linestyle='dashed', color='black')
    plt.setp(ax3.get_xticklabels(), rotation=90, ha="right",
            rotation_mode="anchor")
    plt.xlabel('Target name')
    plt.ylabel('Num of overlapping images')
    plt.title(f'Sum of columns distribution\nmean={mean_sum_rows:.2f}\n')

    fig.tight_layout()
    plt.savefig(f'compression_matrices/{compression_matrices_form_name} statistics.png', dpi=300)

# creating txt file
if txt:
    lines = [f'Table name: {compression_matrices_form_name}\n', f'Num of channels: {len(proteins_names)}', f'Num of compressed channels: {len(multis_names)}',
                f'Channels: {proteins_names}', f'Compressed_channels: {multis_names}\n', f'Max sum of columns: {sums_cols.max()}', f'Mean sum of columns: {mean_sum_cols:.2f}\n',
                f'Max sum of rows {sums_rows.max()}', f'Mean sum of rows: {mean_sum_rows:.2f}\n', f'Mean hamming distance of columns: {mean_hamming_dist:.2f}',
                f'Min hamming distance: {hamming_matrix[hamming_matrix > 0].min():.2f}']
    with open(f'compression_matrices/{compression_matrices_form_name} statistics.txt', 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')