import math
import itertools
import numpy as np
import pandas as pd
import csv

# set these parameters before executing
output_filename = 'my_new_compression_matrices_file' # The output compression matrices excel filename 
num_proteins = 7 # Number of proteins to compress. This parameter corresponds to the number of columns.
num_channels = 3 # optional, insert None or number. This parameter corresponds to the number of rows.
max_antibody_partition = math.inf # optional, insert math.inf or number. This parameter corresponds to the maximum sum of a column.
max_overlapping_imgs = math.inf # optional, insert math.inf or number. This parameter corresponds to the maximum sum of a row.

def fill_compression_matrix(possible_permutations, num_targets):
    # fill compression matrix until (max_column_sum - 1)
    max_column_sum = possible_permutations[:, num_targets - 1].sum()
    current_table = possible_permutations[:, possible_permutations.sum(axis=0) < max_column_sum]
    optional_permutations = possible_permutations[:, possible_permutations.sum(axis=0) >= max_column_sum]
    while current_table.shape[1] < num_targets:
        best_distance = math.inf
        # choose the column in optional_permutations that minimizes the imbalance between row sums
        for col_idx in range(optional_permutations.shape[1]):
            min_partition_in_options = optional_permutations.sum(axis=0).min()
            if optional_permutations[:, col_idx].sum() != min_partition_in_options:
                continue
            maybe_table = np.c_[current_table, optional_permutations[:, col_idx]]
            mean_maybe_table = maybe_table.sum() / maybe_table.shape[0]
            distance = ((maybe_table.sum(axis=1) - mean_maybe_table) ** 2).sum()
            if distance < best_distance:
                best_distance = distance
                best_col_idx = col_idx
        current_table = np.c_[current_table, optional_permutations[:, best_col_idx]]
        optional_permutations = np.delete(optional_permutations, best_col_idx, 1)
    return current_table

def create_compression_matrix(num_targets, max_antibody_partition, max_overlapping_imgs, num_channels):
    # if num_channels isn't set, start with the minimum possible
    if num_channels == None:
        num_channels = 2
    while True:
        # get all possible barcodes with sum of column not greater than max_antibody_partition
        all_permutations = np.array((list(itertools.product([0,1],repeat=num_channels)))).T[:,1:]
        all_permutations  = all_permutations[:,np.argsort(all_permutations.sum(axis=0))] 
        possible_permutations = all_permutations[:, all_permutations.sum(axis=0) <= max_antibody_partition] 
        # check feasibility considering the number of possible permutations
        if (possible_permutations.shape[1] < num_targets):
            print(f"Couldn't create a compression matrix of {num_channels} channels with the given constraints, trying with {num_channels + 1}...")
            num_channels += 1
            continue
        else:
            # Finding a balanced table given the possible permutations
            final_table = fill_compression_matrix(possible_permutations, num_targets)
            # check feasibility considering the constrain of possible permutations
            if False in (max_overlapping_imgs >= final_table.sum(axis=1)):
                print(f"Couldn't create a compression matrix of {num_channels} channels with the given constraints, trying with {num_channels + 1}...")
                num_channels += 1
                continue
            # sorting and adding column and row names
            final_table  = final_table[:,np.argsort(final_table.sum(axis=0))]
            num_channels = final_table.shape[0]
            targets_names, channels_names = [], []
            for i in range(num_targets):
                targets_names.append('Protein{}'.format(i + 1))
            for j in range(num_channels):
                channels_names.append('Compressed_image{}'.format(j + 1))
            print(f'Done (num_channels={num_channels}), compression matrices excel file has been created!')
            return pd.DataFrame(data=final_table, columns=targets_names, index=channels_names)
            

A_df = create_compression_matrix(num_proteins, max_antibody_partition, max_overlapping_imgs, num_channels)

# wrtie new compression matrices csv file
separators = ['Reconstruction matrix A (note: protein and channel names can be represented by nicknames instead of filenames):\n',
              'Training compression matrix (note: the columns and rows headers must be filenames. Also; keep the same order)\n',
              'Test compression matrix (note: the columns and rows headers must be filenames. Also; keep the same order)\n',
              'GT filename for each protein (note: keep the same order):\n']
combined_csv = ''
for sep in separators[0:-1]:
    combined_csv += sep
    combined_csv += A_df.to_csv()
combined_csv += separators[-1]
combined_csv += 'GT filenames,' + ','.join(A_df.columns.to_list())
lines = combined_csv.split('\n')
# Write the lines to a CSV file
with open(f'compression_matrices/{output_filename}.csv', "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    for line in lines:
        cells = line.split(',')
        [cell.replace(';', ',') for cell in cells]
        csv_writer.writerow(cells)


