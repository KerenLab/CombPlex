import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import skimage.io as io
import os
import pandas as pd
import wandb
import pprint
import csv
from util.util import read_compression_matrix_form

# set these parameters before executing
compression_matrices_csv_path = 'compression_matrices/your_compression_matrices_form.csv'
dataset_path = 'datasets/your_validation_set' # path to the validation set for adjusting the compression matrix
learning_rate = 10 ** -3  
batch_size = 1
num_epochs = 10000

wandb.init(project='Combplex',name='compression_matrix_adjustment', entity='liorbensha', config={'batch size': batch_size, 'learning rate': learning_rate})
log_data = {}

# Define device as CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

A_df, _, _, train_compression_matrix, _, singles_filenames = read_compression_matrix_form(compression_matrices_csv_path)
multis_filenames = train_compression_matrix.index.tolist()
input_channels = len(singles_filenames)  # number of channels in the input data X
output_channels = len(multis_filenames) # number of channels in the output data Y

class Multis_Dataset(Dataset):
    def __init__(self, dataset_path, singles_filenames, multis_filenames, device):
        self.device = device
        
        # set tranforms according to data type
        self.len_singles_list = len(singles_filenames)
        
        # load datasets
        self.data = []
        for fov in os.listdir(dataset_path):
            fov_path = os.path.join(dataset_path, fov)
            if not os.path.isdir(fov_path):
                continue
            # stack all channels as one image
            fov_stacked = []
            for file_name in singles_filenames:
                fov_stacked.append(io.imread(os.path.join(fov_path, '{}.tif'.format(file_name))))
            for file_name in multis_filenames:
                fov_stacked.append(io.imread(os.path.join(fov_path, '{}.tif'.format(file_name))))
            fov_stacked = torch.from_numpy(np.stack(fov_stacked).astype(np.float32)).float()
            # add stacked image to dataset
            self.data.append(fov_stacked.to(device))

        print('loaded %d images' % len(self.data))

    def __getitem__(self, index):
        index = torch.randint(0,len(self.data),(1,))
        fov_data = self.data[index].float()
        return fov_data[:self.len_singles_list], fov_data[self.len_singles_list:]

    def __len__(self):
        return batch_size


class Net(nn.Module):
    def __init__(self, C, M, initial_A):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(C, M, kernel_size=1)
        self.conv.weight.data = initial_A
        self.mask = (initial_A != 0).float().to(device)
        

    def forward(self, x):
        self.conv.weight.data[self.mask ==0] = 0
        self.conv.weight.data = self.conv.weight.data.clamp(min=0)
        x = self.conv(x)
        return x

# Create the network
initial_A = torch.from_numpy(A_df.values).unsqueeze(-1).unsqueeze(-1)
net = Net(input_channels, output_channels, initial_A).to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# Load the labeled data in pairs (X,Y) and create PyTorch DataLoader
train_data = Multis_Dataset(dataset_path, singles_filenames, multis_filenames, device)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Train the network
loss_values_train = []
previous_A = initial_A
epoch = 0
while True:
    running_loss_train = 0.0
    running_loss_val = 0.0
    
    # Train on the training data
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss_train += loss.item()
        
    # Save the losses for plotting later
    log_data['epoch'] = epoch
    loss_values_train.append(running_loss_train / len(train_loader))
    log_data['train_loss'] = running_loss_train / len(train_loader)
    wandb.log(log_data)
    
    if epoch % 1000 == 0 and epoch != 0:
        A = net.conv.weight.data.cpu() * (initial_A > 0)
        A_squeezed = A.squeeze(-1).squeeze(-1).tolist()
        # Round each float to 2 decimal places
        A_rounded = [[round(elem, 2) for elem in row] for row in A_squeezed]
        print(f'Epoch = {epoch} | loss = {running_loss_train / len(train_loader)}', )
        # Pretty-print the nested list
        pprint.pprint(A_rounded)
        # Finish if the matrix has been converged
        if torch.mean(torch.abs(A[initial_A != 0] - previous_A[initial_A != 0])) < 0.01:
            print('\nFinished! Compression matricess form is updated! :)')
            break
        previous_A = A
    epoch += 1

# rewrite the compression matrices file
separators = ['Reconstruction matrix A (note: protein and channel names can be represented by nicknames instead of filenames):\n',
              'Training compression matrix (note: the columns and rows headers must be filenames. Also; keep the same order)\n',
              'Test compression matrix (note: the columns and rows headers must be filenames. Also; keep the same order)\n',
              'GT filename for each protein (note: keep the same order):\n']
# change A_df values with the adjusted ones
adjusted_A_df = pd.DataFrame(A_rounded)
A_df.iloc[:, :] = adjusted_A_df.values
combined_csv = ''
for sep in separators[0:-1]:
    combined_csv += sep
    combined_csv += A_df.to_csv()
combined_csv += separators[-1]
combined_csv += 'GT filenames,' + ','.join(A_df.columns.to_list())
lines = combined_csv.split('\n')
# Write the lines to a CSV file
with open(compression_matrices_csv_path, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    for line in lines:
        cells = line.split(',')
        [cell.replace(';', ',') for cell in cells]
        csv_writer.writerow(cells)