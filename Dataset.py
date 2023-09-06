from torch.utils.data import Dataset
from torchvision import transforms as T
import torch.nn.functional as F
import torch
import skimage.io as io
import pandas as pd
import numpy as np
import os
from util.util import read_compression_matrix_form

class RandomRotation(object):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        x = T.RandomHorizontalFlip()(x)
        k = torch.randint(low=0, high=4, size=(1,))[0]
        return torch.rot90(x, k=k, dims=[-2, -1])

def fluorescence_skew(x, num_proteins_map, multis_names, skew_parameters, fluorescence_gaussian_noise):
    num_proteins_vals = np.unique(num_proteins_map.cpu())
    # iterate over all multis
    for i, multi_name in enumerate(multis_names):
        # iterate over all possible values for how many proteins expressed in a pixel
        for num_protein_val in num_proteins_vals[num_proteins_vals > 0]:
            # skew the multis according to the supplied table
            skew_coeff = 1 / (1 + skew_parameters.loc['mean', f'{multi_name}_{int(num_protein_val)}'])
            x[:,i][num_proteins_map[:, i] == num_protein_val] = torch.normal(skew_coeff * x[:,i][num_proteins_map[:, i] == num_protein_val], 
                                                                             fluorescence_gaussian_noise * x[:,i][num_proteins_map[:, i] == num_protein_val])
    return x

class Combplex_Dataset(Dataset):
    def __init__(self, dataset_path, cfg, device):
        self.cfg = cfg
        self.device = device

        # load simulation details
        _, _, multis_names, train_compression_matrix, _, GT_filenames = read_compression_matrix_form(cfg['compression_matrix_form_path'])
        simulation_filenames = list(train_compression_matrix.columns)
        self.simulation_mat = torch.from_numpy(train_compression_matrix.values).unsqueeze(-1).unsqueeze(-1).to(device)
        self.num_proteins = len(GT_filenames)
        self.num_sim_imgs = len(simulation_filenames)
        self.multis_names = multis_names
        if cfg['data_type'] == 'Fluorescence':
            self.fluorescence_skew = fluorescence_skew
            self.skew_parameters = pd.read_csv(cfg['skew_parameters_csv_path'], index_col=0)
        
        # set tranforms according to data type
        x_transform_lst, y_transform_lst = [T.RandomCrop(cfg['crop_size']), RandomRotation()], []
        if cfg['data_type'] == 'MIBI':
            x_transform_lst.append(torch.nn.AvgPool2d(kernel_size=5, stride=1, padding=2))
            y_transform_lst.append(T.Lambda(lambda x: torch.poisson(x)))
        self.x_transform, self.y_transform = T.Compose(x_transform_lst), T.Compose(y_transform_lst) 
        
        # load datasets
        self.data = []
        for fov in os.listdir(dataset_path):
            fov_path = os.path.join(dataset_path, fov)
            if not os.path.isdir(fov_path):
                continue
            # stack all images to one tensor
            fov_stacked = []
            for file_name in simulation_filenames:
                fov_stacked.append(io.imread(os.path.join(fov_path, '{}.tif'.format(file_name))))
            for file_name in GT_filenames:
                fov_stacked.append(io.imread(os.path.join(fov_path, '{}.tif'.format(file_name))))
            fov_stacked_tensor = torch.from_numpy(np.stack(fov_stacked).astype(np.float32)).float()
            # for fluorescence data, calculate the num proteins expressed in every pixel in each multi
            if cfg['data_type'] == 'Fluorescence':
                for multi_name in multis_names:
                    indices = [i for i, value in enumerate(train_compression_matrix.loc[multi_name].tolist()) if value != 0]
                    num_proteins_map = (fov_stacked_tensor[indices] > 0).float().sum(dim=0).int()
                    fov_stacked.append(num_proteins_map)
                fov_stacked_tensor = torch.from_numpy(np.stack(fov_stacked).astype(np.float32)).float()
            # add stacked tensor to dataset
            self.data.append(fov_stacked_tensor)

        print('loaded %d images' % len(self.data))
        self.step = torch.zeros(1) - 1

    def __getitem__(self, index):
        self.step += 1
        sample = {'step': self.step}
        batch_data = []
        # create singles batch tensor
        for _ in range(self.cfg['batch_size']):
            # sample a random image
            i = torch.randint(0, len(self.data), (1,))[0]
            fov_data = self.data[i].float()
            fov_data = self.x_transform(fov_data)
            batch_data.append(fov_data)
        batch_data = torch.stack(batch_data, dim=0).to(self.device)
        # create sample of dataset
        simulation_mat = torch.normal(mean=self.simulation_mat, std=(self.cfg['compression_matrix_noise'] * self.simulation_mat))
        if self.cfg['data_type'] == 'Fluorescence':
            simulation_singles, sample['GT_singles'], num_proteins_map = torch.split(batch_data, [self.num_sim_imgs, self.num_proteins, len(self.multis_names)], dim=1)
            multis = F.conv2d(simulation_singles, simulation_mat)
            multis = self.fluorescence_skew(multis, num_proteins_map, self.multis_names, self.skew_parameters, self.cfg['fluorescence_gaussian_noise'])
        else:
            simulation_singles, sample['GT_singles'] = torch.split(batch_data, [self.num_sim_imgs, self.num_proteins], dim=1)
            multis = self.y_transform(F.conv2d(simulation_singles, simulation_mat))
        sample['multis'] = multis
        return sample

    def __len__(self):
        return len(self.data)


