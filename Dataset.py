from torch.utils.data import Dataset
from torchvision import transforms as T
import torch.nn.functional as F
import torch
import skimage.io as io
import numpy as np
import os
from util.util import read_compression_matrix_form

# Pytorch random 90 degrees rotation object
class Random90Rotation(object):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        k = torch.randint(low=0, high=4, size=(1,))[0]
        return torch.rot90(x, k=k, dims=[-2, -1])

# Random rotation for each 2*2 patch in the image, used for augmenting MIBI data 
def swap_img_patches(protein_single_imgs_tensor):
    # Get the dimensions of the input image
    batch_size, channels, rows, cols = protein_single_imgs_tensor.shape
    
    for fov_idx in range(batch_size):
        fov_imgs = protein_single_imgs_tensor[fov_idx]
        # Check if the image size is divisible by 2
        if rows % 2 != 0 or cols % 2 != 0:
            raise ValueError("Image dimensions must be divisible by 2 for splitting into 2x2 patches.")

        # Split the image into 2x2 patches
        patches = fov_imgs.view(1, channels, rows // 2, 2, cols // 2, 2)

        # Divide the patches into two groups
        num_groups = 8
        group_size = patches.shape[2] // num_groups
        group_indices = torch.randperm(patches.shape[2])

        # Randomly rotate only one group of patches
        rotated_patches = torch.zeros_like(patches)

        for i in range(num_groups):
            group_mask = (group_indices >= i * group_size) & (group_indices < (i + 1) * group_size)
            rotation_angle = torch.randint(4, (1,)).item()
            rotated_patches[:, :, group_mask] = torch.rot90(patches[:, :, group_mask], k=rotation_angle, dims=(3, 5))

        # Reshape the rotated patches back to the original shape
        protein_single_imgs_tensor[fov_idx] = rotated_patches.view(1, channels, rows, cols)

    return protein_single_imgs_tensor

class Combplex_Dataset(Dataset):
    def __init__(self, dataset_path, is_validation, config, device):
        self.batch_size = config['batch_size']
        scale_CODEX_vals = True if ((config['model_type'] == 'Value reconstruction network' ) and (config['imaging_platform'] == 'CODEX')) else False
        self.device = device
        self.is_validation = is_validation

        # load compression matrix form details
        _, _, multis_names, train_compression_matrix, test_compression_matrix, GT_filenames = read_compression_matrix_form(config['compression_matrix_form_path'])
        if is_validation:
            # use real multis
            self.simulation_mat = torch.from_numpy(test_compression_matrix.values).unsqueeze(-1).unsqueeze(-1).to(device)
            simulation_filenames = list(test_compression_matrix.columns)
        else:
            # use single protein images for simulation
            self.simulation_mat = torch.from_numpy(train_compression_matrix.values).unsqueeze(-1).unsqueeze(-1).to(device)
            simulation_filenames = list(train_compression_matrix.columns)
        self.num_proteins = len(GT_filenames)
        self.num_sim_imgs = len(simulation_filenames)

        # load noise parameters
        self.noise_parameters = config['noise_parameters']

        # set tranforms according to data type
        self.general_augmentations = T.Compose([T.RandomCrop(config['crop_size']), T.RandomHorizontalFlip(), T.RandomVerticalFlip(), Random90Rotation()])
        self.imaging_platform = config['imaging_platform']

        # load dataset
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
            if scale_CODEX_vals:
                fov_stacked_tensor = fov_stacked_tensor / (2 ** 16 - 1)
            # add stacked tensor to dataset
            self.data.append(fov_stacked_tensor)

        print('loaded %d images' % len(self.data))
        self.step = torch.zeros(1) - 1

    def __getitem__(self, idx):
        self.step += 1
        sample = {'step': self.step}
        
        # create singles batch tensor
        batch_data = []
        for _ in range(self.batch_size):
            # sample a random image
            i = torch.randint(0, len(self.data), (1,))[0]
            fov_data = self.data[i].float()
            fov_data = self.general_augmentations(fov_data)
            batch_data.append(fov_data)
        batch_data = torch.stack(batch_data, dim=0).to(self.device)

        # create sample of dataset
        if self.is_validation:
            # apply no noise or additional augmentations
            simulation_imgs, sample['GT_singles'] = torch.split(batch_data, [self.num_sim_imgs, self.num_proteins], dim=1)
            if self.imaging_platform == 'MIBI':
                sample['GT_singles'] = torch.nn.AvgPool2d(kernel_size=5, stride=1, padding=2)(sample['GT_singles'])
            sample['multis'] = F.conv2d(simulation_imgs, self.simulation_mat)
        else:
            n1, n2, n3, n4 = self.noise_parameters
            # create noisy compression matrix A' using n1,n2
            simulation_mat = torch.abs(np.random.normal(1, n1) * torch.normal(mean=self.simulation_mat, std=(n2 * self.simulation_mat)))
            simulation_imgs, sample['GT_singles'] = torch.split(batch_data, [self.num_sim_imgs, self.num_proteins], dim=1)
            # if MIBI, apply transform of swapping 2*2 patches and smoothing GT
            if self.imaging_platform == 'MIBI':
                simulation_imgs = swap_img_patches(simulation_imgs)
                sample['GT_singles'] = torch.nn.AvgPool2d(kernel_size=5, stride=1, padding=2)(sample['GT_singles'])
            # create multis y = A' * x
            multis = F.conv2d(simulation_imgs, simulation_mat)
            # Apply pixelwise noise on multis using n3,n4 
            sample['multis'] = torch.abs(torch.normal(multis, n3 * multis) + torch.normal(n4 * torch.ones_like(multis), 0.2 * n4 * torch.ones_like(multis))) * (multis > 0)

        return sample

    def __len__(self):
        return len(self.data)


