import pathlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset
from sting.utils.types import RecursiveNamespace
from typing import Union

class MMDatasetOmni(Dataset):

    def __init__(self, phase_dir: Union[str, pathlib.Path], cell_labels_dir: Union[str, pathlib.Path],
                channel_labels_dir: Union[str, pathlib.Path],
                transforms=None, phase_fileformat: str = '.png', cell_labels_fileformat: str = '.png',
                channel_labels_fileformat: str = '.png'
                ):
        """
        Each dataset has two sets of images, one phase image and its
        corresponding labels image. This class bundles them for 
        dataloading for network training. 

        Generally for omnipose data, we calculate the flows and 
        target outputs before we train the network, as they can take 
        significant amount of time if you compute them on-the-fly for 
        every epoch.

        Args:
            phase_dir (str, pathlib.Path): a directory containing phase images
            cell_labels_dir (str, pathlib.Path): a directory containing labelled cells images
            channel_labels_dir (str, pathlib.Path): a directory containing labelled channels images
            trasnforms : Transforms that are performed on the network
            phase_fileformat (str) : '.png' or '.tiff' or '.tif' or whatever, 
                it is used to grab the list of files based on extensions
            cell_labels_fileformat (str): '.png' or '.npy' or '.tiff' or whatever,
                it is used to grab the list of files based on extensions
            channel_labels_fileformat (str): '.png', '.npy' or '.tiff' or '.tif'
                is is used to grab the list of files based on 

        """
        super(MMDatasetOmni, self).__init__()
        phase_fileformats = ['.png', '.tiff', '.tif']
        cell_fileformats = ['.tiff', '.tif', '.png', '.npy']
        channel_fileformats = ['.tiff', '.tif', '.png', '.npy']

        self.phase_dir = phase_dir if isinstance(phase_dir, pathlib.Path) else Path(phase_dir)
        self.cell_labels_dir = cell_labels_dir if isinstance(cell_labels_dir, pathlib.Path) else Path(cell_labels_dir)
        self.channel_labels_dir = channel_labels_dir if isinstance(channel_labels_dir, pathlib.Path) else Path(channel_labels_dir) 

        assert phase_fileformat in phase_fileformats, "Phase file format is not .png, .tiff, .tif"
        assert cell_labels_fileformat in cell_fileformats, "Label file format is not .png, .tiff, .tif, .npy"
        assert channel_labels_fileformat in channel_fileformats, "Label file format is not .png, .tiff, .tif, .npy"

        

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def plot(self, idx):
        pass

class MMDatasetUnet(Dataset):

    def __init__(self, phase_dir: Union[str, pathlib.Path], labels_dir: Union[str, pathlib.Path],
                transforms=None, weights=False, phase_fileformat: str = '.tif',
                labels_fileformat: str = '.tif'):
        """

        """
        super(MMDatasetUnet, self).__init__()
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass 

# one function for concatenating datasets
def construct_dataset(param: RecursiveNamespace):
    # set up the dataset after constructing directories
    datasets = []
    species = param.Datasets.species
    dataset_type = param.Datasets.type
    for each_species in species: 
        # construct dataset using the directories
        if dataset_type == 'omni_train_cells':        

            species_dataset = []
        elif dataset_type == 'omni_train_dual':
            species_dataset = []
        elif dataset_type == 'omni_train_channels':
            species_dataset = []
        elif dataset_type == 'unet_train':
            species_dataset = []
        elif dataset_type == 'unet_train_dual':
            species_dataset = []
        elif dataset_type == 'unet_train_channels'
            species_dataset = []
        
        datasets.append(species_dataset)
    
    return ConcatDataset(datasets)

class MMDatasetTest(Dataset):

    def __init__(self):
        pass

    def __len__(self):
        pass
    
    def __getitem__(self):
        pass
