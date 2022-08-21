import pathlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset
from sting.utils.types import RecursiveNamespace
from typing import Union

class MMDatasetOmni(Dataset):

    def __init__(self, phase_dir: Union[str, pathlib.Path], labels_dir: Union[str, pathlib.Path],
                transforms=None, phase_fileformat: str = '.png', labels_fileformat: str = '.png',
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
            labels_dir (str, pathlib.Path): a directory containing labelled images
            trasnforms : Transforms that are performed on the network
            phase_fileformat (str) : '*.png' or '*.tiff' or '*.tif' or whatever, 
                it is used to grab the list of files based on extensions
            labels_fileformat (str): '*.png' or '*.npy' or '*.tiff' or whatever,
                it is used to grab the list of files based on extensions

        """
        super(MMDatasetOmni, self).__init__()
        phase_fileformats = ['.png', '.tiff', '.tif']
        label_fileformats = ['.tiff', '.tif', '.png', '.npy']
        self.phase_dir = phase_dir if isinstance(phase_dir, pathlib.Path) else Path(phase_dir)
        self.labels_dir = labels_dir if isinstance(labels_dir, pathlib.Path) else Path(labels_dir)
        assert phase_fileformat in phase_fileformats, "Phase file format is not .png, .tiff, .tif"
        assert labels_fileformat in label_fileformats, "Label file format is not .png, .tiff, .tif, .npy"



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
    for each_species in species: 
        # construct dataset using the directories
        pass
    
    return

class MMDatasetTest(Dataset):

    def __init__(self):
        pass

    def __len__(self):
        pass
    
    def __getitem__(self):
        pass
