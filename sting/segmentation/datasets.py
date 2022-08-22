import pathlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset
from sting.utils.types import RecursiveNamespace
from typing import Union
from tqdm import tqdm
from sting.segmentation.utils import labels_to_output_omni
from skimage import io

class MMDatasetOmni(Dataset):

    def __init__(self, phase_dir: Union[str, pathlib.Path], cell_labels_dir: Union[str, pathlib.Path] = None,
                channel_labels_dir: Union[str, pathlib.Path] = None,
                transforms=None, phase_fileformat: str = '.png', cell_labels_fileformat: str = '.png',
                channel_labels_fileformat: str = '.png',
                dataset_type: str = 'cells', recalculate_flows : bool = False,
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
            dataset_type (str): 'cells', 'channels', 'dual'

            recalculate_flows (bool) : defaults to False

        """
        super(MMDatasetOmni, self).__init__()
        phase_fileformats = ['.png', '.tiff', '.tif']
        cell_fileformats = ['.tiff', '.tif', '.png', '.npy']
        channel_fileformats = ['.tiff', '.tif', '.png', '.npy']

        self.phase_dir = phase_dir if isinstance(phase_dir, pathlib.Path) else Path(phase_dir)
        self.dataset_type = dataset_type
        self.transforms = transforms

        # which type of data is this ['train', 'validation', 'test']
        self.dataset_string = self.phase_dir.name.split('_')[-1]

        if self.dataset_type in ['cells', 'dual']:
            assert cell_labels_dir != None, "Cell labels directory is None, the model expects to train on cells"
            self.cell_labels_dir = cell_labels_dir if isinstance(cell_labels_dir, pathlib.Path) else Path(cell_labels_dir)
            self.cell_flows_dir = self.phase_dir.parents[0] / Path('cell_flows_' + self.dataset_string)
        else:
            self.cell_labels_dir = None
            self.cell_flows_dir = None
        
        if self.dataset_type in ['channels', 'dual']:
            assert channel_labels_dir != None, "Channel labels directory is None, the model expects to train on channels"
            self.channel_labels_dir = channel_labels_dir if isinstance(channel_labels_dir, pathlib.Path) else Path(channel_labels_dir) 
            self.channel_flows_dir = self.phase_dir.parents[0] / Path('channel_flows_' + self.dataset_string)
        else:
            self.channel_labels_dir = None
            self.channel_flows_dir = None

        assert phase_fileformat in phase_fileformats, "Phase file format is not .png, .tiff, .tif"
        assert cell_labels_fileformat in cell_fileformats, "Label file format is not .png, .tiff, .tif, .npy"
        assert channel_labels_fileformat in channel_fileformats, "Label file format is not .png, .tiff, .tif, .npy"

        self.phase_fileformat = phase_fileformat
        self.cell_labels_fileformat = cell_labels_fileformat
        self.channel_labels_fileformat = channel_labels_fileformat

        # always check that there is a phase image dir
        assert self.phase_dir.exists() and self.phase_dir.is_dir()

        # grab all the file in the phase dir
        file_paths = list(self.phase_dir.glob('*' + self.phase_fileformat))
        self.filenames = [filepath.stem for filepath in file_paths]

        # calculate flows if they don't exist
        if self.dataset_type in ['cells', 'dual']:
            # do the calculations for cell flows here if the files are not there
            if not self.cell_flows_dir.exists():
                self.cell_flows_dir.mkdir(parents=True, exist_ok=True)
                # calculate flows
                for filename in tqdm(self.filenames):
                    # get labels
                    label_filename = filename + self.cell_labels_fileformat
                    label_filename = self.cell_labels_dir / Path(label_filename)
                    label_img = io.imread(label_filename)
                    flows = labels_to_output_omni(label_img)
                    # write the flows to the directory
                    flows_filename = filename + '.npy'
                    flows_filename = self.cell_flows_dir / Path(flows_filename)
                    np.save(flows_filename, flows)
            else:
                # if exists, check the number of flows
                assert len(list(self.cell_flows_dir.glob('*.npy'))) == len(self.filenames), f"Flows don't seem to match for {self.cell_flows_dir}"

        if self.dataset_type in ['channels', 'dual']:
            # do the calculations for channels flows here if the files are not there
            if not self.channel_flows_dir.exists():
                self.channel_flows_dir.mkdir(parents=True, exist_ok=True)
                # calculate flows
                for filename in tqdm(self.filenames):
                    # get labels
                    label_filename = filename + self.channel_labels_fileformat 
                    label_filename =  self.channel_labels_dir / Path(label_filename)
                    label_img = io.imread(label_filename)
                    flows = labels_to_output_omni(label_img)
                    # write the flows to directory
                    flows_filename = filename + '.npy'
                    flows_filename = self.channel_flows_dir / Path(flows_filename)
                    np.save(flows_filename, flows)
            else:
                # if exists, check the number of flows
                assert len(list(self.channel_flows_dir.glob('*.npy'))) == len(self.filenames), f"Flows don't seem to match for {self.channel_flows_dir}"


    def __len__(self):
        return len(self.filenames) 

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
    dataset_type = param.Datasets.type # pass the correct thing to each of the datasets

    # get train, validation and test sets for each of thes
    for each_species in species: 
        # construct dataset using the directories
        if dataset_type == 'omni_cells':        

            species_dataset = []
        elif dataset_type == 'omni_dual':
            species_dataset = []
        elif dataset_type == 'omni_channels':
            species_dataset = []
        elif dataset_type == 'unet_cells':
            species_dataset = []
        elif dataset_type == 'unet_dual':
            species_dataset = []
        elif dataset_type == 'unet_channels':
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
