import torch
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
from tifffile import imread

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

################################################
########### Classical U-net dataset ############
################################################

class MMDatasetUnet(Dataset):

    def __init__(self, data_dir: Union[str, pathlib.Path],
                transform=None, weights=False, fileformats = {
                    'phase': '.tif', 'mask': '.tif', 'weights': '.tif'
                }):
        """
        A dataset of a species is in data_dir, with subdirs 'phase', 'mask' & 'weights'
        If you want to include more species, use concatenation of different datasets 
        after creating an MMDatasetUnet for each species

        Args:
            data_dir (str or pathlib.Path): directory containing data in directores,
                        'phase', 'mask' and 'weights'(optional)
                    For each phase file , there must be a mask file and weights file (if used)
                    Note: Pass the extensions correctly for each dataset.
            transform: transforms applied to a datapoint in the dataset
            weights (boolean): are weights included int he dataset or not
            fileformat (dict): fileformats to grab files from directories with
        """
        super(MMDatasetUnet, self).__init__()

        self.data_dir =  data_dir if isinstance(data_dir, pathlib.Path) else Path(data_dir)
        self.phase_dir = self.data_dir / Path('phase')
        self.mask_dir = self.data_dir / Path('mask')
        self.use_weights = weights
        self.fileformats = fileformats
        self.transform = transform

        if self.use_weights:
            self.weights_dir = self.data_dir / Path('weights')

        self.phase_filenames = list(self.phase_dir.glob('*' + fileformats['phase']))
        self.mask_filenames = [self.mask_dir / Path(filename.stem + fileformats['mask']) 
                                    for filename in self.phase_filenames]
        if self.use_weights:
            self.weights_filenames = [self.weights_dir / Path(filename.stem + fileformats['weights']) 
                                    for filename in self.phase_filenames]


    def __len__(self):
        return len(self.phase_filenames)

    def __getitem__(self, idx):

        phase_img = imread(self.phase_filenames[idx])
        mask_img = imread(self.mask_filenames[idx]) 
        if self.use_weights:
            weights_img = imread(self.weights_filenames[idx])
        else:
            weights_img = None
        
        height, width = phase_img.shape

        sample = {
            'phase': phase_img,
            'mask': mask_img,
            'weights': weights_img,
            'filename': self.phase_filenames[idx].name,
            'raw_shape': (height, width)
        }

        if self.transform != None:
            sample = self.transform(sample)
        
        return sample
    
    def plot_datapoint(self, idx):
        pass


class MMDatasetUnetTest(Dataset):

    def __init__(self, images_dir: Union[str, pathlib.Path], fileformat='.tif*',
                transform=None):
        super(MMDatasetUnetTest, self).__init__()
        
        self.images_dir = images_dir if isinstance(images_dir, pathlib.Path) else Path(images_dir)
        self.fileformat = fileformat
        self.transform = transform

        self.filenames = list(self.images_dir.glob('*' + self.fileformat))

        self.batch_count = 0
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        
        phase_img = imread(self.filenames[idx]).astype('float32')
        height, width = phase_img.shape

        sample = {
            'phase': phase_img,
            'filename': self.filenames[idx].name,
            'raw_shape' : (height, width)
        }

        if self.transform != None:
            sample = self.transform(sample)
        
        return sample
    
    def plot_datapoint(self, idx):
        pass

    def collate_fn(self, batch):
        self.batch_count += 1

        # drop invalid images
        batch = [data for data in batch if data is not None]

        phase = torch.stack([data['phase'] for data in batch])

        filenames = [data['filename'] for data in batch]
        raw_shapes = [data['raw_shape'] for data in batch]

        return phase, filenames, raw_shapes


####################################################################
####### Dual U-net dataset to predict both cell and channels #######
####################################################################


class MMDatasetUnetDual(Dataset):

    def __init__(self, data_dir: Union[str, pathlib.Path],
                transform=None, weights=False, fileformats = {
                    'phase': '.tif', 'cell_mask': '.tif', 'channel_mask': '.tif', 'weights': '.tif'
                }):
        """
        A dataset of a species is in data_dir, with subdirs 'phase', 'mask' & 'weights'
        If you want to include more species, use concatenation of different datasets 
        after creating an MMDatasetUnetDual for each species

        Dual means that we use one network to predict both cells and channels masks
        So that datasets and dataloaders should deliver identically transformed images
        of cell mask and channels mask for each transformed phase-contrast training image.
        If there are weigths, they should be specified for cells only, as we wont need weights
        for channels as they never really touch each other.

        Args:
            data_dir (str or pathlib.Path): directory containing data in directores,
                        'phase', 'cell_mask', 'channel_mask', 'weights_cell'(optional) 
                    For each phase file , there must be a cell mask file and channel mask file
                    and weights file (if used)
                    Note: Pass the extensions correctly for each dataset.
            transform: transforms applied to a datapoint in the dataset
            weights (boolean): are weights included int he dataset or not
            fileformat (dict): fileformats to grab files from directories with
        """
        super(MMDatasetUnetDual, self).__init__()

        self.data_dir =  data_dir if isinstance(data_dir, pathlib.Path) else Path(data_dir)
        self.phase_dir = self.data_dir / Path('phase')
        self.cell_mask_dir = self.data_dir / Path('mask')
        self.channel_mask_dir = self.data_dir / Path('channel_mask')
        self.use_weights = weights
        self.fileformats = fileformats
        self.transform = transform

        if self.use_weights:
            self.weights_dir = self.data_dir / Path('weights')

        self.phase_filenames = list(self.phase_dir.glob('*' + fileformats['phase']))
        self.cell_mask_filenames = [self.cell_mask_dir / Path(filename.stem + fileformats['cell_mask']) 
                                    for filename in self.phase_filenames]
        self.channel_mask_filenames = [self.channel_mask_dir / Path(filename.stem + fileformats['channel_mask']) 
                                    for filename in self.phase_filenames]
        if self.use_weights:
            self.weights_filenames = [self.weights_dir / Path(filename.stem + fileformats['weights']) 
                                    for filename in self.phase_filenames]


        self.batch_count = 0

    def __len__(self):
        return len(self.phase_filenames)

    def __getitem__(self, idx):

        phase_img = imread(self.phase_filenames[idx])
        cell_mask_img = imread(self.cell_mask_filenames[idx])
        channel_mask_img = imread(self.channel_mask_filenames[idx])
        if self.use_weights:
            weights_img = imread(self.weights_filenames[idx])
        else:
            weights_img = None
        
        height, width = phase_img.shape

        sample = {
            'phase': phase_img.astype('float32'),
            'mask': cell_mask_img,
            'channel_mask': channel_mask_img,
            'weights': weights_img,
            'filename': self.phase_filenames[idx].name,
            'raw_shape': (height, width)
        }

        if self.transform != None:
            sample = self.transform(sample)
        
        return sample
    
    def plot_datapoint(self, idx):
        pass

    def collate_fn(self, batch):
        self.batch_count += 1

        # drop invalid images
        batch = [data for data in batch if data is not None]

        phase = torch.stack([data['phase'] for data in batch])

        mask = []
        channel_mask = []
        weights = []
        for data in batch:
            if data['mask'] is not None:
                mask.append(data['mask'])
            else:
                mask.append(torch.tensor([-1]))
            
            if data['channel_mask'] is not None:
                channel_mask.append(data['channel_mask'])
            else:
                channel_mask.append('0')
            
            if data['weights'] is not None:
                weights.append(data['weights'])
            else:
                weights.append('0')

        if batch[0]['mask'] is not None:
            mask = torch.stack(mask)
        if batch[0]['channel_mask'] is not None:
            channel_mask = torch.stack(channel_mask)
        if batch[0]['weights'] is not None:
            weights = torch.stack(weights)

        filenames = [data['filename'] for data in batch]
        raw_shapes = [data['raw_shape'] for data in batch]

        return phase, mask, channel_mask, weights, filenames, raw_shapes


####################################################################
################# Concatenation of datasets ########################
####################################################################

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
