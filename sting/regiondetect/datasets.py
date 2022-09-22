
from tifffile import imread
import pathlib
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset
import cv2


class BarcodeDataset(Dataset):
    """
    
    Constructs datapoints that are seen by the network
    
    Arguments:
        data_dir (pathlib.Path or str): path of the data directory 
    
    """
    def __init__(self, data_dir, img_size=(256, 800), multiscale=False, transform=None):
        super(BarcodeDataset, self).__init__()
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        assert isinstance(data_dir, pathlib.Path) == False, "data directory is not a valid path"
        
        # images are in /images directory
        self.images_dir = self.data_dir / Path('images')
        # labels are in /labels directory
        self.labels_dir = self.data_dir / Path('labels')
        
        self.img_size = img_size
        self.multiscale = multiscale
        self.transform = transform
        
        # build tuples of filenames and labels here
        filenames = self.images_dir.glob('*.tiff')
        filestems = [file.stem for file in filenames]
        self.data_tuples = [(self.images_dir / Path(filestem + '.tiff'),
                             self.labels_dir / Path(filestem + '.txt')) 
                                    for filestem in filestems]

        # not figured out where this will be used exactly
        self.batch_count = 0
        
    def __len__(self):
        return len(self.data_tuples)
    
    def __getitem__(self, idx):
        # get one data point from the dataset that is transformed
        image_filename, bbox_filename = self.data_tuples[idx]
        image = imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = np.loadtxt(bbox_filename)
        #for bbox in bboxes:
        #    bbox.append(0)
        
        datapoint = {
            'image': image,
            'bboxes': bboxes,
        }
        
        if self.transform != None:
            transformed = self.transform(datapoint)
            image = transformed['image']
            bboxes = transformed['bboxes']
        return {
            'image': image,
            'bboxes': bboxes,
            'path': str(image_filename)
        }

    def collate_fn(self, batch):
        self.batch_count += 1 

        # drop invalid images
        batch = [data for data in batch if data is not None]

        # images stack
        images = torch.stack([data['image'] for data in batch])

        # you need to impelement multiscale option here
        # TODO: later. Need to figure out how bboxes need
        # to be transformed for multiscale training
        # for now, we do only one scale

        bb_targets = [data['bboxes'] for data in batch]
        for i, bboxes in enumerate(bb_targets):
            # first col of each bbox in a batch corresponds to image idx in the batch
            bboxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        paths = [data['path'] for data in batch]
        
        return paths, images, bb_targets
    
    def plot_datapoint(self, idx):
        datapoint = self.__getitem__(idx)
        image = datapoint['image']
        #print(image.shape)
        bboxes = datapoint['bboxes']
        #print(bboxes)
        if image.ndim == 3:
            image = image[0, :,:,]
        #image = image/65535
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image, cmap='gray')
        height, width = image.shape
        #print(f"Height: {height} -- width: {width}")
        if bboxes != None:
            for bbox in bboxes:
                w = bbox[4] * width
                h = bbox[5] * height
                l = int((bbox[2] - bbox[4]/2) * width)
                t = int((bbox[3] - bbox[5]/2) * height)
                rect = patches.Rectangle((l, t), w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

        plt.show()
    