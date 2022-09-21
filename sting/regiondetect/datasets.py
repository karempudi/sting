
from tifffile import imread
import pathlib
from pathlib import Path
import numpy as np
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
    def __init__(self, data_dir, img_size=(800, 256), multiscale=True, transform=None):
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
        
    def __len__(self):
        return len(self.data_tuples)
    
    def __getitem__(self, idx):
        # get one data point from the dataset that is transformed
        image_filename, bbox_filename = self.data_tuples[idx]
        image = imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = np.loadtxt(bbox_filename)[:, 1:].tolist()
        for bbox in bboxes:
            bbox.append('barcode')
        
        datapoint = {
            'image': image,
            'bboxes': bboxes,
        }
        
        if self.transform != None:
            transformed = self.transform(datapoint)
            image = transformed['image']
            bboxes = transformed['bboxes']
        return {
            'image': image[:, :, 0],
            'bboxes': bboxes
        }
    
    def plot_datapoint(self, idx):
        datapoint = self.__getitem__(idx)
        image = datapoint['image']
        bboxes = datapoint['bboxes']
        if image.ndim == 3:
            image = image[:,:, 0]
        #image = image/65535
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image, cmap='gray')
        height, width = image.shape
        #print(f"Height: {height} -- width: {width}")
        if bboxes != None:
            for bbox in bboxes:
                w = bbox[2] * width
                h = bbox[3] * height
                l = int((bbox[0] - bbox[2]/2) * width)
                t = int((bbox[1] - bbox[3]/2) * height)
                rect = patches.Rectangle((l, t), w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

        plt.show()
    