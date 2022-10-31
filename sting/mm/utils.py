import numpy as np
import albumentations as A
import cv2
import torch
from torchvision import transforms

class YoloLiveAugmentations:

    """
    Test time augmentations done on the image before the yolo net sees
    and image, basically doing image resizing and tensorizing

    Args:
        parameters: parameters used by various transformations.

    """

    def __init__(self, parameters = {
        'to_float' : {'max_value': 65535.0},
        'resize': {'height': 256, 'width': 800, 'interpolation': cv2.INTER_LINEAR, 'always_apply': True},
        'from_float': {'max_value': 65535.0}
    }):
        self.parameters = parameters
        self.transform = A.Compose([
            A.ToFloat(**parameters['to_float']),
            A.Resize(**parameters['resize']),
            A.FromFloat(**parameters['from_float'])
        ])
        self.img_size = (self.parameters['resize']['height'], self.parameters['resize']['width'])
    def __call__(self, datapoint):
        # datapoint is a dict with keys = {'phase', ...}
        datapoint['phase'] = cv2.cvtColor(datapoint['phase'], cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=datapoint['phase'])
        image = transforms.ToTensor()(transformed['image'][:, :, 0].astype('float32'))
        datapoint['phase'] = image
        return datapoint

class YoloLiveUnAugmentations:

    """
    Test time Unagumentatoins done on the  image after the yolo net sees the images and 
    gives the bounding boxes
    Args:
        parameters: parameters used by various transforms

    """
    def __init__(self, parameters = {
        'resize' : {'height': 816, 'width': 4096, 'always_apply':True} # this the resize height to ie, the original image
    }):
        self.transform = A.Compose([
            A.Resize(**parameters['resize'])
        ], bbox_params=A.BboxParams(format='pascal_voc')) #here we would have already converted the bboxes to pascal_voc
        
    def __call__(self, datapoint):
        # datapoint should have 'bboxes' and 'yolo_size'(size on which bboxes were calculated by the net)
        yolo_size = datapoint['yolo_size']
        transformed = self.transform(image = np.zeros(yolo_size), bboxes=datapoint['bboxes'])
        return transformed['bboxes']
