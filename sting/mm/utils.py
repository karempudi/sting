import numpy as np
import albumentations as A
import cv2
import torch
from torchvision import transforms
import copy

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

def plot_channels_img(phase, channel_locations):
    phase_rescaled = 255 * (phase - phase.min()) / (phase.max() - phase.min())
    phase_rescaled = phase_rescaled.astype('int')
    phase_rgb = np.stack([phase_rescaled, phase_rescaled, phase_rescaled], axis=-1)
    colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0]]
    for block in channel_locations:
        for each_channel in  channel_locations[block]['channel_locations']:
            phase_rgb[:, each_channel-20:each_channel-16, :] = np.array(colors[block%3])
            #phase_rgb[:, channel_locations+20:channel_locations+22, :] = np.array(colors[block%3])
    return phase_rgb

def plot_inference_img_pyqtgraph(result):
    phase = result['phase']
    height, width = phase.shape
    channel_locations = result['channel_locations']
    barcode_locations = result['barcode_locations']

    phase_rgb = np.stack([phase, phase, phase], axis=-1)
    colors_block = [[0, 0, 65535], [0, 65535, 0], [65535, 0, 0]]
    color_barcode = np.array([49087, 16448, 49087])
    for block in channel_locations:
        for each_channel in  channel_locations[block]['channel_locations']:
            phase_rgb[:, max(each_channel-20, 0): max(each_channel-16, 0), :] = np.array(colors_block[block%3])
            #phase_rgb[:, channel_locations+20:channel_locations+22, :] = np.array(colors[block%3])
    for barcode in barcode_locations:
        # safegaurd for barcodes
        phase_rgb[max(int(barcode[1]), 0): min(int(barcode[3]), height),
                 max(int(barcode[0])-2, 0): max(int(barcode[0])+2, 0), :] = color_barcode
        phase_rgb[max(int(barcode[1]), 0): min(int(barcode[3]), height),
                min(int(barcode[2])-2, width): min(int(barcode[2])+2, width), :] = color_barcode

        phase_rgb[max(int(barcode[1])-2, 0): max(int(barcode[1])+2, 0), 
                max(int(barcode[0]), 0): min(int(barcode[2]), width), :] = color_barcode
        phase_rgb[min(int(barcode[3])-2, height): min(int(barcode[3])+2, height), 
                max(int(barcode[0]), 0): min(int(barcode[2]), width), :] = color_barcode

    return phase_rgb