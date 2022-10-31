import torch
from sting.utils.types import RecursiveNamespace
from typing import Union
from sting.mm.networks import LiveNet
from sting.mm.utils import YoloLiveAugmentations, YoloLiveUnAugmentations
from sting.segmentation.transforms import UnetTestTransforms
from sting.regiondetect.utils import non_max_suppression, to_cpu, outputs_to_bboxes
import sys

def get_loaded_model(param: RecursiveNamespace):
    """
    Function to return a list of models and a list of 
    transforms applied before and after, each of the model is run 

    Arguments:
        param: parameters used for the analysis
    
    Returns:
        models: a model
    """
    # net usually has two nets net.segment_model, net.barcode_model
    net = LiveNet(param.Analysis)
    net.load_state_dict()
    net.eval()
    return net


def bboxes_compare_error():
    pass

def get_locations_btn_barcodes(channel_img, barcode_bboxes):
    """
    This will take the channel_img, barcode_bboxes
    """
    pass

def process_image(datapoint, model, param):
    """
    Function to process one datapoint in the live analysis pipeline
    Arguments:
        datapoint: a dict with keys 'image', 'time', 'position',
        model: an instance of live net model loaded on device

    """
    device = model.device
    # transformations
    if param.Analysis.Segmentation.transformations.before_type == 'UnetTestTransforms':
        pre_segment_transforms = UnetTestTransforms() 
    if param.Analysis.Barcode.transformations.before_type == 'YoloLiveAugmentations':
        pre_barcode_transforms = YoloLiveAugmentations()

    raw_shape = datapoint['image'].shape
    seg_sample = pre_segment_transforms({'phase': datapoint['image'].astype('float32'),
                                         'raw_shape': raw_shape})
    barcode_sample = pre_barcode_transforms({'phase': datapoint['image']})

    #print(barcode_sample)
    # inference
    with torch.no_grad():
        seg_pred = model.segment_model(seg_sample['phase'].unsqueeze(0).to(device)).sigmoid().cpu().numpy().squeeze(0)
        barcode_pred = model.barcode_model(barcode_sample['phase'].unsqueeze(0).to(device))
        bboxes  = outputs_to_bboxes(barcode_pred, model.anchors_t, model.strides_t)
        bboxes_cleaned = non_max_suppression(bboxes, conf_thres = param.Analysis.Barcode.thresholds.conf,
                                                iou_thres = param.Analysis.Barcode.thresholds.iou)
        bboxes_barcode = [bbox.numpy() for bbox in bboxes_cleaned][0] # only one class so we should get this at index 0

    # untransformations barcodes to original shape
    sys.stdout.write(f"After Pos: {datapoint['position']} time: {datapoint['time']} , segmentation shape: {seg_pred.shape} -- barcodes_shape: {bboxes_barcode.shape}\n")
    sys.stdout.flush()

    yolo_datapoint = {
        'yolo_size': tuple(param.Analysis.Barcode.img_size),
        'bboxes': bboxes_barcode
    }
    post_barcode_transformations = YoloLiveUnAugmentations(
        parameters = {'resize': {
            'height': raw_shape[0],
            'width': raw_shape[1],
            'always_apply': True
            }
        }
    )
    bboxes_final = post_barcode_transformations(yolo_datapoint)

    #return None

    return { 
        'cells': seg_pred[0],
        'channels': seg_pred[1],
        'barcode_locations': bboxes_final,
        'channel_locations': None,
        'raw_shape': seg_sample['raw_shape']
    }# segmented cells, segmented channels, barcode locations, channel locations
