import torch
import numpy as np
from sting.utils.types import RecursiveNamespace
from typing import Union
from sting.mm.networks import LiveNet
from sting.mm.utils import YoloLiveAugmentations, YoloLiveUnAugmentations
from sting.segmentation.transforms import UnetTestTransforms
from sting.regiondetect.utils import non_max_suppression, to_cpu, outputs_to_bboxes
import sys
from scipy.signal import find_peaks, peak_prominences


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

def get_locations_btn_barcodes(channel_img, bbox_pred, param, raw_shape):
    """

    This will take the channel_img, barcode_bboxes and gets the locations of the 
    channels to cut channels stacks out of the image
    Arguments:
        channel_img: channel segmentation img
        bbox_pred: bbox predictions from the net that were cleaned
        param: parameters used that continals things more things to clean up 
               in the barcode segment
        raw_shape: raw shape of the phase contrast image shot

    """
    bbox_centers = np.array(sorted([(bbox[0] + bbox[2])/2 for bbox in bbox_pred]))
    distance_bboxes = np.diff(bbox_centers)
    bbox_ok = np.logical_and(distance_bboxes > param.Analysis.Barcode.dist_thresholds.min,
                            distance_bboxes < param.Analysis.Barcode.dist_thresholds.max)
    all_bbox_ok = np.all(bbox_ok)
    n_good_bboxes = sum(bbox_ok)
    if n_good_bboxes >= 1:
        first_idx = np.where(bbox_ok==True)[0][0]
        first_good_bbox = bbox_centers[first_idx]
    else:
        return None, True
    ideal_dist = param.Analysis.Barcode.dist_thresholds.dist
    constructed_bboxes = np.concatenate((np.arange(first_good_bbox, 0, -ideal_dist),
                                        np.arange(first_good_bbox+ideal_dist, raw_shape[1], ideal_dist)))
    # calculate threshold on the bbox error here accurately
    # there should be only one
    #constructed_good_idx = np.where(constructed_bboxes==first_good_bbox)[0][0]
    # we will use constructed bboxes by default
    final_bbox_centers = constructed_bboxes.astype('int')
    forbidden = [] # channels' centers can't be in this indices as they are take by barcode
    barcode_regions = [forbidden.extend(list(range(center-param.Analysis.Barcode.dist_thresholds.size, 
                                                  center+param.Analysis.Barcode.dist_thresholds.size))) 
                           for center in final_bbox_centers]
    
    hist = np.sum(channel_img, axis=0)
    peaks, _ = find_peaks(hist, distance=param.Analysis.Barcode.dist_thresholds.channel_dist)
    prominences, _, _ = peak_prominences(hist, peaks)
    peaks = peaks[prominences > param.Analysis.Barcode.dist_thresholds.prominences]
    first_bbox = final_bbox_centers[0]
    last_bbox = final_bbox_centers[-1]
    peaks = [peak for peak in peaks if (peak > first_bbox and peak < last_bbox)]
    peaks_np = np.array(peaks)
    num_channels = len(peaks_np)
    btn_barcodes = list(zip(final_bbox_centers[:-1], final_bbox_centers[1:]))
    channels_btn_barcode = {}
    for i, r in enumerate(btn_barcodes, 0):
        channels_btn_barcode[i] = {}
        valid_channels = np.logical_and(peaks_np > r[0], peaks_np < r[1])
        channels_btn_barcode[i]['num_channels'] = np.sum(valid_channels)
        channels_btn_barcode[i]['channel_locations'] = peaks_np[valid_channels]
    
    return channels_btn_barcode, False

def process_image(datapoint, model, param):
    """
    Function to process one datapoint in the live analysis pipeline
    Arguments:
        datapoint: a dict with keys 'image', 'time', 'position',
        model: an instance of live net model loaded on device

    """
    try:
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
        bboxes_final = sorted(bboxes_final, key=lambda x: x[0]) # sort according to top left corner
        #print(bboxes_final)
        #return None
        channel_locations, error = get_locations_btn_barcodes(seg_pred[1], bboxes_final, param, raw_shape)
        total_channels = 0
        list_channel_locations = []
        for block in channel_locations:
            total_channels += channel_locations[block]['num_channels']
            list_channel_locations.extend(channel_locations[block]['channel_locations'].tolist())

        sys.stdout.write(f"After Pos: {datapoint['position']} time: {datapoint['time']} , no ch: {total_channels} ... \n")
        sys.stdout.flush()

        return { 
            'phase': datapoint['image'],
            'position': datapoint['position'],
            'time': datapoint['time'],
            'cells': seg_pred[0][:raw_shape[0], :raw_shape[1]],
            'channels': seg_pred[1][:raw_shape[0], :raw_shape[1]],
            'barcode_locations': bboxes_final,
            'channel_locations': channel_locations,
            'channel_locations_list': list_channel_locations,
            'raw_shape': seg_sample['raw_shape'],
            'total_channels': total_channels,
            'error': error # if error is true we are going to skip the position
        }# segmented cells, segmented channels, barcode locations, channel locations
    except Exception as e:
        sys.stdout.write(f"Error {e} in process image function at position: {datapoint['position']} - time: {datapoint['time']}\n")
        sys.stdout.flush()
        return {
            'phase': datapoint['image'],
            'position': datapoint['position'],
            'time': datapoint['time'],
            'total_channels': -1,
            'error': True
        }
