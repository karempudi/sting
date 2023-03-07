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
from skimage.measure import label, regionprops
import time


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
    print(final_bbox_centers)
    forbidden = [] # channels' centers can't be in this indices as they are take by barcode
    barcode_regions = [forbidden.extend(list(range(center-param.Analysis.Barcode.dist_thresholds.size, 
                                                  center+param.Analysis.Barcode.dist_thresholds.size))) 
                           for center in final_bbox_centers]
    
    hist = np.sum(channel_img, axis=0)
    peaks, _ = find_peaks(hist, distance=param.Analysis.Barcode.dist_thresholds.channel_dist)
    prominences, _, _ = peak_prominences(hist, peaks)
    peaks = peaks[prominences > param.Analysis.Barcode.dist_thresholds.prominences]
    print(peaks)
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


def get_channel_locations(channel_img, bboxes_final, param, raw_shape):
    bbox_centers_confidences = [((bbox[0] + bbox[2])/2, bbox[4]) for bbox in bboxes_final]
    bbox_centers_confidences = sorted(bbox_centers_confidences, key=lambda x:x[0])
    bbox_centers = np.array([x[0] for x in bbox_centers_confidences])
    bbox_confidences = np.array([x[1] for x in bbox_centers_confidences]) 
    distance_bboxes = np.diff(bbox_centers)
    block_ok = np.logical_and(distance_bboxes > param.Analysis.Barcode.dist_thresholds.min,
                                distance_bboxes < param.Analysis.Barcode.dist_thresholds.max)
    
    # TODO: use these to convey error status later
    all_blocks_ok = np.all(block_ok)
    n_possible_good_blocks = sum(block_ok)

    broken_blocks = np.where(block_ok == False)[0]

    corrected_bboxes = bbox_centers.copy()
    for broken_idx in broken_blocks:
        left_center, right_center = bbox_centers[broken_idx], bbox_centers[broken_idx+1]
        left_conf, right_conf = bbox_confidences[broken_idx], bbox_confidences[broken_idx+1]
        if left_conf >= right_conf:
            corrected_bboxes[broken_idx+1] = min(corrected_bboxes[broken_idx] + param.Analysis.Barcode.dist_thresholds.dist,
                                                 raw_shape[1])
        else:
            corrected_bboxes[broken_idx] = max(corrected_bboxes[broken_idx+1] - param.Analysis.Barcode.dist_thresholds.dist, 
                                                0)
    
    #print(f"After correcttions: {corrected_bboxes}")
    final_bboxes = np.concatenate((np.sort(np.arange(corrected_bboxes[0], 0, -param.Analysis.Barcode.dist_thresholds.dist)),
                                    corrected_bboxes[1:-1],
                                   np.sort(np.arange(corrected_bboxes[-1], raw_shape[1], param.Analysis.Barcode.dist_thresholds.dist))))
    #print(f"Final bboxes: {final_bboxes}")
    channel_img = channel_img > param.Analysis.Segmentation.thresholds.channels.probability
    hist = np.sum(channel_img, axis=0)
    #print(hist)
    #peaks, props = find_peaks(hist, distance=param.Analysis.Barcode.dist_thresholds.channel_dist)
    peaks, props = find_peaks(hist, prominence=param.Analysis.Barcode.dist_thresholds.prominences, 
                                distance=param.Analysis.Barcode.dist_thresholds.channel_dist/1.5)
    #prominences, _, _ = peak_prominences(hist, peaks, wlen=2*param.Analysis.Barcode.dist_thresholds.channel_dist)
    prominences = props['prominences']
    #print(peaks, prominences)
    peaks = peaks[prominences > param.Analysis.Barcode.dist_thresholds.prominences]
    
    btn_barcodes = []
    btn_barcodes.append((0, final_bboxes[0]))
    btn_barcodes.extend(list(zip(final_bboxes[:-1], final_bboxes[1:])))
    btn_barcodes.append((final_bboxes[-1], raw_shape[1]))

    channels_btn_barcode = {}
    for i, (b_l, b_r) in enumerate(btn_barcodes, 0):
        channels_btn_barcode[i] = {}
        valid_channels = np.logical_and(peaks > b_l, peaks < b_r)
        channels_btn_barcode[i]['num_channels'] = np.sum(valid_channels)
        channels_btn_barcode[i]['channel_locations'] = peaks[valid_channels]
    #print(channels_btn_barcode)
    
    return channels_btn_barcode, False


def get_channel_locations_corr(channel_img, bboxes_final, param, raw_shape, prev_channels):
    pass

def cut_channels_and_props(image, raw_shape, channel_locations, channel_width, min_area=20):
    """
    A function that takes a segmented binary mask and returns labelled images and 
    properties that are pushed to the tracking queue for cell-tracking
    """
    n_channels = len(channel_locations)
    height, width = raw_shape[0], raw_shape[1]
    labelled_slices = np.zeros((height, 2*channel_width*n_channels), dtype='uint8')
    props = {}
    for i, location in enumerate(channel_locations, 0):
        sliced_img = label(image[:, location-channel_width:location+channel_width])
        labelled_slices[:, i * 2 * channel_width: (i+1) * 2 * channel_width] = sliced_img
        props_slice = regionprops(sliced_img)
        props[str(i)] = {}
        for cell_i, properties in enumerate(props_slice):
            if (properties['area']) > min_area:
                cell = {}
                cell['area'] = int(properties['area'])
                cell['cm'] = (float(properties['centroid'][0]), float(properties['centroid'][1]))
                cell['bbox'] = properties['bbox']
                cell['activity'] = 0
                cell['mother'] = None
                cell['index'] = None
                cell['dob'] = 0
                cell['initial_mother'] = 0
                cell['growth'] = None
                cell['state'] = None
                props[str(i)][str(properties['label'])] = cell
    return labelled_slices, props

def process_image(datapoint, model, param, visualize=True):
    """
    Function to process one datapoint in the live analysis pipeline
    Arguments:
        datapoint: a dict with keys 'image', 'time', 'position',
        model: an instance of live net model loaded on device
        param: parameters used
        visualize: To get full results for plotting, set visualize to True. Default
                   is to chop up the image into slices and label each channel slice 
                   to avoid doing it in the tracking process.
    """
    start_time = time.time()
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

        yolo_img_size = tuple(param.Analysis.Barcode.img_size)

        # cleaning up bbox predictions that are outside the size of the image
        # can happen as the net projects outward if the barcodes are at the edge
        # of the image
        for bbox in bboxes_barcode:
            if bbox[0] < 0.0:
                bbox[0] = 0.0
            if bbox[2] > yolo_img_size[1]:
                bbox[2] = yolo_img_size[1]
            if bbox[1] < 0.0:
                bbox[1] = 0.0
            if bbox[3] > yolo_img_size[0]:
                bbox[3] = yolo_img_size[0]

        yolo_datapoint = {
            'yolo_size': yolo_img_size,
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
        bboxes_final = sorted(bboxes_final, key=lambda x: x[0]) # sort according to top left corner in x axis
        #print(bboxes_final)
        #return None
        #channel_locations, error = get_locations_btn_barcodes(seg_pred[1], bboxes_final, param, raw_shape)
        channel_locations, error = get_channel_locations(seg_pred[1], bboxes_final, param, raw_shape)
        total_channels = 0
        list_channel_locations = []
        for block in channel_locations:
            n_channels = channel_locations[block]['num_channels']
            if n_channels > 10:
                total_channels += channel_locations[block]['num_channels']
                list_channel_locations.extend(channel_locations[block]['channel_locations'].tolist())


        cell_prob = param.Analysis.Segmentation.thresholds.cells.probability

        if visualize:

            duration = 1000 * (time.time() - start_time)
            sys.stdout.write(f"Seg Pos: {datapoint['position']} time: {datapoint['time']} , no ch: {total_channels}, duration: {duration:0.4f}ms ...\n")
            sys.stdout.flush()
            return { 
                #'phase': datapoint['image'].astype(),
                'phase': datapoint['image'].astype('uint16'),
                'position': datapoint['position'],
                'time': datapoint['time'],
                #'cells': seg_pred[0][:raw_shape[0], :raw_shape[1]],
                'cells': (seg_pred[0][:raw_shape[0], :raw_shape[1]] > cell_prob),
                'channels': seg_pred[1][:raw_shape[0], :raw_shape[1]],
                #'channels': None,
                'barcode_locations': bboxes_final,
                'channel_locations': channel_locations,
                'channel_locations_list': list_channel_locations,
                'raw_shape': seg_sample['raw_shape'],
                'total_channels': total_channels,
                'error': error # if error is true we are going to skip the position
            }# segmented cells, segmented channels, barcode locations, channel locations
        else:
            # Here we do a chopped up labelled version of each channel slice to avoid 
            # doing it in the tracking pipeline
            cells_binary = seg_pred[0][:raw_shape[0], :raw_shape[1]] > cell_prob
            labelled_slices, channel_props = cut_channels_and_props(cells_binary, raw_shape, list_channel_locations, param.Save.channel_width)

            duration = 1000 * (time.time() - start_time)
            sys.stdout.write(f"Seg Pos: {datapoint['position']} time: {datapoint['time']} , no ch: {total_channels}, duration: {duration:0.4f}ms ...\n")
            sys.stdout.flush()
 
            return {
                'position': datapoint['position'],
                'time': datapoint['time'],
                'labelled_slices': labelled_slices,
                'props': channel_props, 
                'barcode_locations': bboxes_final,
                'total_channels': total_channels,
                'channel_locations_list': list_channel_locations,
                'raw_shape' : raw_shape,
                'error': error
            }
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
