import numpy as np
from scipy import optimize
import scipy
import operator
from numba import njit
import pathlib
from pathlib import Path
import h5py
import pickle
import matplotlib.pyplot as plt
import sys
from sting.utils.disk_ops import write_files
from scipy.optimize import linear_sum_assignment
from skimage.io import imread
from skimage.measure import label, regionprops
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json


np.seterr(divide='ignore', invalid='ignore')


def construct_cellprops(mask, min_area=20):
    """
    For each cell mark the activity and some indices and return a dict.
    Activity field is not filled and will have to be filled once you
    get the next frame.
    Arguments:
        mask: a labelled image slice of a single channel 
                (works for a full image as well)
        min_area: min area of the labelled region in pixels
    Returns:
        frame_dict: a dict with keys as the labels of the region
                and values defined by the keys
    """
    img_props = regionprops(mask)
    frame_dict = {}
    for i, props in enumerate(img_props):
        if (props['area'] > min_area):
            cell = {}
            cell['area'] = int(props['area'])
            cell['cm'] = (float(props['centroid'][0]), float(props['centroid'][1]))
            cell['activity'] = 0
            cell['mother'] = None
            cell['index'] = None
            cell['dob'] = 0
            cell['initial_mother'] = 0
            cell['growth'] = None
            cell['state'] = None
            frame_dict[props['label']] = cell
    return frame_dict


def fast_tracking(tracking_event, param):
    """
    Function that takes in the results of a tracking event, which is
    all the information concerning an image after segmentation,
    barcode detection, channel localization and clean up.
    The job of this function is to plainly do the tracking for 
    all the channels in the image and write data to disk.

    Arguments:
        tracking_event: dict containing all the info of one image
        param: parameters used
    
    """
    
    position = tracking_event['position']
    timepoint = tracking_event['time']

    save_dir = Path(param.Save.directory) if isinstance(param.Save.directory, str) else param.Save.directory
    position_dir = save_dir / Path('Pos' + str(position))

    return None
    