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

class CellsDictEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def frame_dict(mask):
    """
    For each cell mark the activity and some indices and return a dict.
    Activity field is not filled and will have to be filled once you
    get the next frame.
    """
    img_props = regionprops(mask)
    frame_dict = {}
    for i, props in enumerate(img_props):
        if (props['area'] > 0):
            cell = {}
            cell['area'] = props['area']
            cell['cm'] = props['centroid']
            cell['activity'] = 0
            cell['mother'] = None
            cell['index'] = None
            cell['dob'] = 0
            cell['initial_mother'] = 0
            cell['growth'] = None
            cell['state'] = None
            frame_dict[props['label']] = cell
    return frame_dict

def link_a_bundle(bundle,):



def fast_tracking(tracking_event, param):
    pass