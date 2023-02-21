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
import time
import zarr
from numcodecs import Zlib


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


def fast_tracking_cells(tracking_event, param):
    """
    Function that takes in the results of a tracking event, which is
    all the information concerning an image after segmentation,
    barcode detection, channel localization and clean up.
    The job of this function is to plainly do the tracking for 
    all the channels in the image and write data to disk.

    Arguments:
        tracking_event: dict containing all the info of one image, position and time
        param: parameters used
    
    """
    
    start_time = time.time()
    position = tracking_event['position']
    timepoint = tracking_event['time']
    props = tracking_event['props']
    image = tracking_event['labelled_slices']
    height, width = image.shape
    n_channels = len(tracking_event['channel_locations_list'])


    save_dir = Path(param.Save.directory) if isinstance(param.Save.directory, str) else param.Save.directory
    position_dir = save_dir / Path('Pos' + str(position))

    cells_filename = position_dir / Path('cells.zarr')
    tracks_dir = position_dir / Path('tracks')
    compressor = Zlib(level=param.Save.small_file_compression_level)

    if not tracks_dir.exists():
        tracks_dir.mkdir(exist_ok=True, parents=True)
    
    tracks_write_filename = tracks_dir / Path('tracks_' + str(timepoint).zfill(4) + '.json')
    with open(tracks_write_filename, 'w') as tracks_fh:
        tracks_fh.write(json.dumps(props))

   
    if timepoint == 0:
        # we create a file with appropriate compression and chunking and write the first image
        cells_array = zarr.convenience.open(cells_filename, mode='a', shape=(1, height, width),
                        chunks=(1, height, 2*param.Save.channel_width), order='C', 
                        dtype='uint8', compressor=compressor)
        cells_array[0] = image

    else:
        cells_array = zarr.convenience.open(cells_filename, mode='a', shape=(1, height, width),
                chunks=(1, height, 2*param.Save.channel_width), order='C', 
                dtype='uint8', compressor=compressor)
        cells_array.append(image[np.newaxis, :])

        # read the tracks of the previous file so that you can modify the attributes
        # of cells in the previous frame, (for now we test read times and write)
        prev_track_filename = tracks_dir / Path('tracks_' + str(timepoint-1).zfill(4) + '.json')
        with open(prev_track_filename, 'r') as prev_track_fh:
            prev_track_data = json.load(prev_track_fh) 

    duration = 1000 * (time.time() - start_time)
    sys.stdout.write(f"Tracking Pos: {tracking_event['position']} time: {tracking_event['time']} , no ch: {len(props)}, duration: {duration:0.4f}ms ...\n")
    sys.stdout.flush()

    return None
    