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
from skimage.measure import label

def track_one_position(mask, phase):
    # caculate the things needed from one image and paralleize 
    # for all channels of the image

    pass


def gaussian_heatmap(center = (2, 2), image_size = (10, 10), sig = 1):
    """
    It produces single gaussian at expected center
    Arguments:
        param center:  the mean position (X, Y) - where high value expected
        param image_size: The total image size (width, height)
        param sig: The sigma value
    Returns:
        a guassian kernel centersd around center with given sigma
    """
    x_axis = np.linspace(0, image_size[0]-1, image_size[0]) - center[0]
    y_axis = np.linspace(0, image_size[1]-1, image_size[1]) - center[1]
    xx, yy = np.meshgrid(x_axis, y_axis)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel.T

class activityTrackingPosition(object):

    """
    To do activity based tracking you need the following thing, that will be put 
    in the tracking queue to make life easy
        1. current segmentation mask
        2. current phase image
        3. channel locations to grab the correct channel data from the previous frames
        4. a file that is read from disk that contains information from the previous time
            point to link to 
        5.
    Arguments:
        tracking_event: a data dictionary containing all the information needed for tracking
                        one position, the rest will be fetched form the file system, based on
                        position
        track_pos: (default: True), if track_pos is false, only segmentation data will be written
                   and no tracking will be done.

        
    """

    def __init__(self, tracking_event, param, track_pos=True):

        # the keys that are important are 'position', 'time', 'phase', 'cells', 
        # and 'channel_locations'
        self.position = tracking_event['position']
        self.timepoint = tracking_event['time']
        self.param = param

        save_dir = Path(param.Save.directory) if isinstance(param.Save.directory, str) else param.Save.directory
        self.position_dir = save_dir / Path('Pos' + str(self.position))
        # we will write data to disk here 
        sys.stdout.write(f"Inside activity tracking got data for Position: {self.position} and timepoint: {self.timepoint} id: {id(tracking_event)} ...\n")
        sys.stdout.flush()

        # if there is a failure to detect channels, write the data and quit, no tracking is done
        if tracking_event['error'] == True:
            if self.param.Save.save_channels:
                write_files(tracking_event, 'cells_channels', self.param)
            else:
                write_files(tracking_event, 'cells', self.param)
        else:

            # do tracking here
            # read two files here previous full phase_image to calculate the diffs
            # and the tracking channels file, that contains the segmented blobs and other information
                # just write the cut channels to files
            #write_files(tracking_event, 'cells_cut_track_init', self.param)
            self.filename = self.position_dir / Path('cells_tracks.hdf5')
            #self.file_handle = h5py.File(self.filename, 'a')

            cell_prob = param.Analysis.Segmentation.thresholds.cells.probability
            cells_data = (tracking_event['cells'] > cell_prob)
            # for each channel iterate over and create groups and datasets
            channel_locations = []
            for block in tracking_event['channel_locations']:
                channel_locations.extend(tracking_event['channel_locations'][block]['channel_locations'])
            channel_width = param.Save.channel_width

            # we only grab stuff between barcodes and ignore the ends, so this operation will not result in errors
            with h5py.File(self.filename, 'a') as cells_file:
                for i, location in enumerate(channel_locations, 0):
                    img_slice = cells_data[:, max(location-channel_width, 0): 
                                    min(tracking_event['raw_shape'][1], location+channel_width)]
                    # label regions and make them uints for good fast compression
                    img_slice = (label(img_slice) % 255).astype('uint8')
                    # chanel no, cells + _ time.zfil(4)
                    write_string = str(i) + '/cells/cells_' + str(tracking_event['time']).zfill(4)
                    cells_file.create_dataset(write_string, data=img_slice,
                            compression=param.Save.small_file_compression_type,
                            compression_opts=param.Save.small_file_compression_level)
            

            if self.param.Save.save_channels:
                write_files(tracking_event, 'cells_channels', self.param)
            else:
                write_files(tracking_event, 'cells', self.param)

        # check for error if there is one, then you don't have anything to track and 
        # write to a different file instead of writing to the main line and corrupt 
        # tracking, if there is more than a few frames missing, we write by default
        # to the error save files instead of the main line

    def track(self,):
        # run a bunch of threads and start a tracking of all the channels
        # pool the results into one file per position
        if self.timepoint == 0:
            # do something
            pass
        else:
            # do something
            pass
    
    def load_required_objects(self,):
        pass
        


class activtiyTrackingChannel(object):

    def __init__(self, position, channel):
        pass



def track_one_channel(mask1, mask2, activity1, activity2):
    """
    Function that tracks objects between 2 frames and returns a dictionary
    containing some results about the cells linked between these two frames

    Arguments:
        mask1: mask at time 't'
        mask2: mask at time  't+1'
        activity1: activity mask at time 't'
        activity2: activity mask at time 't+1'
    Returns
        tracking_results: dictionary
    """
    frame_dict1 = {}
    # for each cell in frame t make a dictionary object for each cell
    for i in range(1, int(mask1.max() + 1)):
        # if there are enough pixels, here you can write cell size cutoffs already
        one_cell = (mask1 == i)
        area = np.sum(one_cell)
        if area > 0:
            cell = {}
            # center of mass
            cell['cm'] = np.mean(one_cell, axis=1)
            # get activity of the center of mass pixel
            cell['activity'] = activity1[cell['cm'][0].astype(int), cell['cm'][1].astype(int)]
            # area
            cell['area'] = area
            cell['mother'] = None
            cell['index'] = None
            cell['dob'] = 0
            cell['initial_mother'] = 0
            frame_dict1[i] = cell
        
    frame_dict2 = {}

    # for each cell in frame t+1 make a dictionary object for each cell
    for i in range(1, int(mask2.max() + 1)):
        # if there are enough pixels, here you can write cell size cutoffs already
        one_cell = (mask2 == i)
        area = np.sum(one_cell)
        if area > 0:
            cell = {}
            # center of mass
            cell['cm'] = np.mean(one_cell, axis=1)
            # get activity of the center of mass pixel
            cell['activity'] = activity2[cell['cm'][0].astype(int), cell['cm'][1].astype(int)]
            # area
            cell['area'] = area
            cell['mother'] = None
            cell['index'] = None
            cell['dob'] = 0
            cell['initial_mother'] = 0
            frame_dict2[i] = cell

    # sort cells in the curretn frame based on activity
    new_index = sorted(frame_dict1, key=lambda x: frame_dict1[x]['activity'])

    # arrays to kept track of cells already linked
    leftover1 = list(new_index)
    lefover2 = list(frame_dict2.keys())




    
        


