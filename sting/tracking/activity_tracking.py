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

        # we will write data to disk here 
        sys.stdout.write(f"Inside activity tracking got data for Position: {self.position} and timepoint: {self.timepoint} id: {id(tracking_event)} ...\n")
        sys.stdout.flush()
        if tracking_event['error'] == True:
            if self.param.Save.save_channels:
                write_files(tracking_event, 'cells_channels', self.param)
            else:
                write_files(tracking_event, 'cells', self.param)
        else:
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




    
        


