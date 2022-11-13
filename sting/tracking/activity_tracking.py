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




    
        


