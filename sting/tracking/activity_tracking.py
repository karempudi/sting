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

@njit
def activtiy_map(mask, diff):
    """
    For mask at time t, diff is the phase[t+1] - phase[t]
    We use this diff to make activity map of each cell in the mask
    """
    activity_mask =  np.zeros(mask.shape)
    active_map = 0.5 * np.abs(diff)
    for i in range(1, int(mask.max()+1)):
        ma = (mask == i) # isolate a single instacne of a cell
        area = np.sum(ma) # sum of all the ones
        cell_activity = np.sum(active_map * ma) / (area + 0.0001) # just to avoid 0 in denominator
        activity_mask += cell_activity * ma
    return activity_mask

def frame_dict(mask):
    """
    For each cell mark the activity and some indices and return a dict.
    Activity field is not filled and will have to be filled once you
    get the next frame.
    """
    frame_dict = {}
    for i in range(1, int(mask.max() + 1)):
        one_cell = (mask == i)
        cell_area = np.sum(one_cell)
        if cell_area > 0:
            cell = {}
            # center of mass
            cell['cm'] = np.mean(np.where(one_cell), axis = 1)
            # activity of the cell is the activity of its center of mass  
            #cell['activity'] = activity_mask[cell['cm'][0].astype(int), cell['cm'][1].astype(int)]
            cell['activity'] = 1.0
            # area
            cell['area'] = cell_area
            cell['mother'] = None
            cell['index'] = None
            cell['dob'] = 0
            cell['initial_mother'] = 0
        frame_dict[i] = cell
    return frame_dict

def set_activities(frame_dict, mask, diff):
    """

    Fill in the activity field once you have enough information that arrives
    in the future
    Arguements:
        frame_dict: dictionary containing information corresponding to the mask
        mask: a labelled mask 
        diff: a phase diff of the corresponding mask
    Returns:
        frame_dict: after filling in the activity field 

    """
    #activity_mask =  np.zeros(mask.shape)
    active_map = 0.5 * np.abs(diff)
    for key in frame_dict:
        ma = (mask == key) # isolate a single instacne of a cell
        area = np.sum(ma) # sum of all the ones
        cell_activity = np.sum(active_map * ma) / (area + 0.0001) # just to avoid 0 in denominator
        #activity_mask += cell_activity * ma
        frame_dict[key] = cell_activity
    return frame_dict
 

@njit
def gaussian_heatmap(center = (2, 2), image_size = (10, 10), sig = 1):
    """
    It produces single gaussian at expected center
    :param center:  the mean position (X, Y) - where high value expected
    :param image_size: The total image size (width, height)
    :param sig: The sigma value
    :return:
    """
    x_axis = np.linspace(0, image_size[0]-1, image_size[0]) - center[0]
    y_axis = np.linspace(0, image_size[1]-1, image_size[1]) - center[1]
    xx, yy = np.meshgrid(x_axis, y_axis)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel.T
 

@njit
def generate_gaussian_maps(frame_dict, image_size, k=2.5):
    """
    For each cell around the center of mass, put a gaussian kernel and return
    the image slice of the correspoinding

    """
    gaussian_maps = {}
    for key in frame_dict:
        gaussian_maps[key] = gaussian_heatmap(center=(int(frame_dict[key]['cm'][0]), int(frame_dict[key]['cm'][1])),
                            image_size=image_size, sig=frame_dict[key]['activity']/k)
    return gaussian_maps

def track_a_bundle(bundle):
    """
        A bundle should be self contained in everything that is need to track 
        2 masks of cells and return a dictionary

    Arguments:
        bundle is a dictionary containing the following keys
            channel_number:
            frame_dict1: updating the frames' activity and the linking states 
            mask1: used to generate gaussian maps based on activities
            mask2: to figure out centroids in frame2
            diff: diff is the phase diff used to assign activities in frame1

    Returns:
        Two frame dictionaries, that are serialized and stored as tracking data
        which has all the links between frames and ways to iterate over the cells
        channel_number: just to help the thread to know which channel to put the resutl in
        frame_dict1 : dictionary of cells in frame1
        frame_dict2 : dictionary of cells in frame2

    """
    # 1. calculate activities of cells in frame1 and fill them in the frame_dict
    # 2. Generate gaussian maps for all the cells in frame1 
    # 3. Iterate over the cells and link them and solve the problem of linking
    #    in 2 stages, first based on just activity and second by solving the splitting problem


    frame_dict2 = None
    channel_number = None
    return (channel_number, frame_dict1, frame_dict2)

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

        # each bundle has the diff of phase and current seg_mask, prev_seg_mask and
        # dictionary object of the previous frame
        self.bundles_to_track = []

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
            if self.timepoint == 0:
                # do something
                # 1. make a frame dict and write the files
                pass

            else:
                # 1. make frame dict for the current mask
                # 2. calculate the diff after getting the previous phase image
                # 3. fetch the previous mask and fill in their activities
                # 4. Put them in a bundle and call tracking functions for all channels
                pass

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

            # find a good way to put together a bundle to track
                    
            
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




    
        


