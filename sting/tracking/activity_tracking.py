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
import json


class CellsDictEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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
            frame_dict[props['label']] = cell
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
    # loop over cells
    for key in frame_dict:
        ma = (mask == int(key)) # isolate a single instacne of a cell
        area = frame_dict[key]['area']# sum of all the ones
        cell_activity = np.sum(active_map * ma) / (area + 0.0001) # just to avoid 0 in denominator
        #activity_mask += cell_activity * ma
        frame_dict[key]['activity'] = cell_activity
        #sys.stdout.write(f"Activity: of {key} is {area} -- {np.sum(ma)} \n")
        #sys.stdout.flush()
    #for key in frame_dict:
    #    sys.stdout.write(f"{key} --> {frame_dict[key]['activity']} ..")
    #sys.stdout.write("\n")
    #sys.stdout.flush()
    return frame_dict

def find_max_index(cells_frame_dict) :
    cell_indices = []
    for cell_no in cells_frame_dict.keys():
        if cells_frame_dict[cell_no]['index'] is not None:
            cell_indices.append(cells_frame_dict[cell_no]['index'])
    #print(cell_indices)
    if len(cell_indices) == 0:
        return 0
    else:
        return np.max(cell_indices)

def gaussian_heatmap(center = (2, 2), image_size = (10, 10), sigma = 1):
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
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel.T

def gaussian_heatmap_cell(center=(2, 2), image_size=(10, 10), sigma=(1.0, 2.0)):
    """
    This function produces gauissians that have different sigmas, the activity 
    sigma is used for the x-axis (row numbers) and we keep constant sigma on
    the y-axis (column numbers)
    """
    x_axis = np.linspace(0, image_size[0] - 1, image_size[0]) - center[0]
    y_axis = np.linspace(0, image_size[1] - 1, image_size[1]) - center[1]
    xx, yy = np.meshgrid(x_axis, y_axis)
    kernel = np.exp(-0.5 * ((np.square(xx)/np.square(sigma[0])) + (np.square(yy)/np.square(sigma[1]))))
    return kernel.T

def generate_gaussian_maps(frame_dict, image_size, k=50.0):
    """
    For each cell around the center of mass, put a gaussian kernel and return
    the image slice of the correspoinding

    """
    gaussian_maps = {}
    for key in frame_dict:
        gaussian_maps[key] = gaussian_heatmap_cell(center=(int(frame_dict[key]['cm'][0]), int(frame_dict[key]['cm'][1])),
                            image_size=image_size, sigma=(frame_dict[key]['activity']/k, image_size[1]/2))
    return gaussian_maps
 

def track_a_bundle(bundle, area_thres=0.75):
    """
        A bundle should be self contained in everything that is need to track 
        2 masks of cells and return a dictionary

    Arguments:
        bundle is a dictionary containing the following keys
            channel_number:
            frame_dict1: updating the frames' activity and the linking states 
            frame_dict2: 
            frame1: used to generate gaussian maps based on activities
            frame2: to figure out centroids in frame2
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

    img1 = bundle['frame1']
    img2 = bundle['frame2']
    channel_number = bundle['channel_no']
    frame_dict1 = bundle['frame_dict1']
    frame_dict2 = bundle['frame_dict2']
    frame1_no = bundle['t1']
    frame2_no = bundle['t2']

    # Startig the tracking function
    none_keys = []
    adjacent_dict = {}
    # sort the indices of the first frame based on activity
    new_index = sorted(frame_dict1, key=lambda x: frame_dict1[x]['activity'])
    leftover1 = list(new_index)
    leftover2 = list(frame_dict2.keys())
    # generate gaussian heatmap for all cells in frame1,
    # gm will have the same keys as frame_dict1
    gm = generate_gaussian_maps(frame_dict1, img1.shape, k=50.0)
    
    # setting the max index
    max_prev_cell_index = find_max_index(frame_dict1)
    if max_prev_cell_index == 0:
        # no tracking if there are not cells in previous frame
        return (channel_number, frame_dict1, frame_dict2)
    #print(channel_number, max_prev_cell_index)
    max_max = [max_prev_cell_index + 1]
    
    # stage 1: link cells with highest activity if they meet area threshold
    for key1 in new_index:
        # has all the possible 
        subdic = {}
        for key2 in leftover2:
            pr = gm[key1][int(frame_dict2[key2]['cm'][0]), int(frame_dict2[key2]['cm'][1])]
            if pr > 0.01:
                subdic[key2] = pr
        # all possible cells that could be connnected to key1
        #print(key1, "----> ", subdic)
        #print("---------------------")
        
        if len(subdic) == 0:
            # no cells cross the threshold of probablity in frame2, so we don't link
            # this cell to anything in frame 2
            leftover1.remove(key1)
        else:
            # find the cell that matches with the maximum probability
            key2 = max(subdic, key=subdic.get)
            # check if the area threshold matches
            # if the cell in frame2 has atleast area_thres fraction of the area in frame1
            if area_thres * frame_dict1[key1]['area'] < frame_dict2[key2]['area']:
                # if the area is increasing match
                #print(f"Area frame1: {frame_dict1[key1]['area']} ---> frame2: {frame_dict2[key2]['area']}")
                # now adjsut the indices of the cells
                if frame_dict1[key1]['index'] is not None:
                    # if it is not None, it is assigned a number
                    # indices of cells are global and are assigned at birth 
                    # until death
                    # mother is the index of the cell it is connected
                    frame_dict2[key2]['mother'] = 1 * frame_dict1[key1]['index']
                    frame_dict2[key2]['index'] = frame_dict2[key2]['mother']
                    # when was the cell born
                    frame_dict2[key2]['dob'] = frame_dict1[key1]['dob']
                    frame_dict2[key2]['initial_mother'] = frame_dict1[key1]['initial_mother']
                    max_max.append(frame_dict1[key1]['index'])
                else:
                    # if for some reason, the cell in the previous frame is not assigned
                    none_keys.append(key2)
                
                # remove the maximum from frame2
                leftover2.remove(key2)
                # remove the cell from the first frame as well, if there are daughters we will
                # see it below
                leftover1.remove(key1)
                
            # sort the list of possible candidates still available + the linked one if you linked
            subdic = sorted(subdic.items(), key=operator.itemgetter(1), reverse=True)
            # add the possible adjacent cells in increasing probability 
            # subdic has [key2]-> [activity]
            adjacent_dict[key1] = dict(subdic)
    
    # stage2: we loop over and remove things that are already linked
    #print("Stage2: ------------")
    #print("Adjacent dict: ", adjacent_dict)
    #print("Leftover1: ", leftover1)
    #print("Leftover2: ", leftover2)
    # This is sort of irrelevant to do, as we don't alter leftover1 or leftover 2
    # stage2 can be scrapped in my opinion
    for key1 in adjacent_dict.copy().keys():
        # remove everything that is assigned in first frame
        #print("Looking at key1: ", key1)
        if key1 not in leftover1:
            del adjacent_dict[key1]
        # if itis not assigned, remove everything that is already assigned
        # in frame2 for this cell
        else:
            for key2 in leftover2:
                if key2 not in leftover2:
                    #print("Deleting: ", key1 , "---", key2)
                    del adjacent_dict[key1][key2]
    
            
    # stage 3: linking daughter and splits
    daughters = leftover2
    mothers = leftover1
    mothers_twice = mothers + mothers # double so that each mother can have 2 daughters at most
    C = np.zeros([len(mothers_twice), len(daughters)])
    # fill in the cost matrix 
    row = 0
    for mother_idx in mothers_twice:
        column = 0
        for daughter_idx in daughters:
            if daughter_idx in set(adjacent_dict[mother_idx].keys()):
                C[row, column] = 1 - adjacent_dict[mother_idx][daughter_idx]
            else:
                C[row, column] = 1
        column += 1
    row += 1
    
    # solve the linear assignment problem, this will give indices into 
    # mothers_twice and daughters
    pairs_mother, pairs_daughter = linear_sum_assignment(C)
    max_max = np.max(max_max) + 1 # not sure why we are adding 1 here, but will see if needed
    found = []
    for i in range(len(pairs_mother)):
        frame_dict2[daughters[pairs_daughter[i]]]['mother'] = frame_dict1[mothers_twice[pairs_mother[i]]]['index']
        frame_dict2[daughters[pairs_daughter[i]]]['dob'] = frame2_no
        frame_dict2[daughters[pairs_daughter[i]]]['index'] = max_max
        frame_dict2[daughters[pairs_daughter[i]]]['initial_mother'] = frame_dict1[mothers_twice[pairs_mother[i]]]['index']
        frame_dict1[mothers_twice[pairs_mother[i]]]['dod'] = True
        found.append(daughters[pairs_daughter[i]])
        max_max += 1
    
    # stage 4: clean up of unassigned
    still_leftover2 = set(leftover2).difference(set(found))
    none_keys  += list(still_leftover2)
    for k in none_keys:
        frame_dict2[k]['index'] = max_max
        max_max += 1
        

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
        #self.bundles_to_track = []

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
            # threshold cells_data 
            cell_prob = param.Analysis.Segmentation.thresholds.cells.probability
            cells_data = (tracking_event['cells'] > cell_prob)
 
            # aggregate channel locations
            channel_locations = []
            for block in tracking_event['channel_locations']:
                channel_locations.extend(tracking_event['channel_locations'][block]['channel_locations'])
            channel_width = param.Save.channel_width

            filename = self.position_dir / Path('cells_tracks.hdf5')
            # do tracking here
            # read two files here previous full phase_image to calculate the diffs
            if self.timepoint == 0:
                # do something
                # 1. make a frame dict and write the files
                with h5py.File(filename, 'a') as cells_file:
                    write_string_phase = '/prev_phase'
                    cells_file.create_dataset(write_string_phase, data=tracking_event['phase'].astype('uint16'))
                    for i, location in enumerate(channel_locations, 0):
                        img_slice = cells_data[:, max(location-channel_width, 0): 
                                        min(tracking_event['raw_shape'][1], location+channel_width)]
                        # label regions and make them uints for good fast compression
                        img_slice = (label(img_slice) % 255).astype('uint8')
                        # calculate properties, everything other than activities, which will
                        # be filled when you get the next frame
                        img_slice_dict = frame_dict(img_slice)
                        # write cell number for t = 0 frame
                        for cell_no in img_slice_dict:
                            img_slice_dict[cell_no]['index'] = int(cell_no)
                        # write the slice and the dictionary
                        write_string_slice =  str(i) + '/cells/cells_' + str(tracking_event['time']).zfill(4)
                        write_string_dict = str(i) + '/tracks/tracks_' + str(tracking_event['time']).zfill(4)
                        cells_file.create_dataset(write_string_slice, data=img_slice, 
                                compression=param.Save.small_file_compression_type,
                                compression_opts=param.Save.small_file_compression_level)
                        cells_file.create_dataset(write_string_dict, data=json.dumps(img_slice_dict, cls=CellsDictEncoder))

            else:
                # 1. make frame dict for the current mask
                # 2. calculate the diff after getting the previous phase image
                # 3. fetch the previous mask and fill in their activities
                # 4. Put them in a bundle and call tracking functions for all channels

                #write_files(tracking_event, 'cells_cut_track_init', self.param)

                # calculate the diff of phase images here
                # read the previous phase image
                prev_time_str = str(self.timepoint-1).zfill(4)
                #phase_img_filename = self.position_dir / Path('phase') / Path('phase_' + prev_time_str + '.tiff')
                #prev_phase_img = imread(phase_img_filename).astype('float32')
                #phase_diff = tracking_event['phase'] - prev_phase_img

                bundles = [] 

                with h5py.File(filename, 'a') as cells_file:
                    # read the prev_phase
                    prev_phase_img = cells_file['/prev_phase'][()]
                    phase_diff = tracking_event['phase'] - prev_phase_img
                    # we only grab stuff between barcodes and ignore the ends, so this operation will not result in errors
                    for i, location in enumerate(channel_locations, 0):
                        bundle_item = {}
                        img_slice2 = cells_data[:, max(location-channel_width, 0): 
                                        min(tracking_event['raw_shape'][1], location+channel_width)]
                        # label regions and make them uints for good fast compression
                        img_slice2 = (label(img_slice2) % 255).astype('uint8')
                        read_string_slice1 = str(i) + '/cells/cells_' + prev_time_str
                        read_string_dict1 = str(i) + '/tracks/tracks_' + prev_time_str
                        img_slice1 = cells_file.get(read_string_slice1)[()]
                        img_slice_dict1 = json.loads(cells_file.get(read_string_dict1)[()])
                        img_slice_dict2 = frame_dict(img_slice2)
                        diff_slice = phase_diff[:, max(location-channel_width, 0):
                                        min(tracking_event['raw_shape'][1], location+channel_width)]
                    
                        img_slice_dict1 = set_activities(img_slice_dict1, img_slice1, diff_slice)

                        # Now you have everything you need to track a slice
                        # put them in somewhere and track them
                        bundle_item = {
                            't1': self.timepoint-1,
                            't2': self.timepoint,
                            'channel_no': i,
                            'frame1': img_slice1,
                            'frame2': img_slice2,
                            'frame_dict1': img_slice_dict1,
                            'frame_dict2': img_slice_dict2,
                            'diff': diff_slice
                        }
                        (channel_no, img_slice_dict1, img_slice_dict2) = track_a_bundle(bundle_item)
                        # chanel no, cells + _ time.zfil(4)
                        write_string_slice = str(i) + '/cells/cells_' + str(tracking_event['time']).zfill(4)
                        write_string_dict = str(i) + '/tracks/tracks_' + str(tracking_event['time']).zfill(4)
                        write_string_dict_prev = str(i) + '/tracks/tracks_' + prev_time_str
                        cells_file.create_dataset(write_string_slice, data=img_slice2,
                                compression=param.Save.small_file_compression_type,
                                compression_opts=param.Save.small_file_compression_level)
                        cells_file.create_dataset(write_string_dict, data=json.dumps(img_slice_dict2, cls=CellsDictEncoder))

                        # overwrite the cells dict with activities for now
                        #if i == 0:
                        #    sys.stdout.write(f"{img_slice_dict1} ... {diff_slice}\n")
                        #    sys.stdout.flush()

                        cells_file[write_string_dict_prev][()] = json.dumps(img_slice_dict1, cls=CellsDictEncoder)
                        
                        bundles.append(bundle_item)
                    cells_file['/prev_phase'][...] = tracking_event['phase']
                        
            
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




    
        


