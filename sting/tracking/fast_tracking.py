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
import copy
import time
import zarr
from numcodecs import Zlib


np.seterr(divide='ignore', invalid='ignore')

def jsonKeys2int(x):
    if isinstance(x, dict):
        return {int(k): v for k, v in x.items()}
    return x

def intersection_scores(props1, props2, exponent=1):
    """
    Gives a scoring matrix that is useful for linking objects,
    This uses a one dimensional bbox overlaps, as we don't align
    images to account for drifts.
    The score is a one dimensional jaccard index (overlap/Union)**exponent
    """
    n_detections1, n_detections2 = len(props1), len(props2)
    #scores = []
    score_arr = np.zeros((n_detections1, n_detections2))
    for i, key1 in enumerate(props1, 0):
        for j, key2 in enumerate(props2, 0):
            bbox1 = props1[key1]['bbox']
            bbox2 = props2[key2]['bbox']
            len1 = props1[key1]['bbox'][2] - props1[key1]['bbox'][0]
            len2 = props2[key2]['bbox'][2] - props2[key2]['bbox'][0]
            x1 = max(bbox1[0], bbox2[0])
            x2 = min(bbox1[2], bbox2[2])
            if (x2 < x1):
                #score_arr[i, j] = 0.0
                jaccard_score = 0
            else:
                overlap = abs(x1 - x2)
                #score_arr[i, j] = (overlap / (len1 + len2 - overlap))**exponent
                jaccard_score = (overlap / (len1 + len2 - overlap)) ** exponent
            #scores.append([key1, key2, jaccard_score])
            score_arr[i, j] = jaccard_score
    #score_arr = np.asarray(scores)
    # converting score_arr to probabilities
    scores_sum = np.sum(score_arr, axis=1)[:, None]
    score_probs = np.divide(score_arr, scores_sum, out=np.zeros_like(score_arr), where=scores_sum!=0.0)
    return score_probs

def find_max_index(props):
    cell_indices = []
    # use this but there could be None's that need to be checkout, we can do it later
    #max_index = max(props.items(), key=lambda x: x[1]['index'])[1]['index']
    for cell_no in props:
        if props[cell_no]['index'] is not None:
            cell_indices.append(props[cell_no]['index'])
    if len(cell_indices) == 0:
        return 0
    else:
        return max(cell_indices)

def link_two_frames(props1, props2, frame1_no=0, frame2_no=1, area_thres=0.70, jacc_thres=0.15,):
    """
    Function that add the links and changes the properties dictionaries
    that are provided as input, and add the links, states, growth 
    and other things.
    
    Arguments:
        props1 (dict): properties dict of each cell to keep track of at time t
        props2 (dict): properties dict of each cell to keep track of at time t+1
    Returns:
        props1 (dict): update cell properties dict which includes links, growth rates, etc at time t
        props2 (dict): update cell properties dict which includes links, growth rates, etc at time t+1

    """
    scores = intersection_scores(props1, props2)
    
    # tracking start
    none_keys = []
    adjacent_dict = {}
    key1_copy = list(zip(np.arange(len(props1)), props1.keys()))
    leftover1 = copy.deepcopy(key1_copy)
    leftover2 = list(zip(np.arange(len(props2)), props2.keys())) # used for indexing into the score matrix
    
    # max_cell_index in the previous frame
    max_prev_cell_index = find_max_index(props1)
    #print(max_prev_cell_index)
    if max_prev_cell_index == 0:
        # This occurs if there are no cells in the previous frame,
        # or for whatever reason, they are empty.
        # In this case we make a judgement to start the cell
        # indices at the beginning in frame t+1 and set it
        # to the initial conditions at t = 0
        for key2 in props2:
            props2[key2]['index'] = int(key2)
        return props1, props2

    max_max = [max_prev_cell_index]
    
    # Stage 1: Linking movements
    for index1, key1 in key1_copy:
        # find the maximal overlapping cell and add it as a link 
        # if it passes the area threshold
        subdic = {}
        # we have to loop over leftovers cuz we don't want to 
        # link a cell that is already linked
        for index2, key2 in leftover2:
            prob = scores[index1, index2]
            if prob > jacc_thres:
                subdic[(index2, key2)] = prob
        
        if len(subdic) == 0:
            # no cells cross the threshodl of probability in props2, so we don't link
            leftover1.remove((index1, key1))
            # nothing meets the overlap threshold
            props1[key1]['state'] = 'H' # set it to hanging mode
        else:
            # find the cell that matches with max prob
            max_overlapping_index, max_overlapping_key = max(subdic, key=subdic.get)
            # check the area threshold
            if area_thres * props1[key1]['area'] < props2[max_overlapping_key]['area']:
                # add just a link and remove the cell from further processing of movement link type
                # adjust the indices of the cells
                if props1[key1]['index'] is not None:
                    # if it is not None, it is assigned a number
                    # indices of cells are global and are assigned only at birth
                    # until death
                    # mother is the index of the cell it is connected
                    props2[max_overlapping_key]['mother'] = props1[key1]['index']
                    props2[max_overlapping_key]['index'] = props1[key1]['index']
                    # when the cell was born, same as the mother cell, this is just a movement link
                    props2[max_overlapping_key]['dob'] = props1[key1]['dob']
                    props2[max_overlapping_key]['initial_mother'] = props1[key1]['initial_mother']
                    props2[max_overlapping_key]['state'] = 'M'
                    props2[max_overlapping_key]['growth'] = props2[max_overlapping_key]['area'] - props1[key1]['area']
                    max_max.append(props1[key1]['index'])
                else:
                    # if for some reason, the cell in previous frame is not assigned
                    none_keys.append((max_overlapping_index, max_overlapping_key))
                
                # remove the maximum from frame 2 as it is already linked
                leftover2.remove((max_overlapping_index, max_overlapping_key))
                # remove the cell from the first frame as well, if there are daughters we 
                # will link them in stage2
                leftover1.remove((index1, key1))
            # if you don't pass the area thershold, then these could
            # still be probable candidates as they pass the overlap threshold
            adjacent_dict[(index1, key1)] = subdic
    
    # Stage 2: Linking daughters
    daughters = leftover2 # (index, key) tuple to index into scoring array
    mothers = leftover1
    mothers_twice = mothers + mothers # double so that each mother can have 2 daughters at most
    C = np.zeros([len(mothers_twice), len(daughters)])
    # fill in the cost matrix
    #print("Mothers: ", leftover1)
    #print("Daughters: ", daughters)
    row = 0
    for mother_idx, mother_key in mothers_twice:
        column = 0
        for daughter_idx, daughter_key in daughters:
            if (daughter_idx, daughter_key) in set(adjacent_dict[(mother_idx, mother_key)].keys()):
                C[row, column] = 1 - adjacent_dict[(mother_idx, mother_key)][(daughter_idx, daughter_key)]
            else:
                C[row, column] = 1
            column += 1
        row += 1
    #print(C)
    
    # solve the linear sum assignment to determine the splits
    # If there are weird assignments, here is where you can do track
    # corrections
    # Not we only detect splits, here
    pairs_mother, pairs_daughter = linear_sum_assignment(C)
    #print("Pairs mother: ", pairs_mother)
    #print("Pairs daughter: ", pairs_daughter)
    #pairs_mother = [mothers_twice[m] for m in pairs_mother]
    #pairs_daughter = [daughters[d] for d in pairs_daughter]
    #print("Pairs mother: ", pairs_mother)
    #print("Pairs daughter: ", pairs_daughter)
    max_max = max(max_max) + 1
    found = []
    for i in range(len(pairs_mother)):
        if area_thres * props1[mothers_twice[pairs_mother[i]][1]]['area'] > props2[daughters[pairs_daughter[i]][1]]['area']: 
            props2[daughters[pairs_daughter[i]][1]]['mother'] = props1[mothers_twice[pairs_mother[i]][1]]['index']
            props2[daughters[pairs_daughter[i]][1]]['dob'] = frame2_no
            props2[daughters[pairs_daughter[i]][1]]['index'] = int(max_max)
            props2[daughters[pairs_daughter[i]][1]]['growth']= (props1[mothers_twice[pairs_mother[i]][1]]['area']//2) - props2[daughters[pairs_daughter[i]][1]]['area']
            props2[daughters[pairs_daughter[i]][1]]['initial_mother'] = props1[mothers_twice[pairs_mother[i]][1]]['index']
            props2[daughters[pairs_daughter[i]][1]]['state'] = 'S'
            props1[mothers_twice[pairs_mother[i]][1]]['dod'] = True

            found.append(daughters[pairs_daughter[i]])
            max_max += 1
    # Stage 3: Fix something up for merges just in case
    
    # Stage 4: Clean Up to update states of unassigned elements
    still_leftover2 = set(leftover2).difference(set(found))
    none_keys += list(still_leftover2)
    for (index, key) in none_keys:
        props2[key]['index'] = max_max
        max_max +=1
    
    return props1, props2


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
    
    try:
        start_time = time.time()
        position = tracking_event['position']
        timepoint = tracking_event['time']
        # props is a dictionary with keys as channel numbers
        props_all_channels = tracking_event['props']
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
        
    
        if timepoint == 0:
            # we create a file with appropriate compression and chunking and write the first image
            cells_array = zarr.convenience.open(cells_filename, mode='a', shape=(1, height, width),
                            chunks=(1, height, 2*param.Save.channel_width), order='C', 
                            dtype='uint8', compressor=compressor)
            cells_array[0] = image

            # write the initial tracks 

            # for each channel at timepoint 0, set the index value of each cell
            # in the channel to be it's key
            for channel_no, one_channel_props in props_all_channels.items():
                for cell_idx in one_channel_props:
                    one_channel_props[cell_idx]['index'] = int(cell_idx)
                    
            tracks_write_filename = tracks_dir / Path('tracks_' + str(timepoint).zfill(4) + '.json')
            with open(tracks_write_filename, 'w') as tracks_fh:
                tracks_fh.write(json.dumps(props_all_channels))



        else:

            # read the tracks of the previous file so that you can modify the attributes
            # of cells in the previous frame, (for now we test read times and write)
            prev_track_filename = tracks_dir / Path('tracks_' + str(timepoint-1).zfill(4) + '.json')
            with open(prev_track_filename, 'r') as prev_track_fh:
                prev_track_data = json.load(prev_track_fh) 


            # mismatch case
            if len(prev_track_data) != len(props_all_channels) and (len(prev_track_data) != 0):
                # this means that the number of channels didn't match
                #raise ValueError(f"Number of channel don't match between {timepoint} and {timepoint-1}")
                # we put an empty image instead and for the current timepoint and write empty properties
                cells_array = zarr.convenience.open(cells_filename, mode='a', shape=(1, height, width),
                        chunks=(1, height, 2*param.Save.channel_width), order='C', 
                        dtype='uint8', compressor=compressor)
                cells_array.append(np.zeros((1, height, width), dtype='uint8'))


                tracks_write_filename = tracks_dir / Path('tracks_' + str(timepoint).zfill(4) + '.json')

                props_all_channels = {}
                with open(tracks_write_filename, 'w') as tracks_fh:
                    tracks_fh.write(json.dumps(props_all_channels))

            # matched case
            elif len(prev_track_data) == len(props_all_channels) and (len(prev_track_data) != 0):
                # number of channels matched between previous and current timepoint
                # do normal calculations and write back both results
        
                cells_array = zarr.convenience.open(cells_filename, mode='a', shape=(1, height, width),
                        chunks=(1, height, 2*param.Save.channel_width), order='C', 
                        dtype='uint8', compressor=compressor)
                cells_array.append(image[np.newaxis, :])

                for channel_no in prev_track_data:
                    prev_track_data[channel_no], props_all_channels[channel_no] = link_two_frames(prev_track_data[channel_no], props_all_channels[channel_no], 
                                    frame1_no=timepoint-1, frame2_no=timepoint)

                tracks_write_filename = tracks_dir / Path('tracks_' + str(timepoint).zfill(4) + '.json')
                with open(tracks_write_filename, 'w') as tracks_fh:
                    tracks_fh.write(json.dumps(props_all_channels))

                with open(prev_track_filename, 'w') as prev_track_fh:
                    prev_track_fh.write(json.dumps(prev_track_data))
            
            # failed at previous timepoint, restart case
            elif len(prev_track_data) == 0:
                # this means that previous timepoint failed and we wrote zeros
                # we need to restart tracking, do what you do for timepoint zero and reset all the indices
                        # we create a file with appropriate compression and chunking and write the first image
                cells_array = zarr.convenience.open(cells_filename, mode='a', shape=(1, height, width),
                                chunks=(1, height, 2*param.Save.channel_width), order='C', 
                                dtype='uint8', compressor=compressor)
                cells_array.append(image[np.newaxis, :])

                # for each channel at this timepoint, set the index value of each cell
                # in the channel to be it's key, this will reset the tracking to start 
                # from this frame onwards
                for channel_no, one_channel_props in props_all_channels.items():
                    for cell_idx in one_channel_props:
                        one_channel_props[cell_idx]['index'] = int(cell_idx)
                        
                tracks_write_filename = tracks_dir / Path('tracks_' + str(timepoint).zfill(4) + '.json')
                with open(tracks_write_filename, 'w') as tracks_fh:
                    tracks_fh.write(json.dumps(props_all_channels))


        duration = 1000 * (time.time() - start_time)
        sys.stdout.write(f"Tracking Pos: {tracking_event['position']} time: {tracking_event['time']} , no ch: {len(props_all_channels)}, duration: {duration:0.4f}ms ...\n")
        sys.stdout.flush()

        return None
    
    except Exception as e:
        sys.stdout.write(f"Tracking Pos: {tracking_event['position']} time: {tracking_event['time']} failed completely due to {e}:(\n")
        sys.stdout.flush()
        return None
        