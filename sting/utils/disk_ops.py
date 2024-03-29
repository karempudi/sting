
# File containing function that help you
# write file generated during the experiment
import numpy as np
import sys
import pickle
import pathlib
from pathlib import Path
from sting.utils.types import RecursiveNamespace
from skimage.io import imsave, imread
from skimage.measure import label
import h5py
from sting.utils.db_ops import read_from_db
import zarr
import json
from skimage.draw import line, line_aa

def write_files(event_data, event_type, param):
    """
    Function that writes data of all kinds in the appropriate directories 
    based on data obtained

    Arguments:
        event_data: dict containing data to write 
            event data should always have 'position', 'time'
        event_type: what does the data contain can be 'phase', 'segmented_cells', 'segmented_channels',
                'blobs' (single channel stacks with channel number), 'tracking'
        param: param used from the param file, contains experimental
               information
    """
    try:
        if 'position' not in event_data.keys():
            raise KeyError
        if 'time' not in event_data.keys():
            raise KeyError

        save_dir = Path(param.Save.directory) if isinstance(param.Save.directory, str) else param.Save.directory
        position_dir = save_dir / Path('Pos' + str(event_data['position']))

        # create position directory
        if not position_dir.exists():
            position_dir.mkdir(exist_ok=True, parents=True)
        
        
        if event_type == 'phase':
            # create event directory
            events_dir = position_dir / Path(event_type)
            if not events_dir.exists():
                events_dir.mkdir(exist_ok=True, parents=True)
            # put the image in phase directory at position with timepoint 
            image_filename = 'phase_' + str(event_data['time']).zfill(4) + '.tiff'
            image_filename = events_dir / Path(image_filename)
            imsave(image_filename, event_data['image'].astype('uint16'))
        if event_type == 'cells':
            # create events directory
            file_type = param.Save.small_file_format
            if file_type == '.tiff':
                cells_events_dir = position_dir / Path('segmented_cells')
                if not cells_events_dir.exists():
                    cells_events_dir.mkdir(exist_ok=True, parents=True)

                cells_filename = 'cells_' + str(event_data['time']).zfill(4) + '.tiff'
                cells_filename = cells_events_dir / Path(cells_filename)
                cell_prob = param.Analysis.Segmentation.thresholds.cells.probability
                imsave(cells_filename, (event_data['cells'] > cell_prob).astype('float32'),
                        plugin='tifffile', check_contrast=False, compress=6)
            elif file_type == '.hdf5':
                # if the fileformat is hdf5, we will make only one file and 
                # store each image as a dataset that 
                cells_events_store = position_dir / Path('segmented_cells.hdf5')
                cell_prob = param.Analysis.Segmentation.thresholds.cells.probability
                cells_data = (label(event_data['cells'] > cell_prob) % 255).astype('uint8')
                with h5py.File(cells_events_store, 'a') as cells_file:
                    cells_file.create_dataset(str(event_data['time']).zfill(4), data= cells_data,
                                compression=param.Save.small_file_compression_type,
                                compression_opts=param.Save.small_file_compression_level)                


        elif event_type == 'cells_channels':
            file_type = param.Save.small_file_format
            if file_type == '.tiff':
                # create event directory
                cells_events_dir = position_dir / Path('segmented_cells')
                if not cells_events_dir.exists():
                    cells_events_dir.mkdir(exist_ok=True, parents=True)

                channels_events_dir = position_dir / Path('segmented_channels')            
                if not channels_events_dir.exists():
                    channels_events_dir.mkdir(exist_ok=True, parents=True)

                cells_filename = 'cells_' + str(event_data['time']).zfill(4) + '.tiff'
                cells_filename = cells_events_dir / Path(cells_filename)
                channels_filename = 'channels_' + str(event_data['time']).zfill(4) + '.tiff'
                channels_filename = channels_events_dir / Path(channels_filename)
                cell_prob = param.Analysis.Segmentation.thresholds.cells.probability
                channel_prob = param.Analysis.Segmentation.thresholds.channels.probability
                imsave(cells_filename, (event_data['cells'] > cell_prob).astype('float32'),
                        plugin='tifffile', check_contrast=False, compress=6)
                imsave(channels_filename, (event_data['channels'] > channel_prob).astype('float32'), 
                        plugin='tifffile', check_contrast=False, compress=6)
            elif file_type == '.hdf5':
                # if the fileformat is hdf5, we will make only one file and 
                # store each image as a dataset that 
                cells_events_store = position_dir / Path('segmented_cells.hdf5')
                cell_prob = param.Analysis.Segmentation.thresholds.cells.probability
                cells_data = (label(event_data['cells'] > cell_prob) % 255).astype('uint8')
                with h5py.File(cells_events_store, 'a') as cells_file:
                    cells_file.create_dataset(str(event_data['time']).zfill(4), data= cells_data,
                                compression=param.Save.small_file_compression_type,
                                compression_opts=param.Save.small_file_compression_level)                

                # if the fileformat is hdf5, we will make only one file and 
                # store each image as a dataset that 
                channels_events_store = position_dir / Path('segmented_channels.hdf5')
                channel_prob = param.Analysis.Segmentation.thresholds.channels.probability
                channels_data = (label(event_data['channels'] > channel_prob) % 255).astype('uint8')
                with h5py.File(channels_events_store, 'a') as channels_file:
                    channels_file.create_dataset(str(event_data['time']).zfill(4), data= channels_data,
                                compression=param.Save.small_file_compression_type,
                                compression_opts=param.Save.small_file_compression_level)                

        elif event_type == 'cells_cut_track_init':
            # here were write something that is useful for tracking
            # no writing tiffs for this format, 
            sys.stdout.write(f"Trying to write cut channels ...\n")
            sys.stdout.flush()
            cells_events_store = position_dir / Path('cells_tracks.hdf5')
            cell_prob = param.Analysis.Segmentation.thresholds.cells.probability
            cells_data = (event_data['cells'] > cell_prob)
            # for each channel iterate over and create groups and datasets
            channel_locations = []
            for block in event_data['channel_locations']:
                channel_locations.extend(event_data['channel_locations'][block]['channel_locations'])
            channel_width = param.Save.channel_width

            # we only grab stuff between barcodes and ignore the ends, so this operation will not result in errors
            with h5py.File(cells_events_store, 'a') as cells_file:
                for i, location in enumerate(channel_locations, 0):
                    img_slice = cells_data[:, max(location-channel_width, 0): 
                                    min(event_data['raw_shape'][1], location+channel_width)]
                    # label regions and make them uints for good fast compression
                    img_slice = (label(img_slice) % 255).astype('uint8')
                    # chanel no, cells + _ time.zfil(4)
                    write_string = str(i) + '/cells/cells_' + str(event_data['time']).zfill(4)
                    cells_file.create_dataset(write_string, data=img_slice,
                            compression=param.Save.small_file_compression_type,
                            compression_opts=param.Save.small_file_compression_level)
    except Exception as e:
        sys.stdout.write(f"Writing failed due to {e} for data {event_data} ..\n")
        sys.stdout.flush()

    except KeyError:
        sys.stdout.write(f"Writing failed for due to lack of position key in data ..\n")
        sys.stdout.flush()
    except Exception as e:
        sys.stdout.write(f"Writing failed due to {e} for data {event_data} ..\n")
        sys.stdout.flush()

def read_files(read_type, param, position, channel_no, max_imgs=20):
    """
    File that will help read data from disk of various types for
    visualization purposes only

    Arguements:
        read_type: 'phase

    Returns:

    """
    try:
        dir_name = Path(param.Save.directory) if isinstance(param.Save.directory, str) else param.Save.directory

        sys.stdout.write(f"Read files called with Pos:{position}, Ch no: {channel_no}..\n")
        sys.stdout.flush()

        if read_type == 'phase':
            phase_dir = dir_name / Path('Pos' + str(position)) / Path('phase')
            # read all the files 
            phase_filenames = sorted(list(phase_dir.glob('*.tiff')))
            #print(phase_filenames)
            #print(phase_filenames[-1].name)
            last_key = int(phase_filenames[-1].name.split('_')[-1].split('.')[0])
            #print("Last phase image key", last_key)
            prev_phase_img = imread(phase_filenames[-1])
            height, _ = prev_phase_img.shape
            barcode_data = read_from_db('barcode_locations', dir_name, position=position, timepoint=last_key)
            #print(f"Last key: {last_key}, barcode_data: {barcode_data}")
            # get the barcode locations and grab barcode images from the phase image
            channel_location = barcode_data['channel_locations'][channel_no]
            for i, barcode in enumerate(barcode_data['barcode_locations'], 0):
                if (((barcode[0] + barcode[2])/2) > channel_location):
                    break
            left_barcode =  barcode_data['barcode_locations'][i-1]
            right_barcode = barcode_data['barcode_locations'][i]
            left_barcode_img = prev_phase_img[int(left_barcode[1]): int(left_barcode[3]), int(left_barcode[0]): int(left_barcode[2])]
            right_barcode_img = prev_phase_img[int(right_barcode[1]): int(right_barcode[3]), int(right_barcode[0]): int(right_barcode[2])]
            
            # depending upon no of images needed grab the filenames, iterate and stack
            if max_imgs != None:
                files_2_iter = phase_filenames[-max_imgs:]
            else:
                files_2_iter = phase_filenames
            channel_width = 2 * param.Save.channel_width 
            full_img = np.zeros((height, len(files_2_iter)*channel_width)) 
            for i, filename in enumerate(files_2_iter, 0):
                phase_slice = imread(filename)[:, channel_location-(channel_width//2): channel_location+(channel_width//2)]
                full_img[:, (i)*channel_width: (i+1)*channel_width] = phase_slice

            return {
                'image': full_img,
                'left_barcode': left_barcode_img,
                'right_barcode': right_barcode_img,
            }
        elif read_type == 'cell_seg' and param.Save.small_file_format == '.hdf5':
            # get cell segmentation data image from reading the cell images data
            # check if cells_tracks.hdf5 exists

            filename = dir_name / Path('Pos' + str(position)) / Path('cells_tracks.hdf5')
            #print(filename)
            if not filename.exists():
                raise FileNotFoundError(f"File not found error at Pos{position} for readtype {cell_seg} ...:( ")
            with h5py.File(filename, 'r') as cells_file:
                n_images = len(cells_file[str(channel_no) + '/cells'])
                keys = [key for key in cells_file[str(channel_no) + '/cells'].keys()]
                if len(keys) == 0:
                    raise KeyError("n_keys is 0")
                height, width = cells_file[str(channel_no) + '/cells/' + keys[0]][()].shape
                if max_imgs != None:
                    keys_to_get = keys[-max_imgs:]
                else:
                    keys_to_get = keys
                #print(keys_to_get)
                full_img = np.zeros((height, len(keys_to_get)*width))
                for i, key in enumerate(keys_to_get, 0):
                    img_str = str(channel_no) + '/cells/' + key
                    full_img[:, i*width: (i+1)*width] = cells_file[img_str][()]
                
                prev_phase_img = cells_file['/prev_phase'][()]
                last_key = int(keys_to_get[-1].split('_')[1])

                barcode_data = read_from_db('barcode_locations', dir_name, position=position, timepoint=last_key)
                #print(f"Last key: {last_key}, barcode_data: {barcode_data}")
                # get the barcode locations and grab barcode images from the phase image
                channel_location = barcode_data['channel_locations'][channel_no]
                for i, barcode in enumerate(barcode_data['barcode_locations'], 0):
                    if (((barcode[0] + barcode[2])/2) > channel_location):
                        break
                left_barcode =  barcode_data['barcode_locations'][i-1]
                right_barcode = barcode_data['barcode_locations'][i]
                left_barcode_img = prev_phase_img[int(left_barcode[1]): int(left_barcode[3]), int(left_barcode[0]): int(left_barcode[2])]
                right_barcode_img = prev_phase_img[int(right_barcode[1]): int(right_barcode[3]), int(right_barcode[0]): int(right_barcode[2])]

            #print(f"Image shape: {full_img.shape}, channel location: {channel_location} left_barcode: {left_barcode}, right_barcode: {right_barcode}")
            return {
                'image': full_img, 
                'left_barcode': left_barcode_img,
                'right_barcode': right_barcode_img,
            }
        
        elif read_type == 'cell_seg' and param.Save.small_file_format == '.zarr':

            # read the zarr file and load the corresponding segmentation masks
            cells_filename = dir_name / Path('Pos' + str(position)) / Path('cells.zarr')

            channel_width = param.Save.channel_width

            data = zarr.convenience.open(cells_filename, mode='r')
            n_slices, height, width = data.shape
            last_key = n_slices - 1
            prev_phase_filename = dir_name / Path('Pos' + str(position)) / Path('phase') / Path('phase_' + str(last_key).zfill(4) + '.tiff')
            barcode_data = read_from_db('barcode_locations', dir_name, position=position, timepoint=last_key)
            #print(f"Last key: {last_key}, barcode_data: {barcode_data}")
            # get the barcode locations and grab barcode images from the phase image
            channel_location = barcode_data['channel_locations'][channel_no]
            for i, barcode in enumerate(barcode_data['barcode_locations'], 0):
                if (((barcode[0] + barcode[2])/2) > channel_location):
                    break
            left_barcode =  barcode_data['barcode_locations'][i-1]
            right_barcode = barcode_data['barcode_locations'][i]
            prev_phase_img = imread(prev_phase_filename)
            left_barcode_img = prev_phase_img[int(left_barcode[1]): int(left_barcode[3]), int(left_barcode[0]): int(left_barcode[2])]
            right_barcode_img = prev_phase_img[int(right_barcode[1]): int(right_barcode[3]), int(right_barcode[0]): int(right_barcode[2])]

            if max_imgs == None:
                full_img = data[:, :, (channel_no) * 2 * channel_width : (channel_no+1) * 2 * channel_width]
                full_img = np.hstack(full_img)
            else:
                full_img = data[-max_imgs:, :, (channel_no) * 2 * channel_width : (channel_no+1) * 2 * channel_width]
                full_img = np.hstack(full_img)

            
            return {
                'image': 255-full_img,
                'left_barcode': left_barcode_img,
                'right_barcode': right_barcode_img
            }
        
        elif read_type == 'cell_tracks' and param.Save.small_file_format == '.zarr':

            # read the zarr file and load the corresponding segmentation masks
            cells_filename = dir_name / Path('Pos' + str(position)) / Path('cells.zarr')

            channel_width = param.Save.channel_width

            data = zarr.convenience.open(cells_filename, mode='r')
            n_slices, height, width = data.shape
            last_key = n_slices - 1
            prev_phase_filename = dir_name / Path('Pos' + str(position)) / Path('phase') / Path('phase_' + str(last_key).zfill(4) + '.tiff')
            barcode_data = read_from_db('barcode_locations', dir_name, position=position, timepoint=last_key)
            #print(f"Last key: {last_key}, barcode_data: {barcode_data}")
            # get the barcode locations and grab barcode images from the phase image
            channel_location = barcode_data['channel_locations'][channel_no]
            for i, barcode in enumerate(barcode_data['barcode_locations'], 0):
                if (((barcode[0] + barcode[2])/2) > channel_location):
                    break
            left_barcode =  barcode_data['barcode_locations'][i-1]
            right_barcode = barcode_data['barcode_locations'][i]
            prev_phase_img = imread(prev_phase_filename)
            left_barcode_img = prev_phase_img[int(left_barcode[1]): int(left_barcode[3]), int(left_barcode[0]): int(left_barcode[2])]
            right_barcode_img = prev_phase_img[int(right_barcode[1]): int(right_barcode[3]), int(right_barcode[0]): int(right_barcode[2])]

            # tracks dir
            tracks_dir = dir_name / Path('Pos' + str(position)) / Path('tracks')
            if max_imgs == None:
                full_img = data[:, :, (channel_no) * 2 * channel_width : (channel_no+1) * 2 * channel_width]
                full_img = np.hstack(full_img)
                tracks_filenames = [tracks_dir / Path('tracks_' + str(i).zfill(4) + '.json') for i in range(n_slices)]
            else:
                full_img = data[-max_imgs:, :, (channel_no) * 2 * channel_width : (channel_no+1) * 2 * channel_width]
                full_img = np.hstack(full_img)
                tracks_filenames = [tracks_dir / Path('tracks_' + str(i).zfill(4) + '.json') for i in range(n_slices-1, max(-1, n_slices-max_imgs),-1)]

            #full_img = np.stack((full_img, full_img, full_img, full_img), axis=-1)
            full_img = np.stack((full_img, full_img, full_img), axis=-1)
            #full_img[:, :, 3] = 255
            track_data = []
            for filename in tracks_filenames:
                with open(filename, 'r') as fh:
                    track_data.append(json.load(fh)[str(channel_no)])
            indices = list(range(len(track_data)))
            for (i, j) in zip(indices[:-1], indices[1:]):
                res_dict1 = track_data[i]
                res_dict2 = track_data[j]
                for key1 in res_dict1:
                    cell_index = res_dict1[key1]['index']
                    for key2 in res_dict2:
                        # check for same index
                        if res_dict2[key2]['index'] == cell_index:
                            cm_t1_x, cm_t1_y = res_dict1[key1]['cm']
                            cm_t2_x, cm_t2_y = res_dict2[key2]['cm']
                            #rows, cols, weights = line_aa(int(cm_t1_x), int(cm_t1_y) + (i * 2 * channel_width), int(cm_t2_x), int(cm_t2_y) + j * 2 * channel_width)
                            rows, cols = line(int(cm_t1_x), int(cm_t1_y) + (i * 2 * channel_width), int(cm_t2_x), int(cm_t2_y) + j * 2 * channel_width)
                            #w = weights.reshape([-1, 1])
                            #lineColorRgb = [255, 0, 0]
                            #full_img[rows, cols, 0:3] =  (np.multiply((1 - w) * np.ones([1, 3]), full_img[rows, cols, 0:3]) + w * np.array([lineColorRgb]))
                            full_img[rows, cols, :] = np.array([25, 0, 0])
                        elif res_dict2[key2]['mother'] == cell_index and res_dict2[key2]['state'] == 'S':
                            cm_t1_x, cm_t1_y = res_dict1[key1]['cm']
                            cm_t2_x, cm_t2_y = res_dict2[key2]['cm']
                            #rows, cols, weights = line_aa(int(cm_t1_x), int(cm_t1_y) + (i * 2 * channel_width), int(cm_t2_x), int(cm_t2_y) + j * 2 * channel_width)
                            rows, cols = line(int(cm_t1_x), int(cm_t1_y) + (i * 2 * channel_width), int(cm_t2_x), int(cm_t2_y) + j * 2 * channel_width)
                            #w = weights.reshape([-1, 1])
                            #lineColorRgb = [0, 255, 0]
                            #full_img[rows, cols, 0:3] =  (np.multiply((1 - w) * np.ones([1, 3]), full_img[rows, cols, 0:3]) + w * np.array([lineColorRgb]))
                            full_img[rows, cols, :] = np.array([25, 0, 0])
            return {
                'image': 255-full_img,
                'left_barcode': left_barcode_img,
                'right_barcode': right_barcode_img
            }
 

    except KeyError as k:
        sys.stdout.write(f"Reading failded to lay of some key {k}...\n")
        sys.stdout.flush()
    except Exception as e:
        sys.stdout.write(f"Reading data failed due to {e} for {read_type} ..\n")
        sys.stdout.flush()
