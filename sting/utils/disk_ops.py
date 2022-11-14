
# File containing function that help you
# write file generated during the experiment
import sys
import pickle
import pathlib
from pathlib import Path
from sting.utils.types import RecursiveNamespace
from skimage.io import imsave
from skimage.measure import label
import h5py

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

def read_files(read_keys, read_type, param):
    """
    File that will help read data from disk of various types

    Arguements:

    Returns:

    """
    return None
