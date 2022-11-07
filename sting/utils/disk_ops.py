
# File containing function that help you
# write file generated during the experiment
import sys
import pickle
import pathlib
from pathlib import Path
from sting.utils.types import RecursiveNamespace
from skimage.io import imsave

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
        elif event_type == 'cells_channels':
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
