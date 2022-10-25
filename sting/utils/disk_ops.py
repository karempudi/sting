
# File containing function that help you
# write file generated during the experiment
import sys
import pickle
import pathlib
from pathlib import Path
from sting.utils.types import RecursiveNamespace
from tifffile import imsave

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
        
        # create event directory
        events_dir = position_dir / Path(event_type)
        if not events_dir.exists():
            events_dir.mkdir(exist_ok=True, parents=True)
        
        if event_type == 'phase':
            # put the image in phase directory at position with timepoint 
            image_filename = str(event_data['time']) + '.tiff'
            image_filename = events_dir / Path(image_filename)
            imsave(image_filename, event_data['image'].astype('uint16'))
    except KeyError:
        sys.stdout.write(f"Writing failed for due to lack of position key in data ..\n")
        sys.stdout.flush()
    except Exception as e:
        sys.stdout.write(f"Writing failed due to {e} for data {event_data} ..\n")
        sys.stdout.flush()
    

    

def read_file(param):
    pass