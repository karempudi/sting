import sys
import pathlib
from pathlib import Path
from sting.utils.types import RecursiveNamespace
from skimage.io import imsave
from skimage.measure import label
import h5py


def write_files_and_track(event_data, event_type, param):
    """
    Function that writes data of all kinds in the appropirate directories
    based on data obtained after doing the tracking  

    Arguments:
        event_data: diction containing data to write
            event data should always have 'position' and 'time'
        event_type: what data does it contain and what to do with it

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

        # create a position directory if not exists
        if not position_dir.exists():
            position_dir.mkdir(exist_ok=True, parents=True)
        
        if event_type == 'track_position':
            pass
        
    except KeyError:
        sys.stdout.write(f"Writing failed due to lack of position and/or time key data ..\n")
        sys.stdout.flush()