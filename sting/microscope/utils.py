import numpy as np
from pycromanager import Core
import sys

def fetch_mm_image():
    # fetches the current micromanger image
    #return np.random.randint(low = 0, high = 2, size=(10, 10))
    try:
        core = Core()
        core.snap_image()
        tagged_image = core.get_tagged_image()
        pixels = np.reshape(tagged_image.pix,
                newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
        return pixels
    except Exception as e:
        sys.stdout.write(f"Error {e} while snapping image on the microscope ... \n")
        sys.stdout.flush()
        return np.random.randint(low = 0, high = 2, size=(10, 10))

def construct_pos_file(positions_list, devices_dict={
    'xy_device': "XYStage",
    'z_device': 'PFSOffset'
    }, version=2.0):
    """
    Function that will generate a positions (*.pos) file that is loadable into
    Micromanager 2.0/ 1.4 using the list of positions and devices.
    
    Arguments:
        positions_list: 
        devices_dict: 
    """

    if version == 2.0:
        constructed_file = {
                "encoding": "UTF-8",
                "format": "Micro-Manager Property Map",
                "major_version": 2,
                "minor_version": 0,
                "map": {
                    "StagePositions": {
                    "type": "PROPERTY_MAP",
                    "array": []
                    }
                }
        }

        for position in positions_list:
            one_position_dict = {
                "DefaultXYStage": {
                    "type": "STRING",
                    "scalar": devices_dict['xy_device']
                },
                "DefaultZStage": {
                    "type": "STRING",
                    "scalar": devices_dict['z_device']
                },
                "DevicePositions": {
                    "type": "PROPERTY_MAP",
                    "array": [
                    {
                        "Device": {
                        "type": "STRING",
                        "scalar": devices_dict['z_device']
                        },
                        "Position_um": {
                        "type": "DOUBLE",
                        "array": [
                            position['z']
                        ]
                        }
                    },
                    {
                        "Device": {
                        "type": "STRING",
                        "scalar": devices_dict['xy_device']
                        },
                        "Position_um": {
                        "type": "DOUBLE",
                        "array": [
                            position['x'],
                            position['y']
                        ]
                        }
                    }
                    ]
                },
                "GridCol": {
                    "type": "INTEGER",
                    "scalar": position['grid_col']
                },
                "GridRow": {
                    "type": "INTEGER",
                    "scalar": position['grid_row']
                },
                "Label": {
                    "type": "STRING",
                    "scalar": position['label']
                },
                "Properties": {
                    "type": "PROPERTY_MAP",
                    "scalar": {}
                }
                }
            
            # append the position to the file to make full list
            constructed_file["map"]["StagePositions"]["array"].append(one_position_dict)

        return constructed_file

    elif version == 1.4:

        constructed_file = {
            "VERSION": 3,
            "ID": "Micro-Manager XY-position list",
            "POSITIONS": []
            }
        for position in positions_list:
            one_position_dict = {
                    "GRID_COL": position['grid_col'],
                    "DEVICES": [
                        {
                        "DEVICE": devices_dict['z_device'],
                        "AXES": 1,
                        "Y": 0,
                        "X": position['z'],
                        "Z": 0
                        },
                        {
                        "DEVICE": "XYStage",
                        "AXES": 2,
                        "Y": position['y'],
                        "X": position['x'],
                        "Z": 0
                        }
                    ],
                    "PROPERTIES": {},
                    "DEFAULT_Z_STAGE": "",
                    "LABEL": position['label'],
                    "GRID_ROW": position['grid_row'],
                    "DEFAULT_XY_STAGE": devices_dict['xy_device']
                }
            constructed_file["POSITIONS"].append(one_position_dict)

        return constructed_file
