from abc import ABC, abstractmethod
import pycromanager
import numpy as np
import pathlib
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
import torch.multiprocessing as tmp
import logging
import json
import sys

class Motion(ABC):
    """
    
    Abstract class defining motion

    Args:
        objective: Magnification of the objective under use (20, 40 or 100)
    """

    def __init__(self, objective: int = 40):
        super().__init__()
        self.objective = objective
        
    @abstractmethod
    def plot_motion_plan(self):
        """
        Plots the motion plan diagram of one round of the experiment.
        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

class RectGridMotion(Motion):
    """
    Rectangular grid of positions, where the positions are 
    specified by the 4 corners and a grid of certain size specifed by
    the number of positions to grab between the corners in x and y directions

    (TL) (0) ---------- (TR) (1)
     |                   |
     |                   |
     |                   |
     |                   |
     |                   |
     |                   |
    (BL) (3) ---------- (BR) (2)

    The number in the bracket represent the indices you will use to 
    figure out which are the 4 corners in xy plane.

    Args:
        filename: a file name with positions list from micromanager 2.0 with
            position names "PosTL", "PosTR", "PosBR", "PosBL", that mark the
            4 corners of the rectangle containing 25 rows of mother-machine
            channels
        movement_type (str): left or right. More on this in the make_pattern function docstring.
            basically meant to show how the positions are going to be moving.
        corner_names: a list of position labels used 
        nrows (int): number of rows  (depends on how many rows you want to image)
        ncols (int): number of cols  (depends on magnification, stick to 40x, 100 blocks, so 20 positions)
    """
    def __init__(self,
                filename: Union[str, pathlib.Path] = None, objective: int = 40,
                movement_type: str = 'left',
                corner_names=['TL', 'TR', 'BR', 'BL'],
                nrows=None, ncols=None,
                ):
        super().__init__(objective=objective)
       
        self.movement_type = movement_type
        self.corner_names = corner_names
        self.corner_pos_list = None
        self.corner_pos_dict = {}
        self.microscope_props = None
        self.positions = [] # set it to be used by pycromanager event-construction
        self.nrows = nrows
        self.ncols = ncols
        if filename != None:
            self.filename = filename if isinstance(filename, pathlib.Path) else Path(filename)
            self.microscope_props, self.corner_pos_list = self.parse_position_file(filename)
            self.corner_pos_dict = self.fill_corner_positions()
            if not self.verify_corners():
                raise ValueError(f"Corners couldn't be validated as a convex polygon")
            if self.nrows == None  or self.ncols == None:
                raise ValueError(f"Number or rows and columns not set")
            else:
                self.construct_grid()
        
        
    @staticmethod
    def verify_bounds(positions):
        pass
    
    def set_rows(self, rows):
        self.nrows = rows
    
    def set_cols(self, cols):
        self.ncols = cols
        
        
    def verify_corners(self):
        """
        Function to verify that you have a rectangle, check if all the 4 corners
        are valid and form a sort of rectangle and are not going to make a 
        non-convex polygon of some kind.
        Returns:
            True or False
        """
        if len(self.corner_pos_dict) != 4:
            raise ValueError(f"Corners position dict has only {len(self.corner_pos_dict)} corners. Need 4 :(")
        
        invalid_pos = False
        corners_dict = self.corner_pos_dict
        if corners_dict['TL']['x']  <= corners_dict['TR']['x']:
            invalid_pos = True
        if corners_dict['BL']['x'] <= corners_dict['BR']['x']:
            invalid_pos = True
        if corners_dict['TL']['x']  <= corners_dict['BR']['x']:
            invalid_pos = True
        if corners_dict['BL']['x'] <= corners_dict['TR']['x']:
            invalid_pos = True
        
        if corners_dict['TL']['y']  >= corners_dict['BL']['y']:
            invalid_pos = True
        if corners_dict['TL']['y'] >= corners_dict['BR']['y']:
            invalid_pos = True
        if corners_dict['TR']['y']  >= corners_dict['BL']['y']:
            invalid_pos = True
        if corners_dict['TR']['y'] >= corners_dict['BR']['y']:
            invalid_pos = True
        
        if invalid_pos:
            return False
        else:
            return True
        
    
    def set_corner_position(self, label, position_dict):
        """
        Function that will set the positions of the corners,
        by passing one at a time. It is used if you set the positions
        from a UI by grabbing the current location.
        Arguments:
            label: one of 'TL', 'TR', 'BR', 'BL'
            position_dict: dict with keys 'X', 'Y', 'Z', 'grid_row', 'grid_col', 'label'
        """
        if label not in self.corner_names:
            raise ValueError(f"Position label not in the corner label list, found: {label}...")
        else:
            # check if all keys are in the position dict
            keys = ['x', 'y', 'z', 'grid_row', 'grid_col', 'label']
            for key in keys:
                if key not in position_dict:
                    raise ValueError(f"Key {key} not found in position dict: {position_dict}")
            self.corner_pos_dict[label] = position_dict
            
    def fill_corner_positions(self):
        if len(self.corner_pos_list) != 4:
            raise ValueError(f"Positions Corner Construction failed, found {len(self.corner_pos_list)}/4 corners")
        corners = {}
        for position in self.corner_pos_list:
            if position['label'][3:] == self.corner_names[0]:
                corners[self.corner_names[0]] = position
            elif position['label'][3:] == self.corner_names[1]:
                corners[self.corner_names[1]] = position
            elif position['label'][3:] == self.corner_names[2]:
                corners[self.corner_names[2]] = position
            elif position['label'][3:] == self.corner_names[3]:
                corners[self.corner_names[3]] = position
                
        if len(corners) != 4:
            raise ValueError(f"Positions Corner Construction failed, found {len(self.corner_pos_list)}/4 corners")
        
        return corners
    
    def construct_grid(self, start_position_no=1):
        
        # check that you have all the 4 positions and verify them and then
        # construct the list of positions that will be visited in linear fashion
        if len(self.corner_pos_dict) != 4:
            raise ValueError(f"All 4 Corners not set")
        corners = self.corner_pos_dict
        x_top = np.linspace(corners['TL']['x'], corners['TR']['x'], num=self.ncols)
        x_bot = np.linspace(corners['BL']['x'], corners['BR']['x'], num=self.ncols)
        y_left = np.linspace(corners['TL']['y'], corners['BL']['y'], num=self.nrows)
        y_right = np.linspace(corners['TR']['y'], corners['BR']['y'], num=self.nrows)

        z_top = np.linspace(corners['TL']['z'], corners['TR']['z'], num=self.ncols)
        z_bot = np.linspace(corners['BL']['z'], corners['BR']['z'], num=self.ncols)
        z_left = np.linspace(corners['TL']['z'], corners['BL']['z'], num=self.nrows)
        z_right = np.linspace(corners['TR']['z'], corners['BR']['z'], num=self.nrows)


        def get_xyz(row, col):
            x = np.linspace(x_top[col], x_bot[col], num=self.nrows)[row]
            y = np.linspace(y_left[row], y_right[row], num=self.ncols)[col]

            z_x_interp = np.linspace(z_top[col], z_bot[col], num=self.nrows)[row]
            z_y_interp = np.linspace(z_left[row], z_right[row], num=self.ncols)[col]

            # might have to do an acutual bilinear interp on a quadrilateral later on if this is causing trouble
            # for interpolating in 'z' we assume we have something that looks more like a rectangle
            # Interpolate 'z' first in x and then in y
            #z_top_interp = (((corner['TR']['x'] - x)*corner['TL']['z']) + ((x - corner['TL']['x'])*corner['TR']['z'])) / (corner['TR']['x'] - corner['TL']['x'])
            #z_bot_interp = (((corner['BR']['x'] - x)*corner['BL']['z']) + ((x - corner['BL']['x'])*corner['BR']['z'])) / (corner['BR']['x'] - corner['BL']['x'])
            # Interpolate in 'y'
            #z = () / (corner['BR']['y'] - corner['BL'])
            z = (z_x_interp + z_y_interp)/2.0
            return (x, y, z)


        tuples = self.make_pattern(self.nrows, self.ncols, self.movement_type)
        positions = []
        for counter, (i, j) in enumerate(tuples, start_position_no):
            one_position = get_xyz(i, j)

            positions.append({
                'x': one_position[0],
                'y': one_position[1],
                'z': one_position[2],
                'grid_row': j,
                'grid_col': i, 
                'label': 'Pos'+ str(counter).zfill(5),
            })
        self.positions = positions
    
    def make_pattern(self, nrows, ncols, movement_type='left'):
        """
        Make a long snake (aka meander) pattern that minimize movement 
        between two rows
        Arguments:
            nrows: number of rows in the snake
            ncols: number of cols in the snake
            movement_type: 'left' or 'right' (these are mean to be how you move
                        on the chip). Left half starts at TL->TR-> going down
                        Right half starts BL-> BR -> going up
        Returns:
            tuples: a list of tuples with i, j of the position
        The microscope is intended to stop at each of the
        tuple.
        """
        if movement_type == 'left':
            tuples = []
            for i in range(nrows):
                if i%2 == 0:
                    for j in range(ncols):
                        tuples.append((i, j))
                elif i%2 == 1:
                    for j in range(ncols-1, -1, -1):
                        tuples.append((i, j))
            return tuples
        elif movement_type == 'right':
            tuples = []
            for i in range(nrows-1, -1, -1):
                if i%2 == 0:
                    for j in range(ncols):
                        tuples.append((i, j))
                elif i%2 == 1:
                    for j in range(ncols-1, -1, -1):
                        tuples.append((i, j))
            return tuples    

    @staticmethod
    def parse_position_file(filename: Union[str, pathlib.Path]):
        """
        Takes in positions filename from micromanager stage positions list
        and then spits out a list of positions and microscope properties
        Args:
            filename (Union[str, pathlib.Path]): .pos filename from micromanager
                    stage positions list
        """
        filename = filename if isinstance(filename, pathlib.Path) else Path(filename)

        if not filename.exists():
            raise FileNotFoundError(f"Positions filename : {filename} not found")

        if filename.suffix != '.pos':
            raise ValueError(f"Position file {filename} doesn't have .pos extension")

        # log to console
        with open(filename, 'r') as json_handle:
            data = json.load(json_handle)
        
        # log micromanger file read and print the version number
        sys.stdout.write(f"Micro-manager version: {data['major_version']} - {data['minor_version']}\n")
        sys.stdout.flush()
        
        positions_dicts = data['map']['StagePositions']['array']
        
        if len(positions_dicts) == 0:
            raise ValueError(f"No positions found in the positions file {filename}")
        
        # find the default stage positions list
        defaultXYStage = positions_dicts[0]['DefaultXYStage']['scalar']
        defaultZStage = positions_dicts[0]['DefaultZStage']['scalar']
        
        microscope_props = {
            'XYStage': defaultXYStage,
            'ZStage': defaultZStage,
        }
        
        # Now we loop over and collect all positions
        
        # We need to find the device indices for each position as their
        # order can vary depending on many factors like pressing the replace
        # button to manaually adjust pfsoffset for positions you find are
        # way too off after you marked the position.
        
        sys.stdout.write(f"DefaultXYStage device: {defaultXYStage} found ...\n")
        sys.stdout.write(f"DefualtZStage device: {defaultZStage} found ... \n")
        sys.stdout.flush()
        
        # a list of positions: each position {'x': X, 'y': Y, 'z': Z,
        # 'grid_row': grid_row, 'grid_col': grid_col}
        # Not sure how grid is useful yet
        positions = []
        
        # loop over the positions_dicts, find the right XY device, Z device indices
        # and then construct positions appropriately
        for position_item in positions_dicts:
            # a list with device and their props
            position_data = position_item['DevicePositions']['array']
            xy_device = position_item['DefaultXYStage']['scalar']
            z_device = position_item['DefaultZStage']['scalar']
            for device_item in position_data:
                if device_item['Device']['scalar'] == xy_device:
                    # fill the x and y
                    X = device_item['Position_um']['array'][0] # x is 0 index
                    Y = device_item['Position_um']['array'][1] # y is 1 index
                elif device_item['Device']['scalar'] == z_device:
                    Z = device_item['Position_um']['array'][0]
            grid_row = position_item['GridRow']['scalar']
            grid_col = position_item['GridCol']['scalar']
            # Most important thing to grab
            label = position_item['Label']['scalar']
            
            positions.append({'x': X, 'y': Y, 'z': Z, 'grid_row': grid_row,
                            'grid_col': grid_col, 'label': label})
        
        # we not the x and y bounds for future use when we add safety checks
        # in-case anyone fucks up the position list naming/generation in
        # some situations
        x_values = [item['x'] for item in positions]
        y_values = [item['y'] for item in positions]
        z_values = [item['z'] for item in positions]
        
        microscope_props['x_range'] = (min(x_values), max(x_values))
        microscope_props['y_range'] = (min(y_values), max(y_values))
        microscope_props['z_range'] = (min(z_values), max(z_values))
        
        return microscope_props, positions
    
    @classmethod
    def parse(cls, param):
        """ 
            Takes parameter object and returns a new motion object
        Args:
            param: a recursive namespace containing details of motion
        """
        pass

    def plot_motion_plan(self):
        plt.figure()
        plt.close()    

class TwoRectGridMotion(Motion):
    """
    Two Rectangular grids of positions, where the positions are 
    specified by the 4 corners and a grid of certain size specifed by
    the number of positions to grab between the corners in x and y directions

    (TL1) (0) ---------- (TR1) (1)   (TL2) (0) ---------- (TR2) (1)
     |                   |            |                    |
     |                   |            |                    |
     |                   |            |                    |
     |                   |            |                    |
     |                   |            |                    |
     |                   |            |                    |
    (BL1) (3) ---------- (BR1) (2)   (BL2) (3) ---------- (BR2) (2)

    The number in the bracket represent the indices you will use to 
    figure out which are the 8 corners in xy plane.

    Args:
        filename: a file name with positions list from micromanager 2.0 with
            position names "PosTL1", "PosTR1", "PosBR1", "PosBL2", "PosTL2", 
            "PosTR2", "PosBR1", "PosBL2" that mark the 8 corners of the rectangle
            containing 25 rows of mother-machine channels or how many ever you 
            specify, of this class, the restriction will be that
            number of rows should be odd.
        corner_names: a list of position labels used 
        nrows (int): number of rows  (depends on how many rows you want to image)
        ncols (int): number of cols  (depends on magnification, stick to 40x, 100 blocks, so 20 positions)
    """
    def __init__(self, filename: Union[str, pathlib.Path] = None, objective: int = 40,
                corner_names=['TL', 'TR', 'BR', 'BL'],
                nrows=None, ncols=None):
        # The strategy here is to create two sub rectangles and merge their positions
        # to create full scale
        super().__init__(objective=objective)
        self.nrows = nrows
        self.ncols = ncols
        self.corner_names = corner_names
        self.left_half_motion = RectGridMotion(objective=objective, movement_type='left',
                                corner_names=corner_names)
        self.right_half_motion = RectGridMotion(objective=objective, movement_type='right',
                                corner_names=corner_names)

        self.positions = None
    
    def set_rows(self, rows):
        self.left_half_motion.set_rows(rows)
        self.right_half_motion.set_rows(rows)
        self.nrows = rows
    
    def set_cols(self, cols):
        self.left_half_motion.set_cols(cols)
        self.right_half_motion.set_cols(cols)
        self.ncols = cols

    
    def set_corner_position(self, label, position_dict):
        """
        Function that will set the positions of the corners,
        by passing one at a time. It is used if you set the positions
        from a UI by grabbing the current location.
        Arguments:
            label: one of 'TL1', 'TR1', 'BR1', 'BL1', 'TL2', 'TR2', 'BR2', 'BL2'
            position_dict: dict with keys 'X', 'Y', 'Z', 'grid_row', 'grid_col', 'label'
        """
        if label[:2] not in self.corner_names:
            raise ValueError(f"Position label not in the corner label list, found: {label}...")
        else:
            # check if all keys are in the position dict
            keys = ['x', 'y', 'z', 'grid_row', 'grid_col', 'label']
            for key in keys:
                if key not in position_dict:
                    raise ValueError(f"Key {key} not found in position dict: {position_dict}")
            if int(label[2:]) == 1:
                self.left_half_motion.corner_pos_dict[label[:2]] = position_dict
            elif int(label[2:]) == 2:
                self.right_half_motion.corner_pos_dict[label[:2]] = position_dict
    
    def construct_grid(self, starting_position_no=1):
        if len(self.left_half_motion.corner_pos_dict) != 4 or len(self.right_half_motion.corner_pos_dict) != 4:
            raise ValueError(f"All 8 corners not set .. check that everything is set")

        print(self.left_half_motion.corner_pos_dict)
        print(self.right_half_motion.corner_pos_dict)
        
        self.left_half_motion.construct_grid(starting_position_no)
        n_left_positions = len(self.left_half_motion.positions)
        self.right_half_motion.construct_grid(starting_position_no + n_left_positions)

        left_positions = self.left_half_motion.positions
        right_positions = self.right_half_motion.positions
        self.positions = left_positions + right_positions
        print(n_left_positions, len(self.positions))

    @classmethod
    def parse(cls, param):
        pass
    
    def plot_motion_plan(self):
        plt.figure()
        plt.close()
        
class MotionFromFile(Motion):
    """
    Motion pattern determined by the order of positions laid out in 
    the position lists of micromanager, or however they are in the file

    Args:
        objective: Magnification of the objective (20, 40 or 100)

    """ 
    def __init__(self, filename: Union[str, pathlib.Path], objective: int = 100):
        super().__init__(objective=objective)
        self.filename = filename if isinstance(filename, pathlib.Path) else Path(filename)

        self.microscope_props, self.positions = self.parse_position_file(filename)

    @classmethod
    def parse(cls, param):
        pass

    @staticmethod
    def parse_position_file(filename: Union[str, pathlib.Path]):
        """
        Takes in positions filename from micromanager stage position list
        and then spits out a list of positions and microscope properties

        Args:
            filename (Union[str, pathlib.Path]): .pos filename from micromanger 
                stage positions list
        """
        filename = filename if isinstance(filename, pathlib.Path) else Path(filename)
        
        if not filename.exists():
            raise FileNotFoundError(f"Filename {filename} not found")

        if filename.suffix != '.pos':
            raise ValueError(f"Position file {filename} doesn't have .pos extension")
        
        # log file found
        sys.stdout.write(f"{filename} - file read for positions ..\n")
        sys.stdout.flush()
        
        with open(filename, 'r') as json_handle:
            data = json.load(json_handle)

        # log micromanager file read and print the version of 
        sys.stdout.write(f"Micro-manager version: {data['major_version']} - {data['minor_version']}\n")
        sys.stdout.flush()

        positions_dicts = data['map']['StagePositions']['array']

        if len(positions_dicts) == 0:
            raise ValueError("No positions found in file")
        
        defaultXYStage = positions_dicts[0]['DefaultXYStage']['scalar']
        defaultZStage = positions_dicts[0]['DefaultZStage']['scalar']

        microscope_props = {
            'XYStage': defaultXYStage,
            'ZStage': defaultZStage
        }
        # Find the index of the device in the list 
        XYDeviceIndex = None
        ZDeviceIndex = None
        first_position = positions_dicts[0]['DevicePositions']['array']
        for i, item in enumerate(first_position, 0):
            if item['Device']['scalar'] == defaultXYStage:
                XYDeviceIndex = i
            elif item['Device']['scalar'] == defaultZStage:
                ZDeviceIndex = i
        
        sys.stdout.write(f"DefaultXYStage device: {defaultXYStage} at index {XYDeviceIndex} found ...\n")
        sys.stdout.write(f"DefaultZStage device: {defaultZStage} at index {ZDeviceIndex} found ...\n")
        sys.stdout.flush()

        positions = []
        for position_item in positions_dicts:
            X = position_item['DevicePositions']['array'][XYDeviceIndex]['Position_um']['array'][0] # XYStage device is on list item 1
            Y = position_item['DevicePositions']['array'][XYDeviceIndex]['Position_um']['array'][1] # XYStage device is on list item 1
            Z = position_item['DevicePositions']['array'][ZDeviceIndex]['Position_um']['array'][0] # PFSoffset device is on list item 0
            grid_row = position_item['GridRow']['scalar']
            grid_col = position_item['GridCol']['scalar']
            label = position_item['Label']['scalar']
            positions.append({'x': X, 'y': Y, 'z': Z, 'grid_row': grid_row,
                            'grid_col': grid_col, 'label': label})
    
        # find the ranges of x and y to make it easier to write
        # movement bounds
        x_values = [item['x'] for item in positions]
        y_values = [item['y'] for item in positions]
        z_values = [item['z'] for item in positions]
        
        microscope_props['x_range'] = (min(x_values), max(x_values))
        microscope_props['y_range'] = (min(y_values), max(y_values))
        microscope_props['z_range'] = (min(z_values), max(z_values))

        positions = sorted(positions, key= lambda x: int(x['label'][3:]))
        
        
        return microscope_props, positions
        
    
    def plot_motion_plan(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.gca().invert_yaxis()
        x_min, x_max = self.microscope_props['x_range']
        y_min, y_max = self.microscope_props['y_range']
        ax.set_xlim(x_min-100, x_max+100)
        ax.set_ylim(y_min-100, y_max+100)
        for i, position in enumerate(self.positions, 0):
            #print(position['x'], position['y'])
            circle = plt.Circle((position['x'], position['y']), 5, color='r')
            ax.add_patch(circle)
            if i == len(self.positions) - 1:
                break
            else:
                # draw arrows
                dx = self.positions[i+1]['x'] - self.positions[i]['x']
                dy = self.positions[i+1]['y'] - self.positions[i]['y']
                ax.arrow(position['x'], position['y'], dx, dy, head_width=25,
                        head_length=25, length_includes_head=True)
        
        ax.set_title('XY positions map')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        plt.tight_layout()
        plt.show()
    
