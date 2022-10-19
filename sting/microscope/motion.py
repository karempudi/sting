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

    def __init__(self, objective: int = 100):
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
        
        (TL) ---------- (TR)
         |               |
         |               |
         |               |
         |               |
         |               |
         |               |
        (BL) ---------- (BR)


        Args:
            objective: Magnification of the objective (20, 40 or 100)
            limits: (XYHW) limits
            direction_traversal: horizontal or vertical, the movement between positions
                will always follow a snake pattern. (Plot to verify)
    """    

    def __init__(self, objective: int = 100, limits: tuple = None,
                direction_traversal: str = 'horizontal'):
        super().__init__(objective=objective)
        self.limits = limits
        self.direction_traversal = direction_traversal

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

    

class TwoRectMotion(Motion):
    """
    Two Rect Motion pattern used to image tweezer chip, that has two RectGridMotion patterns
    stitched together.

    Args:
        objective: Magnification of the objective (20, 40 or 100)
        limits: (XYHW) limits
        direction_traversal: horizontal or vertical, the movement between positions
            will always follow a snake pattern. The motion plan will follow
    """
    
    def __init__(self):
        pass

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
        positions = []
        for position_item in positions_dicts:
            X = position_item['DevicePositions']['array'][1]['Position_um']['array'][0] # XYStage device is on list item 1
            Y = position_item['DevicePositions']['array'][1]['Position_um']['array'][1] # XYStage device is on list item 1
            Z = position_item['DevicePositions']['array'][0]['Position_um']['array'][0] # PFSoffset device is on list item 0
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
    
