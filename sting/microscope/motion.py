from abc import ABC, abstractmethod
import pycromanager
import numpy as np
import matplotlib.pyplot as plt

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
        
    