import pycromanager
import pathlib
from pathlib import Path
from typing import Union
from sting.utils.types import RecursiveNamespace
from itertools import cycle
from sting.microscope.motion import MotionFromFile, RectGridMotion, TwoRectMotion 
from tifffile import imread

class ExptAcquisition(object):
    """
    
    Class describing the sequence of events that 
    takes place in the acquisition of images during an
    experiment.

    Basically we need to be able to iterate and construct 
    events to pass on to pycromanager using this class.
    If expt_acq is an instance of ExptAcquision class, 
    next(expt_acq) should give the next event to send to pycromanager
    and bail out once you reached maximum number of events, which is set
    to a large number. This is the maximum number of events one can acquire.

    """
    
    def __init__(self, param: Union[dict, RecursiveNamespace]):
        # depending on the motion pattern and microscope preset 
        # behaviour construct
        self.event_params = param.Experiment.Acquisition.events
        # based on motion type we interpret the positions filename
        self.motion_type = self.event_params.motion_type
        if self.motion_type == 'all_from_file':
            motion = MotionFromFile()
        elif self.motion_type == 'one_rect_from_file':
            motion = RectGridMotion()
        elif self.motion_type = 'two_rect_from_file':
            motion = TwoRectMotion

        self.positions = motion.positions
        self.microscope_props = motion.microscope_props

        # presets 
        self.presets = self.event_params.presets

        # which positions should have slow preset loaded 
        # used for positions that are far away 
        self.slow_positions = None

        # construct events 
        self.events = []

        #
        self.max_events = 15
        self.cycle = cycle(self.events)
        self.events_sent = 0

    @classmethod
    def parse(cls, param):
        return cls(param)

    def __iter__(self):
        return self

    def __next__(self):
        if self.events_sent < self.max_events:
            self.events_sent += 1 
            x = next(self.cycle)
            return x
        else:
            raise StopIteration

def construct_events():
    pass

class simAcquisition(object):

    def __init__(self, param: Union[dict, RecursiveNamespace]):
        self.dir = Path(param.Experiment.sim_directory)
        self.filenames = sorted(list(self.dir.glob('*.tif*')))
        self.current_filenumber = 0

    @classmethod
    def parse(cls, param):
        return cls(param)
    
    def __len__(self):
        return len(self.filenames)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_filenumber < len(self.filenames):
            img = imread(self.filenames[self.current_filenumber]).astype('float32')
            self.current_filenumber += 1
            return img
        else:
            #raise StopIteration
            return None
