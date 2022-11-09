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
            motion = MotionFromFile(self.event_params.pos_filename)
        elif self.motion_type == 'one_rect_from_file':
            motion = RectGridMotion()
        elif self.motion_type == 'two_rect_from_file':
            motion = TwoRectMotion()

        self.positions = motion.positions
        self.microscope_props = motion.microscope_props

        # presets 
        self.available_presets = self.event_params.available_presets

        # which positions should have slow preset loaded 
        # used for positions that are far away 
        self.slow_positions = None

        # construct events 
        self.events = []
        # only one rule so far, if more loop
        self.rules = self.event_params.rules
        if self.rules.units == 'seconds':
            self.time_factor = 1
        elif self.rules.units == 'minutes':
            self.time_factor = 60

        self.min_loop_start_times = list(range(self.rules.start, self.rules.end, self.rules.every))
        self.n_loops = len(self.min_loop_start_times)
        self.loop_interval = self.rules.every * self.time_factor

        for i, position_data in enumerate(self.positions, 0):
            event = {}
            # grab position number from the 
            event['axes']  = {'time': 0, 'position': int(position_data['label'][3:])}
            event['x'] = position_data['x']
            event['y'] = position_data['y']
            event['z'] = position_data['z']
            event['channel'] = {'group': self.rules.group, 'config': self.rules.preset}
            event['exposure'] = self.rules.exposure 
            event['min_start_time'] = 0
            self.events.append(event)
        if self.rules.slow_positions == 'first':
            self.events[0]['channel']['config'] = self.rules.slow_preset
        elif self.rules.slow_positions == 'last':
            self.events[-1]['channel']['config'] = self.rules.slow_preset
        elif self.rules.slow_positions == 'auto':
            # write checks for max distance between consecutive positions and 
            # set approriately
            pass
        elif self.rules.slow_positions == 'none':
            pass

        # calculate max events based on the rules
        self.max_events = self.n_loops * len(self.events)
        self.cycle = cycle(self.events)

        self.events_sent = 0
        self.loop_number = 0

    @classmethod
    def parse(cls, param):
        return cls(param)

    def __iter__(self):
        return self

    def __next__(self):
        if self.events_sent < self.max_events:
            self.events_sent += 1 
            x = next(self.cycle)
            x['axes']['time'] = self.loop_number
            x['min_start_time'] = self.loop_number * self.loop_interval # has to be changed if the loop changes
            if self.events_sent % len(self.events) == 0:
                # we completed one loop
                self.loop_number += 1
            return x
        else:
            raise StopIteration
            return None

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
