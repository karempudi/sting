import pycromanager
import pathlib
from pathlib import Path
from typing import Union
from sting.utils.types import RecursiveNamespace
from itertools import cycle, repeat
from sting.microscope.motion import MotionFromFile, RectGridMotion, TwoRectMotion 
from tifffile import imread
import numpy as np

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

        self.min_loop_start_times = list(np.arange(self.rules.start, self.rules.end, self.rules.every))
        self.n_loops = len(self.min_loop_start_times)
        self.loop_interval = self.rules.every * self.time_factor

        for loop_no in range(self.n_loops):
            for i, position_data in enumerate(self.positions, 0):
                event = {}
                # grab position number from the 
                event['axes']  = {'time': loop_no, 'position': int(position_data['label'][3:])}
                event['x'] = position_data['x']
                event['y'] = position_data['y']
                event['z'] = position_data['z']
                event['channel'] = {'group': self.rules.group, 'config': self.rules.preset}
                event['exposure'] = self.rules.exposure 
                event['min_start_time'] = self.min_loop_start_times[loop_no] * self.time_factor
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
        self.max_events = len(self.events)

        self.events_sent = 0
        self.loop_number = 0

    @classmethod
    def parse(cls, param):
        return cls(param)
    
    def __getitem__(self, idx):
        return self.events[idx]
    
    def __len__(self):
        return len(self.events)

    def get_events(self):
        return self.events

class simAcquisition(object):

    def __init__(self, param: Union[dict, RecursiveNamespace]):
        self.dir = Path(param.Experiment.sim_directory)
        self.filenames = sorted(list(self.dir.glob('*.tif*')))
        self.n_files = len(self.filenames)

        self.cycle = cycle(self.filenames)

        self.max_loops = 1

        self.events_sent = 0
        self.loop_number = 0

    @classmethod
    def parse(cls, param):
        return cls(param)
    
    def __len__(self):
        return len(self.filenames)

    def __iter__(self):
        return self

    def __next__(self):
        if self.loop_number < self.max_loops:
            self.events_sent += 1
            current_filename = next(self.cycle)
            img = imread(current_filename).astype('float32')
            #timepoint = self.loop_number
            timepoint = self.events_sent
            if self.events_sent % self.n_files == 0:
                self.loop_number += 1
                position = self.n_files
            else:
                position = self.events_sent % self.n_files

            position = 1
            return {
                'image': img,
                'position': position,
                'timepoint' : timepoint - 1
            }

        else:
            #raise StopIteration
            return None

class simFullExpt(object):

    def __init__(self, param: Union[dict, RecursiveNamespace],
                n_positions=10):

        self.dir = Path(param.Experiment.sim_directory)
        self.filenames = sorted(list(self.dir.glob('*.tif*')))
        self.n_timepoints = len(self.filenames)
        self.n_positions = n_positions
        self.pos_time = []
        for j in range(self.n_timepoints):
            for i in range(1, self.n_positions+1):
                self.pos_time.append((j, i))
        self.max_events = len(self.pos_time)
        self.events_sent = 0
    @classmethod
    def parse(cls, param):
        return cls(param)

    def __len__(self):
        return len(self.pos_time)

    def __iter__(self):
        return self

    def __next__(self):
        if self.events_sent < self.max_events: 
            time, pos = self.pos_time[self.events_sent]
            self.events_sent += 1
            current_filename = self.filenames[time]
            img = imread(current_filename).astype('float32')
            return {
                'image': img,
                'position': pos,
                'timepoint': time,
                'last': (pos == self.n_positions)
            }
        else:
            return None