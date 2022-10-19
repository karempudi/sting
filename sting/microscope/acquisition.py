import pycromanager
from typing import Union
from sting.utils.types import RecursiveNamespace
from itertools import cycle
from sting.microscope.motion import MotionFromFile 

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

        self.events = [1, 2, 3, 4, 5, 6, 7, 8]
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