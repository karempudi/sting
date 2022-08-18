import pycromanager
from typing import Union
from sting.utils.types import RecursiveNamespace


class ExptAcquisition(object):
    """
    
    Class describing the sequence of events that 
    takes place in the acquisition of images during an
    experiment.

    Basically we need to be able to iterate and construct 
    events to pass on to pycromanager using this class


    """
    
    def __init__(self, param: Union[dict, RecursiveNamespace]):
        pass

    @classmethod
    def parse(cls, param):
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass
