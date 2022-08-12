
# File containing functions that launch different process and manages
# the data sharing between them
import pathlib
from pathlib import Path
from typing import Union
from sting.utils.types import RecursiveNamespace
import torch.multiprocessing as tmp

class ExptRun(object):
    """
    Experiment run object that hold all the queues for 
    managing the data produced during the run and process them
    appropriately, based on the settings provided by the parameter
    file.
    """
    def __init__(self, param: Union[RecursiveNamespace, dict]):
        self.param = param

        # create queues based on what is there in the parameters

    def load_nets(self):
        pass
    
    def acquire(self):
        pass
    
    def acquire_sim(self):
        pass
    
    def segment(self):
        pass
    
    def track(self):
        pass
    
    def growth_rates(self):
        pass

        

def start_live_experiment(expt_run: ExptRun, param: Union[RecursiveNamespace, dict],
                    sim: bool=True):
    """
    Function that starts the processes in the experiment run object 
    created whether you use the UI or run from command line

    Args:
        expt_run (ExptRun): an instance of ExptRun that hold all the 
            handles for the queues and processes.
        param (RecursvieNamespace): param file used to create the ExptRun
            instance.
        sim (bool): set (False) if you run real experiment, set (True) 
            for simulation/debugging, defaults to True.
    """
    try:
        tmp.set_start_method('spawn')
    except:
        pass
    
    # Depending on the parameters passed, start various processes of 
    # the experiment run object 


