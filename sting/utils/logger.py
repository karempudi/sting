import logging
import logging.handlers
import torch.multiprocessing as tmp
from .types import RecursiveNamespace

def setup_root_logger(param: RecursiveNamespace):
    """
    
    Setting up the properties of the root logger of 
    an experiment

    Args:
        param (RecursiveNamespace) : Namespace of the parameters
            used for the experiment
    """
    root = logging.getLogger()
    f = logging.Formatter('%(asctime)s %(processName)-10s %(levelname)-8s %(messages)s')
    if param.Logging.to_file:
        h = logging.FileHandler(filename=param.Logging.directory, mode='w')
        h.setFormatter(f)
        root.addHandler(h)

    if param.Logging.to_console:
        c = logging.StreamHandler()
        c.setFormatter(f)
        root.addHandler(c)

def logger_listener(queue, param):
    setup_root_logger(param)
    while True:
        try:
            record = queue.get()
            if record is None:
                break
            
            logger = logging.getLogger(record.name)
            logger.handle(record)
        except Exception:
            import sys, traceback
            print('Whoops! Logger has a problem: ', file=sys.stderr)
            traceback.print_exec(file=sys.stderr)
    