
# File containing functions that launch different process and manages
# the data sharing between them
import sys
import time
import pathlib
import logging
from pathlib import Path
from typing import Union
from datetime import datetime
from sting.utils.types import RecursiveNamespace
import torch.multiprocessing as tmp
from sting.utils.logger import logger_listener, setup_root_logger
from sting.utils.db_ops import create_databases, write_to_db, read_from_db

class ExptRun(object):
    """
    Experiment run object that hold all the queues for 
    managing the data produced during the run and process them
    appropriately, based on the settings provided by the parameter
    file.
    """
    def __init__(self, param: Union[RecursiveNamespace, dict]):
        self.param = param

        expt_id = self.param.Experiment.number
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        expt_id = expt_id + '-' + timestamp
        # You have to manually create save directory just to be safe..
        # main analysis directory is not autmoatically created. You have to
        # manually do it to be safe
        expt_save_dir = Path(self.param.Save.directory) / Path(expt_id)
        if not expt_save_dir.parent.exists():
            raise FileNotFoundError(f"Experiment save directory {expt_save_dir.parent} doesn't exist\n")
        
        expt_save_dir.mkdir(exist_ok=False)
        sys.stdout.write(f"Experiment save directory: {expt_save_dir} created succesfully ... \n")
        sys.stdout.flush()

        self.expt_save_dir = expt_save_dir


        # create databases to write stuff to depending on the queues preset in 
        # the experiment
        create_databases(self.expt_save_dir, param.Experiment.queues)

        self.logger_queue = tmp.Queue(-1)

        self.logger_kill_event = tmp.Event()

        # create queues based on what is there in the parameters
        # and also kill events
        if 'acquire' in self.param.Experiment.queues:
            self.acquire_kill_event = tmp.Event()
            if self.param.Experiment.events.pass_one_and_wait:
                self.acquire_queue = tmp.Queue()
            else:
                self.acquire_events = None
            # create a motion object that holds 
            # the motion pattern

        if 'segment' in self.param.Experiment.queues:
            self.segment_kill_event = tmp.Event()
            # set up segmentation queue
            self.segmentQueue = tmp.Queue()

        if 'track' in self.param.Experiment.queues:
            self.track_kill_event = tmp.Event()
            # set up tracker queue
            self.trackerQueue = tmp.Queue()

        if 'growth' in self.param.Experiment.queues:
            self.growth_kill_event = tmp.Event()
            # set up growth queue
            self.growthQueue = tmp.Queue()

            
    @staticmethod
    def make_events(pos_filename):
        pass

    def load_nets(self):
        pass

    def logger_listener(self):
        setup_root_logger(self.param, self.expt_save_dir)
        name = tmp.current_process().name
        print(f"Starting {name} process ..")
        while not self.logger_kill_event.is_set():
            try:
                record = self.logger_queue.get()
                if record is None:
                    break
                logger = logging.getLogger(record.name)
                logger.handle(record)

            except KeyboardInterrupt:
                self.logger_kill_event.set()
                sys.stdout.write("Logger process interrupted using keyboard\n")
                sys.stdout.flush()
                break

    def set_process_logger(self):
        h = logging.handlers.QueueHandler(self.logger_queue)
        root = logging.getLogger()
        root.addHandler(h)
        root.setLevel(logging.DEBUG)
    
    def acquire(self):
        # configure acquire logger
        self.set_process_logger()
        name = tmp.current_process().name
        print(f"Starting {name} process ..")
        # keep the process alive
        while not self.acquire_kill_event.is_set():
            try:
                time.sleep(1.0)
                logger = logging.getLogger(name)
                logger.log(logging.INFO, "Acquire wait 1.0s")
            except KeyboardInterrupt:
                self.acquire_kill_event.set()
                sys.stdout.write("Acquire process interrupted using keyboard\n")
                sys.stdout.flush()
                break
                
    
    def acquire_sim(self):
        # configure acquire_sim logger
        self.set_process_logger()
        name = tmp.current_process().name
        print(f"Starting {name} simulation process ..")
        # keep process alive
        position = 0
        while not self.acquire_kill_event.is_set():
            timepoint = 0
            try:
                time.sleep(1.0)
                logger = logging.getLogger(name)
                logger.log(logging.INFO, "Acquire wait 1.0s")
                write_to_db({'position': position, 'timepoint': 0}, self.expt_save_dir, 'acquire')
                position += 1
            except KeyboardInterrupt:
                self.acquire_kill_event.set()
                sys.stdout.write("Acquire process interrupted using keyboard\n")
                sys.stdout.flush()
                break
 
    
    
    def segment(self):
        # configure segment logger
        self.set_process_logger()
        name = tmp.current_process().name
        print(f"Starting {name} process ..")
        # keep the process alive
        while not self.segment_kill_event.is_set():
            try:
                time.sleep(1.0)
                logger = logging.getLogger(name)
                logger.log(logging.INFO, "Segment wait 1.0s")
            except KeyboardInterrupt:
                self.segment_kill_event.set()
                sys.stdout.write("Segment process interrupted using keyboard\n")
                sys.stdout.flush()
                break
    
    def track(self):
        # configure track logger
        self.set_process_logger()
        name = tmp.current_process().name
        print(f"Starting {name} process ..")
        # keep the process alive
        while not self.track_kill_event.is_set():
            try:
                time.sleep(1.0)
                logger = logging.getLogger(name)
                logger.log(logging.INFO, "Track wait 1.0s")
            except KeyboardInterrupt:
                self.track_kill_event.set()
                sys.stdout.write("Track process interrupted using keyboard\n")
                sys.stdout.flush()
                break
        
    def growth_rates(self):
        # configure growth logger
        self.set_process_logger()
        name = tmp.current_process().name
        print(f"Starting {name} process ..")
       # keep the process alive
        while not self.growth_kill_event.is_set():
            try:
                time.sleep(1.0)
                logger = logging.getLogger(name)
                logger.log(logging.INFO, "Growth wait 1.0s")
            except KeyboardInterrupt:
                self.growth_kill_event.set()
                sys.stdout.write("Growth process interrupted using keyboard\n")
                sys.stdout.flush()
                break


    def stop(self,):
        # should kill all processes
        self.logger_kill_event.set()

        if 'acquire' in self.param.Experiment.queues:
            self.acquire_kill_event.set()

        if 'segment' in self.param.Experiment.queues:
            self.segment_kill_event.set()
            
        if 'track' in self.param.Experiment.queues:
            self.track_kill_event.set()
        
        if 'growth' in self.param.Experiment.queues:
            self.growth_kill_event.set()
        

def worker_configurer(queue):
    h = logging.handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(logging.DEBUG)

def worker_process(queue, configurer):
    configurer(queue)

    
        

def start_live_experiment(expt_run: ExptRun, param: Union[RecursiveNamespace, dict],
                    sim: bool=True, from_cmdline: bool=False):
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
    tmp.freeze_support()
    try:
        tmp.set_start_method('spawn')
    except:
        pass
    
    # Depending on the parameters passed, start various processes of 
    # the experiment run object 
    
    # logger queue will always be setup
    try:
            
        expt_run.logger_kill_event.clear()
        logger_process = tmp.Process(target=expt_run.logger_listener,
                                    name='logger')
        logger_process.start()

        # Setup queues and processes to operate on them based on params
        if 'acquire' in param.Experiment.queues:
            # create and call an acquire process 
            # that run the acquisition and put the results in segment queue
            expt_run.acquire_kill_event.clear()
            if sim:
                acquire_process = tmp.Process(target=expt_run.acquire_sim, name='acquire')
            else:
                acquire_process = tmp.Process(target=expt_run.acquire, name='acquire')
            acquire_process.start()

        if 'segment' in param.Experiment.queues:
            # create and call segmentation process
            expt_run.segment_kill_event.clear()
            segment_process = tmp.Process(target=expt_run.segment, name='segment')
            segment_process.start()
            
        if 'track' in param.Experiment.queues:
            # create and call tracking process
            expt_run.track_kill_event.clear()
            track_process = tmp.Process(target=expt_run.track, name='track')
            track_process.start()

        if 'growth' in param.Experiment.queues:
            # create and call growth rates process
            expt_run.growth_kill_event.clear()
            growth_process = tmp.Process(target=expt_run.growth_rates, name='growth')
            growth_process.start()

        if from_cmdline:
            logger_process.join()
            if 'acquire' in param.Experiment.queues:
                acquire_process.join()
            if 'segment' in param.Experiment.queues:
                segment_process.join()
            if 'track' in param.Experiment.queues:
                track_process.join()
            if 'growth' in param.Experiment.queues:
                growth_process.join()
        
        
    except KeyboardInterrupt:
        pass
        
        
