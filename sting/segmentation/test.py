import argparse
import pathlib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from sting.utils.param_io import ParamHandling, load_params, save_params
from art import tprint
import torch
from torch.utils.data import Dataset, DataLoader
from sting.utils.types import RecursiveNamespace
from sting.utils.hardware import get_device_str
import time
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Segmentation testing arguments')

    parser.add_argument('-p', '--param_file',
                        help='Specify the param file for testing segmentation network',
                        required=True)
    
    parser.add_argument('-d', '--device', default=None,
                        help='Specify the device string (cpu, cuda, cuda:0 or cuda:1)',
                        type=str, required=False)
    
    parser.add_argument('-w', '--num_workers_override', default=6,
                        help='Override number of workers for pytorch dataloader.',
                        type=int, required=False)
    
    parser.add_argument('-i', '--images_dir_override', default=None,
                        help='Set directory containing images to override',
                        type=str, required=False)
    
    parser.add_argument('-s', '--save_dir_override', default=None,
                        help='Set directory to save segmentation masks to override',
                        type=str, required=False)
    
    parser.add_argument('-m', '--model_path', default=None,
                        help='Set model path to override',
                        type=str, required=False)
    
    args = parser.parse_args()
    
    return args

def test_model(param_file: str, device_overwrite: str = None,
                num_workers_override: int = None, 
                images_dir_override: str = None,
                save_dir_override: str = None,
                model_path_override: str = None):
    """
    Test a segmentation network model based on parameters given

    Args:
        param_file (str): parameter file path, includes everything needed to
            load a segmentation model for segmentaiton of images 
        device_overwrite (str): overwrite cuda device specified in the param_file
        num_workers_override (int): overwrite num of workers in pytorch dataloaders
        images_dir_override (str): overwrite images_dir in the param_file
        save_dir_override (str): overwrite save_dir in the param_file

        Note:
            All the parameters that are needed will be in the Testing section of 
            the parameter file. They should have everything needed to load the model
            and run it.

    """
    # Loading the param
    param_file = Path(param_file)
    if not param_file.exists():
        raise FileNotFoundError(f"Parameter file {str(param_file)} doesn't exist")
    
    # make a param namespace and fill in the defaults
    param = load_params(param_file, ref_type='segment')

    # set hardware device to test one
    if device_overwrite is not None:
        param.Testing.Hardware.device = device_overwrite
        device = device_overwrite
    else:
        device = param.Testing.Hardware.device
    
    # check for cuda
    if torch.cuda.is_available():
        _, device_idx = get_device_str(device)
        if device_idx is not None:
            torch.cuda.set_device(device)
    else:
        device = 'cpu'

    torch.set_num_threads(param.Testing.Hardware.torch_threads)

    if num_workers_override is not None:
        param.Testing.Hardware.num_workers = num_workers_override
    
    if images_dir_override is not None:
        param.Testing.images_dir = images_dir_override

    if save_dir_override is not None:
        param.Testing.save_dir = save_dir_override

    if model_path_override is not None:
        param.Testing.model_path = model_path_override
    
    expt_id = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    expt_dir = Path(param.Testing.save_dir) / Path(expt_id)

    if not expt_dir.parent.exists():
        expt_dir.parent.mkdir()
    
    if not expt_dir.exists():
        expt_dir.mkdir()
    
    param_used_save = expt_dir / Path('testing_params').with_suffix(param_file.suffix)
    save_params(param_used_save, param.Testing)

    # setup dataset and dataloader
    test_ds = None
    test_dl = None

    print(param)
    # setup network

    # test loop

def setup_tester(param, device):
    pass
    

def main():
    print("Hello from segmentation testing")
    tprint("SEGMENT")
    args = parse_args()

    test_model(args.param_file, 
                device_overwrite=args.device,
                num_workers_override=args.num_workers_override,
                images_dir_override=args.images_dir_override,
                save_dir_override=args.save_dir_override,
                model_path_override=args.model_path)

if __name__ == "__main__":
    main()