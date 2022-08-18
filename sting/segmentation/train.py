import argparse
from pathlib import Path
from sting.utils.param_io import ParamHandling, load_params

import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Segmentation networks Training Arguments")

    parser.add_argument('-p', '--param_file', 
            help='Specify the param file for training segmentation network.',
            required=True)
    
    parser.add_argument('-d', '--device',
            help='Specify the device string (cpu, cuda, cuda:0 or cuda:1) and overwrite parameters.',
            type=str, required=False)
    
    parser.add_argument('-w', '--num_workers_override',
            help='Override number of workers of pytorch dataloaders.', type=int)
    
    parser.add_argument('-l', '--log_dir', default='segment_runs',
            help='Set directory name where you want the logs to go in.')
    
    parser.add_argument('-c', '--log_comment', default=None,
            help='Added a log comment to the run')

    args = parser.parse_args()

    return args

def train_model(param_file: str, device_overwrite: str = None,
            num_workers_overwrite: int = None,
            log_dir: str = 'segment_runs', log_comment: str = None):
    """
    Train a segmentation model based on the parameters given

    Args:
        param_file (str): parameter file path, includes everything need to 
                train a segmentation network, U-net, Omnipose and their 
                variants for both cell and channel detection
        device_overwrite (str): overwrite cuda device specified in param_file
        num_workers_overwrite (int): overwrite num of workers in pytorch dataloaders
        log_dir (str): directory for logs used by tensorboard
        log_comment (str): comment to the experiment

    """
    param_file = Path(param_file)
    if not param_file.exists():
        raise FileNotFoundError(f"Parameter file {str(param_file)} doesn't exist")
        
    param = load_params(param_file, 'segment')
    print(param)

def main():
    print("Hello from segmentation training")
    args = parse_args()

    train_model(args.param_file)

if __name__ == "__main__":
    main()