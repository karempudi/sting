import argparse
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from art import tprint

def parse_args():
    parser = argparse.ArgumentParser(description='Region detection testing arguments')
    
    parser.add_argument('-p', '--param_file', 
                        help='Specify the param file for testing region detection network',
                        required=True)

    parser.add_argument('-d', '--device', default=None, 
                        help='Specify the device string (cpu, cuda, cuda:0, or cuda:1)',
                        type=str, required=False)
    
    parser.add_argument('-w', '--num_workers_override', default=6,
                        help='Override number of workers for pytorch dataloader.',
                        type=int, required=False)
    
    parser.add_argument('-i', '--images_dir_override', default=None,
                        help='Set directory containing images to override',
                        type=str, required=False)
    
    args = parser.parse_args()

    return args

def test_model(param_file: str, device_overwrite: str = None,
                num_workers_override: int = None,
                images_dir_override: str = None):
    """
    Test a region detection (barcode) model based on the parameters given

    Args:
        param_file (str): parameter file path, includes everything needed to 
            load a yolo series model for bounding box detection on the images
        device_overwrite (str): overwrite cuda device specified in the param_file
        num_workers_override (int): overwrite num of workers in pytorch dataloaders
        images_dir_override (str): overwrite images_dir in the param file 


        Note:
            the parameters file is the same as yolo training config, but 
            all the parameters will be read from the testing section of the 
            file instead, everything else will be ignored

    """
    # Loading the param

def main():
    print("Hello from region detection testing")
    tprint("BARCODE TEST")
    args = parse_args()

    test_model(args.param_file,)

if __name__ == "__main__":
    parse()
