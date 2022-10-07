import argparse
import shutil
import socket
from pathlib import Path
from datetime import datetime
from art import tprint
from sting.utils.param_io import ParamHandling, load_params, save_params
from sting.utils.hardware import get_device_str
from sting.segmentation import logger
import torch
from torch.utils.data import Dataset, DataLoader
from sting.segmentation.networks import model_dict 

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

    # Loading parameters for network training from the parameter file
    param_file = Path(param_file)
    if not param_file.exists():
        raise FileNotFoundError(f"Parameter file {str(param_file)} doesn't exist")
    # make a param namespace and fill in defaults
    param = load_params(param_file, 'segment')
    print(param)

    # Load ckpt if the training stops and you need to restart 
    if param.Checkpoints.load_ckpt == False:
        expt_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        from_ckpt = False
    else:
        from_ckpt = True
        expt_id = Path(param.Checkpoints.ckpt_filename).parent.name

    if log_comment:
        expt_id = expt_id + '_' + log_comment

    if not from_ckpt:
        # create new expt dir
        expt_dir = Path(param.Save.directory) / Path(expt_id)
    else:
        expt_dir = Path(param.Checkpoints.ckpt_filename).parent
    
    if not expt_dir.parent.exists():
        expt_dir.parent.mkdir()

    # create dir for expt if not loaded from ckpt
    if not from_ckpt:
        expt_dir.mkdir(exist_ok=False)

    model_out = expt_dir / Path('model.pth')
    ckpt_path = expt_dir / Path('ckpt.pth')

    # save parameter file used in the directory of the experiment
    param_in_save = expt_dir / Path('training_run_set').with_suffix(param_file.suffix)
    shutil.copy(param_file, param_in_save)

    # training run params after filling the defaults
    param_used_save = expt_dir/ Path('training_run_used').with_suffix(param_file.suffix)
    save_params(param_used_save, param)

    # set hardware device to train
    if device_overwrite is not None:
        param.Hardware.device = device_overwrite
        device = device_overwrite
    else:
        device = param.Hardware.device
    
    # check for cuda
    if torch.cuda.is_available():
        _, device_idx = get_device_str(device)
        if device_idx is not None:
            torch.cuda.set_device(device) 
    else:
        device = 'cpu'
    
    torch.set_num_threads(param.Hardware.torch_threads)

    # Log system
    log_dir = Path(param.Logging.directory) / Path(log_dir)

    # setup datasets

    # setup models and optimizers, loss functions, etc
    #print(model)

    #dl_train, dl_val, dl_test = setup_dataloader(param, ds_train, ds_val, ds_test)


def setup_trainer(param):
    pass

def setup_dataloader(param, train_ds, val_ds=None, test_ds=None):
    """
    Setup dataloaders for the datasets
    """
    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=param.HyperParameters.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=param.Hardware.num_workers,
        pin_memory=True,
    )
    if val_ds is not None:
        val_dl = DataLoader(
            dataset=val_ds,
            batch_size=param.HyperParameters.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=param.Hardware.num_workers,
            pin_memory=False
        )
    else:
        val_dl = None

    if test_ds is not None:
        test_dl = DataLoader(
            dataset=test_ds,
            batch_size=param.HyperParameters.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=param.Hardware.num_workers,
            pin_memory=False
        )
    else:
        test_dl = None
    
    return train_dl, val_dl, test_dl


def main():
    print("Hello from segmentation training")
    tprint("SEGMENT")
    args = parse_args()

    #train_model(args.param_file)

if __name__ == "__main__":
    main()