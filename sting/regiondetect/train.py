import argparse
import pathlib
import shutil
from pathlib import Path
from datetime import datetime
from sting.utils.param_io import load_params, save_params, ParamHandling
from sting.regiondetect.logger import SummaryWriter
from sting.regiondetect.datasets import BarcodeDataset
from sting.regiondetect.transforms import YoloAugmentations
from sting.regiondetect.networks import YOLOv3
import torch
from torch.utils.data import Dataset, DataLoader
from sting.utils.hardware import get_device_str
from sting.utils.types import RecursiveNamespace
from art import tprint
from torch.utils.data import random_split, ConcatDataset
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Region detection Training Arguments")

    parser.add_argument('-p', '--param_file',
                        help='Specify the param file for training region detection network',
                        required=True)
    parser.add_argument('-d', '--device', default=None,
                        help='Specify the device string (cpu, cuda, cuda:0, or cuda:1',
                        type=str, required=False)

    parser.add_argument('-w', '--num_workers_override', default=6,
                        help='Override number of workers for pytorch dataloader.',
                        type=int, required=False)
    
    parser.add_argument('-l', '--log_dir', default='barcode_logs',
                        help='Set directory name where you want the logs to go in',
                        required=False, type=str)
    
    parser.add_argument('-c', '--log_comment', default=None,
                        help='Added a log comment to the run',
                        type=str, required=False)
    
    args = parser.parse_args()
    
    return args

    
def train_model(param_file: str, device_overwrite: str = None,
                num_workers_overwrite: int = None,
                log_dir: str = 'barcode_logs', log_comment: str = None
                ):
    
    """
    Train are region detection model based on the parameters given
    
    Args:
        param_file (str): parameter file path, includes everything need to
            train a yolo series models for drawing bounding boxes around
            the regions containing barcdes
        device_overwrite (str): overwrite cuda device specified in the param_file 
        num_workers_overwrite (int): overwrite num of workers in pytorch dataloaders
        log_dir (str): directory for logs used by tensorboard
        log_comment (str): comment to the experiment

    """

    # Loading parameters for network training from the parameter file
    param_file = Path(param_file)
    if not param_file.exists():
        raise FileNotFoundError(f"Parameter file {str(param_file)} doesn't exist")
    
    # make a param namespace and fill in defaults
    param = load_params(param_file, ref_type='barcode')
    print(param)

    # Load ckpt if the training stops and you need to restart
    if param.Checkpoints.load_ckpt == False:
        expt_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        from_ckpt = False
    else:
        from_ckpt = True
        expt_id = Path(param.Checkpoints.ckpt_filename).parent.name

    if log_comment:
        expt_id = expt_id + '-' + log_comment
    
    if not from_ckpt:
        # create a new expt dir
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

    # save parameter file used in the directory of the expt
    param_in_save = expt_dir / Path('training_run_set').with_suffix(param_file.suffix)
    shutil.copy(param_file, param_in_save)

    # training run params after filling the defualts
    param_used_save = expt_dir / Path('training_run_used').with_suffix(param_file.suffix)
    save_params(param_used_save, param)

    # set hardware device to train
    if device_overwrite is not None:
        param.Hardware.device = device_overwrite
        device = device_overwrite
    else:
        device = param.Hardware.device
    
    # # check for cuda
    if torch.cuda.is_available():
        _, device_idx = get_device_str(device)
        if device_idx is not None:
            torch.cuda.set_device(device) 
    else:
        device = 'cpu'
    
    torch.set_num_threads(param.Hardware.torch_threads)

    # Log system
    # Use tensorboard on this directory to view logs of all the experimental runs
    log_dir_path = Path(param.Save.directory).parent / log_dir
    if not log_dir_path.exists():
        log_dir_path.mkdir(exist_ok=False)

    # setup tensorboard logger, what to write to tensorboard add things to logger as you train
    logger = SummaryWriter(log_dir=log_dir_path)
    
    # datasets
    if param.Datasets.transformations.type == 'YoloAugmentations':
        dataset_transformations = YoloAugmentations()
    dataset = BarcodeDataset(data_dir=param.Datasets.directory, transform=dataset_transformations) 
    len_dataset = len(dataset)
    train_len = int(len_dataset * param.Datasets.train.percentage)
    val_len = int(len_dataset * param.Datasets.validation.percentage)
    test_len = len_dataset - train_len - val_len
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len]) 
    #print(f"Length of train dataset: {len(train_ds)} , validation: {len(val_ds)}, test: {len(test_ds)}")

    # setup dataloaders
    train_dl, val_dl, test_dl = setup_dataloader(param, train_ds, val_ds, test_ds)

    #train_batch = next(iter(train_dl))
    #val_batch = next(iter(val_dl))
    #test_batch = next(iter(test_dl))
    #print(train_batch[0], train_batch[1].shape, train_batch[2].shape)
    #print(val_batch[0], val_batch[1].shape, val_batch[2].shape)
    #print(test_batch[0], test_batch[1].shape, test_batch[2].shape)

    # setup trainer

    model, optimizer, criterion, lr_scheduler, anchors, strides = setup_trainer(param, 
                    logger, model_out, ckpt_path, device) 

    # train loop



def setup_trainer(param, logger, model_out, ckpt_path, device):
    """
    Sets up all the things needed for training of a network and returns them

    Args:
        param (RecursiveNamespace): contains all the parameters needed
        logger (torch.utils.tensorboard.SummaryWriter): a custom tensorboard logger
        model_out (pathlib.Path): model save path
        ckpt_path (pathlib.Path): ckpt save path
        device (torch.device): device onto which the network is loaded

    Returns:
        model: an instance of the model loaded on the appropriate device
        optimizer: an instance of optimizer based on Hyperparameters
        criterion: a loss function used to calculate loss
        lr_scheduler: learning rate scheduler based on hyperparameters provided

    """
    models_available = {
        'YOLOv3': YOLOv3
    }

    model = models_available[param.HyperParameters.architecture]
    anchors = param.HyperParameters.model_params.anchors.sizes
    strides = param.HyperParameters.model_params.anchors.strides
    print(f"Anchors are .... {anchors} .. strides: {strides}")

    model = model.parse()

    return None, None, None, None, None, None

def setup_dataloader(param, train_ds, val_ds=None, test_ds=None):
    """
    Setup dataloaders for the datasets
    Args:
        param (RecursiveNamespace): 
        train_ds (torch.utils.Dataset): an instance of Dataset class
        val_ds (optional): an instance of Dataset class
        test_ds (optional): an instance of Dataset class
    """

    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=param.HyperParameters.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=param.Hardware.num_workers,
        pin_memory=True,
        collate_fn=train_ds.dataset.collate_fn
    )

    if val_ds is not None:
        val_dl = DataLoader(
            dataset=val_ds,
            batch_size=param.HyperParameters.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=param.Hardware.num_workers,
            pin_memory=False,
            collate_fn=val_ds.dataset.collate_fn
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
            pin_memory=False,
            collate_fn=test_ds.dataset.collate_fn
        )
    else:
        test_dl = None
    

    return train_dl, val_dl, test_dl


def main():
    print("Hello from region detection training")
    tprint("BARCODE")
    args = parse_args()

    train_model(args.param_file, 
                device_overwrite=args.device,
                num_workers_overwrite=args.num_workers_override,
                log_dir=args.log_dir,
                log_comment=args.log_comment,
                )

if __name__ == "__main__":
    main()