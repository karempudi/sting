import numpy as np
import torch
import torch.optim as optim


# Some code in the file is from UVA dlc course some is from 
# https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/pytorchyolo/test.py


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

def make_grid(nx, ny):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

def outputs_to_bboxes(outputs, anchors, strides):
    yolo_bboxes = []
    for i, (out, anch, stri) in enumerate(zip(outputs, anchors, strides)):
        #print(out.shape, anch.shape, stri.shape)
        # scale the outputs to bboxes correctly
        bs, _, ny, nx, no = out.shape
        grid = make_grid(nx, ny).to(out.device)
        anch = anch.view(1, -1, 1, 1, 2).to(out.device)
        out[..., 0:2] = (out[..., 0:2].sigmoid() + grid) * stri.to(out.device)
        out[..., 2:4] = torch.exp(out[..., 2:4]) * anch
        out[..., 4:] = out[..., 4:].sigmoid()
        out = out.view(bs, -1, no)
        yolo_bboxes.append(out)
    return torch.cat(yolo_bboxes, 1)

#############################################################
################# Functions for Metrics #####################
#############################################################

def evaluate(model, dataloader, class_names, img_size, iou_thres, 
             conf_thres, nms_thres, verbose, device):
    """
    Evaluate model on a dataset

    Args:
        model: a torch.nn model loaded with parameters
        dataloader: a torch.utils.DataLoader that can provide images and targets
        class_names: a list of class names 
        img_size (tuple): image size used by yolo model
        iou_thres (float): IoU threshold required to qualify as detected
        conf_thres (float): Object confidence threshold
        nms_thres (float): IoU threshold for non-maximum suppression
        verbose (bool): print more stuff or not
        device (str): device to run the model and inference on
    
    Returns:
        precision, recall, AP, f1, ap_class
    """
    # set model to eval mode
    model.eval()

    #
    pass