import torch
import torch.nn as nn
import torch.nn.functional as F

from sting.segmentation.networks import model_dict as seg_model_dict
from sting.regiondetect.networks import model_dict as barcode_model_dict
from sting.utils.hardware import get_device_str

class LiveNet(nn.Module):

    def __init__(self, param):
        super().__init__()
        device = param.Hardware.device
        if torch.cuda.is_available():
            _, device_idx = get_device_str(device)
            if device_idx is not None:
                torch.cuda.set_device(device)
        else:
            device = 'cpu'

        if 'Segmentation' in param.keys():
            # load the right type of segmentaiton network
            segment_params = param.Segmentation
            if segment_params.type == 'dual':
                # predicting 2 channels at the same time
                model = seg_model_dict[segment_params.architecture]
                self.segment_model = model.parse(channels_by_scale=segment_params.model_params.channels_by_scale,
                                                 num_outputs=segment_params.model_params.num_outputs,
                                                 upsample_type=segment_params.model_params.upsample_type,
                                                 feature_fusion_type=segment_params.model_params.feature_fusion_type).to(device=device)
        else:
            self.segment_model = None

        if 'Barcode' in param.keys():
            # load the right barcode network
            barcode_params = param.Barcode
            barcode_model = barcode_model_dict[barcode_params.architecture]
            anchor_sizes = barcode_params.model_params.anchors.sizes
            strides = barcode_params.model_params.anchors.strides
            num_classes = barcode_params.model_params.num_classes

            anchors_list = [[anchor_sizes[i], anchor_sizes[i+1], anchor_sizes[i+2]] for i in range(0, len(anchor_sizes), 3)]

            self.anchors_t = tuple(torch.tensor(anch).float().to(device=device) for anch in anchors_list)
            self.strides_t = tuple(torch.tensor(stride).to(device=device) for stride in strides)

            self.barcode_model = barcode_model.parse(anchors=anchors_list, num_classes=num_classes).to(device=device)
        else:
            self.barcode_model = None

        self.param = param

    def load_state_dict(self):
        # load the model parameters
        segment_model_path = self.param.Segmentation.model_paths.both
        barcode_model_path = self.param.Barcode.model_path
        self.segment_model.load_state_dict(torch.load(segment_model_path))
        self.barcode_model.load_state_dict(torch.load(barcode_model_path))

    def eval(self):
        self.segment_model.eval()
        self.barcode_model.eval()

    def forward(self, x):
        # x is a dict with keys 'segment' and 'barcode'
        return x
