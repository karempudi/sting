import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from types import SimpleNamespace
from typing import List

class ConvBlock(nn.Module):

    def __init__(self, c_in: int, c_out: int, 
            kernerl_size:int =3, padding: int = 1, stride: int = 1,
            bias: bool = False):
        """
        A conv block of a U-net module

        Takes an input tensor (N, c_in, H, W) --> (N, c_out, H, W)

        Args:
            c_in  (int) : Number of channels in the input 
            c_out (int) : Number of channels in the output
            kernel_size (int) : kernel_size of the conv block
                defualt is 3
            padding (int): padding used in conv layers
                defaults is 1
            stride (int): stride of the conv block
                defaults is 1
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=kernerl_size, 
                    padding=padding, stride=stride, bias=bias),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_in, c_out, kernel_size=kernerl_size, 
                    padding=padding, stride=stride, bias=bias),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

class UpsampleBlock(nn.Module):

    def __init__(self, c_in: int, c_out: int, 
        upsample_type: str ='transpose_conv', feature_fusion_type: str='concat'):
        """
        An upsample block in the U-net architecture

        Args:
            c_in (int): Number of channels in the input
            c_out (int): Number of channels in the output
            upsample_type (str): upsample method, 'transpose_conv' (defualt) or 'upsample'
            feature_fusion_type (str) : 'concat' or 'add', method by which the features
                from the downsample part are added to the upsample features 
        """
        super().__init__()
        self.upsample_type = upsample_type
        self.feature_fusion_type = feature_fusion_type

        if self.upsample_type == 'transpose_conv':
            self.upsample_block = nn.ConvTranspose2d(c_in, c_in, kernel_size=2, stride=2)
        elif self.upsample_type == 'upsample':
            self.upsample_block = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if self.feature_fusion_type == 'concat':
            c_in = c_in + c_out
            self.conv_block = ConvBlock(c_in, c_out)
        elif self.feature_fusion_type == 'add':
            self.shrink_block = ConvBlock(c_in, c_out) # so that the channels match
            self.conv_block = ConvBlock(c_out, c_out)

    def forward(self, x, features=None):
        if features is not None:
            # then you have something coming from the side
            x = self.upsample_block(x)
            if self.feature_fusion_type == 'concat':
                x = torch.cat([features, x], dim=1) # concat along the channel 'C' dimension
            elif self.feature_fusion_type == 'add':
                x = self.shrink_block(x) + features
            x = self.conv_block(x)
            return x
        else:
            x = self.upsample_block(x)
            x = self.conv_block(x)
            return x

class Unet(nn.Module):

    def __init__(self, channels_by_scale: List, num_classes: int = 1,
            upsample_type='transpose_conv', feature_fusion_type='concat'):
        super().__init__()
        

    @classmethod
    def param(cls, param):
        pass

    def forward(self, x):
        pass

class ResUnet(nn.Module):
    
    def __init__(self):
        pass

    @classmethod
    def param(cls, param):
        pass
    
    def forward(self, x):
        pass

model_dict = {
    'Unet': Unet,
    'ResUnet': ResUnet
}