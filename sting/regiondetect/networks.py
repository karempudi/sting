from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class ConvResBlock(nn.Module):
    """

    The repated blocks of the YOLOv3 paper, in the darknet layers

    Arguments:
        n_repeats (int): number of times the block is repeated
        in_channels (int): no of channels as input to the repated block
        layer_number (int): a starting number to number all the conv layers
                in the network

    """    
    def __init__(self, n_repeats, in_channels, layer_number):
        super(ConvResBlock, self).__init__()
        number = layer_number
        self.conv = nn.Sequential(OrderedDict([
                ('conv_' + str(number), 
                nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels, 
                          kernel_size=3, stride=2, padding=1)),
                ('batch_norm_' + str(number), 
                nn.BatchNorm2d(in_channels, momentum=0.1, eps=1e-5)),
                ('leaky_' + str(number), nn.LeakyReLU(0.1))
                    ]))
        number += 1
        self.repeat_list = nn.ModuleList()
        for i in range(n_repeats):
            
            repeat_block = nn.Sequential(OrderedDict([
                ('conv_' + str(number), 
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2,
                         kernel_size=1, stride=1, padding=0)),
                ('batch_norm_' + str(number),
                nn.BatchNorm2d(in_channels//2, momentum=0.1, eps=1e-5)),
                ('leaky_' + str(number), nn.LeakyReLU(0.1)),
                
                ('conv_' + str(number+1), 
                nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels,
                         kernel_size=3, stride=1, padding=1)),
                ('batch_norm_' + str(number+1), 
                nn.BatchNorm2d(in_channels, momentum=0.1, eps=1e-5)),
                ('leaky_' + str(number+1), nn.LeakyReLU(0.1))
            ]))
            self.repeat_list.append(repeat_block)
            number += 2
    def forward(self, x):
        # initial skip that will be added to the block at the end 
        # of each repeat in the repeat list
        x = self.conv(x)
        x_skip = x
        for i, module in enumerate(self.repeat_list):
            x = module(x)
            x = x + x_skip
            x_skip = x
        return x


class Darknet53(nn.Module):

    def __init__(self):
        super(Darknet53, self).__init__()
        number = 1
        self.conv_1 = nn.Sequential(OrderedDict([
                ('conv_' + str(number), 
                nn.Conv2d(in_channels=1, out_channels=32, 
                          kernel_size=3, stride=1, padding=1)),
                ('batch_norm_' + str(number), 
                nn.BatchNorm2d(32, momentum=0.1, eps=1e-5)),
                ('leaky_' + str(number), nn.LeakyReLU(0.1))
                    ]))
        
        # repeated blocks of the  5 times with differet set of repeats
        self.block_1 = ConvResBlock(n_repeats=1, in_channels=64, layer_number=2)
        self.block_2 = ConvResBlock(n_repeats=2, in_channels=128, layer_number=5)
        self.block_3 = ConvResBlock(n_repeats=8, in_channels=256, layer_number=10)
        self.block_4 = ConvResBlock(n_repeats=8, in_channels=512, layer_number=27)
        self.block_5 = ConvResBlock(n_repeats=4, in_channels=1024, layer_number=44)
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        return x