from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

class ResBlock(nn.Module):
    def __init__(self, input_channel, channel, kernel_size, stride_val):
        super(ResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channel, channel, kernel_size, stride_val,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size, stride=1, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(channel)
        
        self.conv3_sc = torch.nn.Conv2d(input_channel,channel,kernel_size=1,stride=stride_val,bias = False)
        self.bn3_sc = torch.nn.BatchNorm2d(channel)
        
        self.conv3 = nn.Conv2d(channel, channel, kernel_size, stride=1,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(channel)
        self.conv4 = nn.Conv2d(channel, channel, kernel_size, stride=1,padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(channel)
    
    def forward(self, x):
        
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        identity = self.bn3_sc(self.conv3_sc(identity))
        out += identity
        out = F.relu(out)
        
        identity1 = out
        out1 = F.relu(self.bn3(self.conv3(out)))
        out1 = self.bn4(self.conv3(out1))
        out1 += identity1
        out1 = F.relu(out1)
        return out     

class ResnetEncoder(nn.Module):
    def __init__(self):
        super(ResnetEncoder, self).__init__()
        
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = ResBlock(input_channel = 64, channel = 64, kernel_size = 3, stride_val = 1)
        self.layer2 = ResBlock(input_channel = 64, channel = 128, kernel_size = 3, stride_val = 2)
        self.layer3 = ResBlock(input_channel = 128, channel = 256, kernel_size = 3, stride_val = 2)
        self.layer4 = ResBlock(input_channel = 256, channel = 512, kernel_size = 3, stride_val = 2)

    
    
    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.conv1(x)
        x = self.bn1(x)
        self.features.append(self.relu(x))
        self.features.append(self.layer1(self.maxpool(self.features[-1])))
        self.features.append(self.layer2(self.features[-1]))
        self.features.append(self.layer3(self.features[-1]))
        self.features.append(self.layer4(self.features[-1]))

        return self.features