from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn


class PoseCNN(nn.Module):
    def __init__(self, num_input_frames):
        super(PoseCNN, self).__init__()

        self.num_input_frames = num_input_frames

        self.convs = {}
        stride = 2
        padding = 1
        kernel = 3
        self.convs[0] = nn.Conv2d(3 * num_input_frames, 16, 7, stride, 3)
        self.convs[1] = nn.Conv2d(16, 32, 5, stride, 2)
        self.convs[2] = nn.Conv2d(32, 64, kernel, stride, padding)
        self.convs[3] = nn.Conv2d(64, 128, kernel, stride, padding)
        self.convs[4] = nn.Conv2d(128, 256, kernel, stride, padding)
        self.convs[5] = nn.Conv2d(256, 256, kernel, stride, padding)
        self.convs[6] = nn.Conv2d(256, 256, kernel, stride, padding)
        self.num_convs = len(self.convs)
        self.pose_conv = nn.Conv2d(256, 6 * (num_input_frames - 1), 1)
        self.relu = nn.ReLU(True)
        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, out):

        for i in range(self.num_convs):
            out = self.convs[i](out)
            out = self.relu(out)
        out = self.pose_conv(out)
        out = out.mean(3).mean(2)
        out = 0.01 * out.view(-1, self.num_input_frames - 1, 1, 6)
        return out[..., :3], out[..., 3:]