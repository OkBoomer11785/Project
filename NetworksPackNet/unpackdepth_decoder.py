from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *



class UnPackBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv3d=True, use_refl=True, need_last_nolin=True):
        super(UnPackBlock, self).__init__()
        self.useconv3d = use_conv3d
        self.need_last_nolin = need_last_nolin

        self.nolin = nn.ELU(inplace=True)

        if use_conv3d:
            self.conv3d_channels = 4
            self.conv3d = nn.Conv3d(1, self.conv3d_channels, 3, bias=True)
            self.nolin3d = nn.ELU(inplace=True)
            self.conv2d = nn.Conv2d(in_channels, np.int(4 * out_channels / self.conv3d_channels), 3, bias=True)
            self.norm2d = nn.GroupNorm(16, np.int(4 * out_channels / self.conv3d_channels), 1e-10)
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels * 4, 3, bias=True)
            self.norm2d = nn.GroupNorm(16, out_channels * 4, 1e-10)

        if use_refl:
            self.pad2d = nn.ReflectionPad2d(1)
            self.pad3d = nn.ReplicationPad3d(1)
        else:
            self.pad2d = nn.ZeroPad2d(1)
            self.pad3d = nn.ConstantPad3d(1, 0)

        self.upsample3d = DepthToSpace(2)

    def forward(self, x):
        x = self.pad2d(x)
        x = self.conv2d(x)
        x = self.norm2d(x)
        x = self.nolin(x)

        # B * 4Co/D * H * W

        N, C, H, W = x.size()

        if self.useconv3d:
            x = x.view(N, 1, C, H, W)
            x = self.pad3d(x)
            x = self.conv3d(x)
            x = self.nolin3d(x)
            x = x.view(N, C * self.conv3d_channels, H, W)

        x = self.upsample3d(x)
        return x
    

class DepthToSpace(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x


class UnPackDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(UnPackDepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128,128])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i+1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("unpack", i)] = UnPackBlock(num_ch_in, num_ch_out)

            # upconv_1
            if i > 0:
                num_ch_in = num_ch_out + self.num_ch_enc[i-1]
            else:
                num_ch_in = num_ch_out + 3

            if i in self.scales:
                self.convs[("dispconv", i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)
            if i < 3:
                num_ch_in += 1
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, idx):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("unpack", i)](x) #12 #14 #17 #20 #23
            if i >= 0:
                x = [input_features[i], x]
            else:
                x = [x]
            if i < 3:
                x += [upsample(self.outputs[("disp", idx, i+1)])]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 0)](x) #13 #15 #18 #21 #24
            if i in self.scales:
                self.outputs[("disp", idx, i)] = self.sigmoid(self.convs[("dispconv", i)](x)) #16 #19 #21 #24

        return self.outputs