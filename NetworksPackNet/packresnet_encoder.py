import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *




class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.norm = nn.GroupNorm(16, out_channels, 1e-10)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        z = (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = cam_points[:, :2, :] / z

        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords, z.view(self.batch_size, 1, self.height, self.width)



class SpaceToDepth(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C *(self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x
    
    
    
    
class ReflectionPad3d(nn.Module):

    def __init__(self, padding):
        super(ReflectionPad3d, self).__init__()
        self.padding = torch.nn.modules.utils._ntuple(6)(padding)

    def forward(self, x):
        return F.pad(x,self.padding, mode='reflect')
    
    



class PackBlock(nn.Module):
    def __init__(self, in_channels, out_channels,use_conv3d=True,use_refl=False, need_last_nolin=True):
        super(PackBlock, self).__init__()
        self.useconv3d = use_conv3d
        self.need_last_nolin = need_last_nolin

        self.downscaled_channel = int(in_channels // 2)
        self.conv2d_1 = nn.Conv2d(in_channels * 4, self.downscaled_channel, 1, bias=False)
        self.norm2d_1 = nn.GroupNorm(16, self.downscaled_channel, 1e-10)
        self.nolin2d_1 = nn.ELU(inplace=True)

        if use_conv3d:
            self.conv3d_channel = 4
            self.conv3d = nn.Conv3d(1, self.conv3d_channel, 3, bias=True)
            self.nolin3d = nn.ELU(inplace=True)

            self.conv2d = nn.Conv2d(np.int(self.downscaled_channel * self.conv3d_channel), out_channels, 3, bias=False)
        else:
            self.conv2d = nn.Conv2d(self.downscaled_channel, out_channels, 3, bias=False)

        if use_refl:
            self.pad2d = nn.ReflectionPad2d(1)
            self.pad3d = nn.ReplicationPad3d(1)
        else:
            self.pad2d = nn.ZeroPad2d(1)
            self.pad3d = nn.ConstantPad3d(1, 0)


        self.norm2d = nn.GroupNorm(16, out_channels, 1e-10)

        self.pool3d = SpaceToDepth(2)
        self.nolin = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.pool3d(x)

        # we do down channel first
        x = self.conv2d_1(x)
        x = self.norm2d_1(x)
        x = self.nolin2d_1(x)

        N, C, H, W = x.size()
        x = x.view(N, 1, C, H, W)
        if self.useconv3d:
            x = self.pad3d(x)
            x = self.conv3d(x) # now is [N, 8, C, H, W]
            x = self.nolin3d(x)
            C *= self.conv3d_channel

        x = x.view(N, C, H, W)
        x = self.pad2d(x)
        x = self.conv2d(x)
        x = self.norm2d(x)
        if self.need_last_nolin:
            x = self.nolin(x)
        return x



def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels,stride=stride,  kernel_size=3, padding=1, bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, need_short=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, in_channels)
        self.norm2d1 = nn.GroupNorm(16, in_channels, 1e-10)
        self.nolin = nn.ELU(inplace=True)

        self.conv2 = conv3x3(in_channels, out_channels)
        self.norm2d2 = nn.GroupNorm(16, out_channels, 1e-10)

        self.need_short = need_short
        if self.need_short:
            self.conv3 = conv1x1(in_channels, out_channels)
            self.norm2d3 = nn.GroupNorm(16, out_channels, 1e-10)
        self.dropout = nn.Dropout2d(p=0.3)



    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm2d1(out)
        out = self.nolin(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2d2(out)

        if self.need_short:
            residual = self.conv3(residual)
            residual = self.norm2d3(residual)

        out += residual
        out = self.nolin(out)

        return out

def make_layer(in_channels, block, out_channels, blocks, downscale=4):

    layers = []
    layers.append(block(in_channels, out_channels, need_short=True))
    in_channels = out_channels * block.expansion
    for i in range(1, blocks):
        layers.append(block(in_channels, out_channels))

    layers.append(downsampleX(out_channels, out_channels))

    return nn.Sequential(*layers)


class downsampleX(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(downsampleX, self).__init__()
        self.pack = PackBlock(in_channel, out_channel, need_last_nolin=True)

    def forward(self, x):
        return self.pack(x)

class PackResNetEncoder(nn.Module):

    def __init__(self, num_input_images=1):
        super(PackResNetEncoder, self).__init__()

        conv_planes = [64, 64, 64, 128, 128]
        self.num_ch_enc = conv_planes
        self.downsample1 = downsampleX(conv_planes[0], conv_planes[1])

        self.conv1 = nn.Conv2d(num_input_images * 3, conv_planes[0], kernel_size=5, stride=1, padding=2, bias=False)
        self.norm2d1 = nn.GroupNorm(16, conv_planes[0], 1e-10)
        self.nolin1 = nn.ELU(inplace=True)

        self.conv2 = nn.Conv2d(conv_planes[0], conv_planes[0], kernel_size=7, stride=1, padding=3, bias=False)
        self.norm2d2 = nn.GroupNorm(16, conv_planes[0], 1e-10)
        self.nolin2 = nn.ELU(inplace=True)

        residuals = [2,2,2,2]
        block = BasicBlock

        self.layers1 = make_layer(conv_planes[0], block, conv_planes[0+1], blocks=residuals[0])
        self.layers2 = make_layer(conv_planes[1], block, conv_planes[1 + 1], blocks=residuals[1])
        self.layers3 = make_layer(conv_planes[2], block, conv_planes[2 + 1], blocks=residuals[2])
        self.layers4 = make_layer(conv_planes[3], block, conv_planes[3 + 1], blocks=residuals[3])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        self.features.append(x) #0
        x = self.conv1(x)
        x = self.norm2d1(x)
        x = self.nolin1(x) #1

        x = self.conv2(x)
        x = self.norm2d2(x)
        x = self.nolin2(x) #2
        self.features.append(self.downsample1(x)) #3
        self.features.append(self.layers1(self.features[-1])) #5
        self.features.append(self.layers2(self.features[-1])) #7
        self.features.append(self.layers3(self.features[-1])) #9
        self.features.append(self.layers4(self.features[-1])) #11

        return self.features