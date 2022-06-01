from termios import N_MOUSE
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from models.BaseModel import BaseModel


def conv_flop(N, in_ch, out_ch, K, bias=False):
    if bias:
        return N * (K**2 * in_ch + 1) * out_ch
    return N * K**2 * in_ch * out_ch

class RCAN(BaseModel):
    def __init__(self, upscale_factor):
        super(RCAN, self).__init__()
        self.upscale_factor = upscale_factor
        self.init_feature = nn.Conv2d(3, 64, 3, 1, 1)
        self.RG1 = ResidualGroup(n_feat=64, n_resblocks=20)
        self.RG2 = ResidualGroup(n_feat=64, n_resblocks=20)
        self.RG3 = ResidualGroup(n_feat=64, n_resblocks=20)
        self.RG4 = ResidualGroup(n_feat=64, n_resblocks=20)
        self.RG5 = ResidualGroup(n_feat=64, n_resblocks=20)
        self.RG6 = ResidualGroup(n_feat=64, n_resblocks=20)
        self.RG7 = ResidualGroup(n_feat=64, n_resblocks=20)
        self.RG8 = ResidualGroup(n_feat=64, n_resblocks=20)
        self.RG9 = ResidualGroup(n_feat=64, n_resblocks=20)
        self.RG10 = ResidualGroup(n_feat=64, n_resblocks=20)
        self.last_feature = nn.Conv2d(64, 64, 3, 1, 1)
        self.upscale = nn.Sequential(
            nn.Conv2d(64, 64 * upscale_factor ** 2, 1, 1, 0),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(64, 3, 3, 1, 1))

    def one_image_output(self, x):
        buffer_00 = self.init_feature(x)
        buffer_01 = self.RG1(buffer_00)
        buffer_02 = self.RG2(buffer_01)
        buffer_03 = self.RG3(buffer_02)
        buffer_04 = self.RG4(buffer_03)
        buffer_05 = self.RG5(buffer_04)
        buffer_06 = self.RG6(buffer_05)
        buffer_07 = self.RG7(buffer_06)
        buffer_08 = self.RG8(buffer_07)
        buffer_09 = self.RG9(buffer_08)
        buffer_10 = self.RG10(buffer_09)
        buffer_11 = self.last_feature(buffer_10) + buffer_00
        out = self.upscale(buffer_11)

        return out

    def forward(self, x_left, x_right):
        x_left = self.one_image_output(x_left)
        x_right = self.one_image_output(x_right)
        return x_left, x_right
    
    def flop(self, H, W):
        N = H * W
        flop = 0
        flop += conv_flop(N, 3, 64, 3)
        flop += 10 * self.RG1.flop(N)
        flop += conv_flop(N, 64, 64 * self.upscale_factor**2, 1)
        flop += conv_flop(N * self.upscale_factor**2, 64, 3, 3)
        return flop


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, 4, 1, padding=0, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(4, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

    def flop(self, N):
        flop = 0
        flop += N * self.channel
        flop += (self.channel + 1) * 4
        flop += (4 + 1) * self.channel
        flop += N * self.channel
        return flop


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, n_feat):
        super(RCAB, self).__init__()
        self.n_feat = n_feat
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, 3, 1, 1))
            if i == 0: modules_body.append(nn.LeakyReLU(0.1, inplace=True))
        self.ca_layer = CALayer(n_feat)
        modules_body.append(self.ca_layer)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

    def flop(self, N):
        flop = 0
        flop += 2 * conv_flop(N, self.n_feat, self.n_feat, 3)
        flop += self.ca_layer.flop()
        return flop

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, n_feat, n_resblocks):
        super(ResidualGroup, self).__init__()
        self.n_feat = n_feat
        self.n_resblocks = n_resblocks
        self.modules_body = [
            RCAB(n_feat) \
            for _ in range(n_resblocks)]
        self.modules_body.append(nn.Conv2d(n_feat, n_feat, 3, 1, 1))
        self.body = nn.Sequential(*self.modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
    
    def flop(self, N):
        flop = 0
        flop += self.n_resblocks * self.modules_body[0].flop(N)
        flop += conv_flop(N, self.n_feat, self.n_feat, 3)
        return flop

