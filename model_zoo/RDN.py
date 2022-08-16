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

class RDN(BaseModel):
    def __init__(self, upscale_factor):
        super(RDN, self).__init__()
        self.upscale_factor = upscale_factor
        self.init_feature = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.RDG = RDG(G0=64, C=8, G=64, n_RDB=16)
        self.reconstruct = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            self.RDG,
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        self.upscale = nn.Sequential(
            nn.Conv2d(64, 64 * upscale_factor ** 2, 1, 1, 0, bias=False),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(64, 3, 3, 1, 1, bias=False))

    def one_image_output(self, x):

        buffer0 = self.init_feature(x)
        buffer = self.reconstruct(buffer0) + buffer0
        out = self.upscale(buffer)

        return out

    def forward(self, x_left, x_right):
        x_left = self.one_image_output(x_left[:, :3])
        x_right = self.one_image_output(x_right[:, :3])
        return x_left, x_right

    def flop(self, H, W):
        N = H * W
        flop = 0 
        flop += conv_flop(N, 3, 64, 3)
        flop += 2 * conv_flop(N, 64, 64, 3)
        flop += self.RDG.flop(N)
        flop += conv_flop(N, 64, 64 * self.upscale_factor**2, 1)
        flop += conv_flop(N * self.upscale_factor**2, 64, 3, 3)
        return 2 * flop


class one_conv(nn.Module):
    def __init__(self, G0, G):
        super(one_conv, self).__init__()
        self.G0, self.G = G0, G
        self.conv = nn.Conv2d(G0, G, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        output = self.relu(self.conv(x))
        return torch.cat((x, output), dim=1)

    def flop(self, N):
        flop = N * (self.G0 * 9 + 1) * self.G
        return flop

class RDB(nn.Module):
    def __init__(self, G0, C, G):
        super(RDB, self).__init__()
        self.G0, self.C , self.G = G0, C, G
        self.convs = []
        for i in range(C):
            self.convs.append(one_conv(G0+i*G, G))
        self.conv = nn.Sequential(*self.convs)
        self.LFF = nn.Conv2d(G0+C*G, G0, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        out = self.conv(x)
        lff = self.LFF(out)
        return lff + x

    def flop(self, N):
        flop = 0
        for cv in self.convs:
            flop += cv.flop(N)
        flop += N * (self.G0 + self.C*self.G + 1) * self.G0
        return flop

class RDG(nn.Module):
    def __init__(self, G0, C, G, n_RDB):
        super(RDG, self).__init__()
        self.G0, self.C, self.G = G0, C, G
        self.n_RDB = n_RDB
        self.RDBs = []
        for i in range(n_RDB):
            self.RDBs.append(RDB(G0, C, G))
        self.RDB = nn.Sequential(*self.RDBs)
        self.conv = nn.Conv2d(G0*n_RDB, G0, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        buffer = x
        temp = []
        for i in range(self.n_RDB):
            buffer = self.RDB[i](buffer)
            temp.append(buffer)
        out = torch.cat(temp, dim=1)
        out = self.conv(out)
        return out

    def flop(self, N):
        flop = 0
        flop += self.n_RDB * self.RDBs[0].flop(N)
        flop += N * (self.n_RDB * self.G0 + 1) * self.G0
        return flop


