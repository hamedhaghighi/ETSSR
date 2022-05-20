import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from models.BaseModel import BaseModel


class RDN(BaseModel):
    def __init__(self, upscale_factor):
        super(RDN, self).__init__()
        self.upscale_factor = upscale_factor
        self.init_feature = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.reconstruct = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            RDG(G0=64, C=8, G=64, n_RDB=16),
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
        x_left = self.one_image_output(x_left)
        x_right = self.one_image_output(x_right)
        return x_left, x_right


class one_conv(nn.Module):
    def __init__(self, G0, G):
        super(one_conv, self).__init__()
        self.conv = nn.Conv2d(G0, G, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        output = self.relu(self.conv(x))
        return torch.cat((x, output), dim=1)


class RDB(nn.Module):
    def __init__(self, G0, C, G):
        super(RDB, self).__init__()
        convs = []
        for i in range(C):
            convs.append(one_conv(G0+i*G, G))
        self.conv = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0+C*G, G0, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        out = self.conv(x)
        lff = self.LFF(out)
        return lff + x


class RDG(nn.Module):
    def __init__(self, G0, C, G, n_RDB):
        super(RDG, self).__init__()
        self.n_RDB = n_RDB
        RDBs = []
        for i in range(n_RDB):
            RDBs.append(RDB(G0, C, G))
        self.RDB = nn.Sequential(*RDBs)
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


if __name__ == "__main__":
    net = RDN(upscale_factor=2).cuda()
    from thop import profile
    input = torch.randn(1, 3, 128, 128).cuda()
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (total / 1e6))
    print('   Number of parameters: %.2fM' % (params/1e6))
    print('   Number of FLOPs: %.2fG' % (flops*2/1e9))
