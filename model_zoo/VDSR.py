import torch
import torch.nn as nn
from math import sqrt
from models.BaseModel import BaseModel
import torch.nn.functional as F

def conv_flop(N, in_ch, out_ch, K, bias=False):
    if bias:
        return N * (K**2 * in_ch + 1) * out_ch
    return N * K**2 * in_ch * out_ch


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))
    

class VDSR(BaseModel):
    def __init__(self, upscale_factor):
        super(VDSR, self).__init__()
        self.upscale_factor = upscale_factor
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    
    def one_image_output(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residual)
        return out

    def forward(self, x_left, x_right):
        x_left_upscale = F.interpolate(x_left[:, :3], scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        x_right_upscale = F.interpolate(x_right[:, :3], scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        x_left = self.one_image_output(x_left_upscale)
        x_right = self.one_image_output(x_right_upscale)
        return x_left, x_right

    
    def flop(self, H, W):
        N = H * W
        flop = 0
        # input
        flop += conv_flop(N, 3, 64, 3, False)
        # residual
        flop += 18 * conv_flop(N, 64, 64, 3, False)
        # output
        flop += conv_flop(N, 64, 3, 3, False)
        return flop


