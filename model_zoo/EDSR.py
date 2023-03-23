import torch.nn as nn

from models.BaseModel import BaseModel


def conv_flop(N, in_ch, out_ch, K, bias=False):
    if bias:
        return N * (K**2 * in_ch + 1) * out_ch
    return N * K**2 * in_ch * out_ch


class EDSR(BaseModel):
    def __init__(self, upscale_factor):
        super(EDSR, self).__init__()
        self.init_feature = nn.Conv2d(3, 256, 3, 1, 1)
        self.upscale_factor = upscale_factor
        self.body = ResidualGroup(256, 32)
        if upscale_factor == 2:
            self.upscale = nn.Sequential(
                nn.Conv2d(256, 256 * 4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.Conv2d(256, 3, 3, 1, 1))
        if upscale_factor == 4:
            self.upscale = nn.Sequential(
                nn.Conv2d(256, 256 * 4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.Conv2d(256, 256 * 4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.Conv2d(256, 3, 3, 1, 1))

    def one_image_output(self, x):
        buffer = self.init_feature(x)
        buffer = self.body(buffer)
        out = self.upscale(buffer)
        return out

    def forward(self, x_left, x_right):
        x_left = self.one_image_output(x_left[:, :3])
        x_right = self.one_image_output(x_right[:, :3])
        return x_left, x_right

    def flop(self, H, W):
        N = H * W
        flop = 0
        flop += conv_flop(N, 3, 256, 3)
        flop += self.body.flop(N)
        if self.upscale_factor == 2:
            flop += conv_flop(N, 256, 256 * 4, 1)
            flop += conv_flop(N * 4, 256, 3, 1)
        elif self.upscale_factor == 4:
            flop += conv_flop(N, 256, 256 * 4, 1)
            flop += conv_flop(N * 4, 256, 256 * 4, 1)
            flop += conv_flop(N * 16, 256, 3, 1)

        return 2 * flop


class ResB(nn.Module):
    def __init__(self, n_feat):
        super(ResB, self).__init__()
        self.n_feat = n_feat
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, 3, 1, 1))
            if i == 0:
                modules_body.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = 0.1 * self.body(x)
        res += x
        return res

    def flop(self, N):
        flop = 0
        flop += 2 * conv_flop(N, self.n_feat, self.n_feat, 3)
        return flop


class ResidualGroup(nn.Module):
    def __init__(self, n_feat, n_resblocks):
        super(ResidualGroup, self).__init__()
        self.n_feat = n_feat
        self.n_resblock = n_resblocks
        self.modules_body = [
            ResB(n_feat)
            for _ in range(n_resblocks)]
        self.modules_body.append(nn.Conv2d(n_feat, n_feat, 3, 1, 1))
        self.body = nn.Sequential(*self.modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

    def flop(self, N):
        flop = 0
        flop += conv_flop(N, 3, 256, 3)
        flop += self.n_resblock * self.modules_body[0].flop(N)
        flop += conv_flop(N, self.n_feat, self.n_feat, 3)
        return flop
