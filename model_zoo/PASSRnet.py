from turtle import up
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from models.BaseModel import BaseModel


def conv_flop(N, in_ch, out_ch, K, bias=False):
    if bias:
        return N * (K**2 * in_ch + 1) * out_ch
    return N * K**2 * in_ch * out_ch

def ResB_flops(N, channel):
    return 2 * conv_flop(N, channel, channel, 3)


def ResASPPB_flops(N, channel):
    return 9 * conv_flop(N, channel, channel, 3) + 3 * conv_flop(N, channel * 3, channel, 3)

class PASSRnet(BaseModel):
    def __init__(self, upscale_factor):
        super(PASSRnet, self).__init__()
        ### feature extraction
        self.upscale_factor = upscale_factor
        self.init_feature = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            ResB(64),
            ResASPPB(64),
            ResB(64),
            ResASPPB(64),
            ResB(64),
        )
        ### paralax attention
        self.pam = PAM(64)
        ### upscaling
        self.upscale = nn.Sequential(
            ResB(64),
            ResB(64),
            ResB(64),
            ResB(64),
            nn.Conv2d(64, 64 * upscale_factor ** 2, 1, 1, 0, bias=False),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(64, 3, 3, 1, 1, bias=False),
            nn.Conv2d(3, 3, 3, 1, 1, bias=False)
        )

    def one_image_output(self, x_left, x_right):
         ### feature extraction
        buffer_left = self.init_feature(x_left)
        buffer_right = self.init_feature(x_right)
        ### parallax attention
        buffer = self.pam(buffer_left, buffer_right)
        ### upscaling
        out = self.upscale(buffer)
        return out

    def forward(self, x_left, x_right):
        out_left = self.one_image_output(x_left[:, :3], x_right[:, :3])
        out_right = self.one_image_output(x_right[:, :3], x_left[:, :3])
        return out_left, out_right

    def flop(self, H, W):
        N = H * W
        flops = 0
        flops += 2 * conv_flop(N, 3, 64, 3)
        # ResB
        flops += 2 * 3 * (ResB_flops(N, 64))
        # ResASPPB
        flops += 2 * 2 * (ResASPPB_flops(N, 64))
        # pam
        flops += self.pam.flop(H, W)
        # upscale
        flops += 4 * (ResB_flops(N, 64))
        flops += conv_flop(N, 64, 64 * self.upscale_factor ** 2, 1)
        flops += conv_flop(N * self.upscale_factor ** 2, 64, 3, 3)
        flops += conv_flop(N * self.upscale_factor ** 2, 3, 3, 3)
        
        return flops




class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )

    def __call__(self, x):
        out = self.body(x)
        return out + x


class ResASPPB(nn.Module):
    def __init__(self, channels):
        super(ResASPPB, self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(
            channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_1 = nn.Sequential(nn.Conv2d(
            channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_1 = nn.Sequential(nn.Conv2d(
            channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(
            channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_2 = nn.Sequential(nn.Conv2d(
            channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_2 = nn.Sequential(nn.Conv2d(
            channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv1_3 = nn.Sequential(nn.Conv2d(
            channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_3 = nn.Sequential(nn.Conv2d(
            channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_3 = nn.Sequential(nn.Conv2d(
            channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.b_1 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)
        self.b_2 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)
        self.b_3 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)

    def __call__(self, x):
        buffer_1 = []
        buffer_1.append(self.conv1_1(x))
        buffer_1.append(self.conv2_1(x))
        buffer_1.append(self.conv3_1(x))
        buffer_1 = self.b_1(torch.cat(buffer_1, 1))

        buffer_2 = []
        buffer_2.append(self.conv1_2(buffer_1))
        buffer_2.append(self.conv2_2(buffer_1))
        buffer_2.append(self.conv3_2(buffer_1))
        buffer_2 = self.b_2(torch.cat(buffer_2, 1))

        buffer_3 = []
        buffer_3.append(self.conv1_3(buffer_2))
        buffer_3.append(self.conv2_3(buffer_2))
        buffer_3.append(self.conv3_3(buffer_2))
        buffer_3 = self.b_3(torch.cat(buffer_3, 1))

        return x + buffer_1 + buffer_2 + buffer_3


class PAM(nn.Module):
    def __init__(self, channels):
        super(PAM, self).__init__()
        self.channels = channels
        self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b3 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(-1)
        self.rb = ResB(64)
        self.fusion = nn.Conv2d(channels * 2 + 1, channels, 1, 1, 0, bias=True)

    def forward(self, x_left, x_right):
        b, c, h, w = x_left.shape
        buffer_left = self.rb(x_left)
        buffer_right = self.rb(x_right)

        ### M_{right_to_left}
        # B * H * W * C
        Q = self.b1(buffer_left).permute(0, 2, 3, 1)
        # B * H * C * W
        S = self.b2(buffer_right).permute(0, 2, 1, 3)
        score = torch.bmm(Q.contiguous().view(-1, w, c),
                          S.contiguous().view(-1, c, w))                                            # (B*H) * W * W
        M_right_to_left = self.softmax(score)

        ### M_{left_to_right}
        # B * H * W * C
        Q = self.b1(buffer_right).permute(0, 2, 3, 1)
        # B * H * C * W
        S = self.b2(buffer_left).permute(0, 2, 1, 3)
        score = torch.bmm(Q.contiguous().view(-1, w, c),
                          S.contiguous().view(-1, c, w))                                            # (B*H) * W * W
        M_left_to_right = self.softmax(score)

        ### valid masks
        V_left_to_right = torch.sum(M_left_to_right.detach(), 1) > 0.1
        V_left_to_right = V_left_to_right.view(b, 1, h, w)  # B * 1 * H * W
        V_left_to_right = morphologic_process(V_left_to_right)

        ### fusion
        buffer = self.b3(x_right).permute(0, 2, 3, 1).contiguous(
        ).view(-1, w, c)                      # (B*H) * W * C
        buffer = torch.bmm(M_right_to_left, buffer).contiguous().view(
            b, h, w, c).permute(0, 3, 1, 2)  # B * C * H * W
        out = self.fusion(torch.cat((buffer, x_left, V_left_to_right), 1))

        return out

    def flop(self, H, W):
        N = H * W
        flop = 0
        flop += 2 * ResB_flops(N, 64)
        flop += 5 * conv_flop(N, self.channels, self.channels, 1, True)
        flop += 2 * H * (W**2) * self.channels
        # flop += 2 * H * W
        flop += H * (W**2) * self.channels
        flop += conv_flop(N, self.channels * 2 + 1, self.channels, 1, True)
        return flop


def morphologic_process(mask):
    device = mask.device
    b, _, _, _ = mask.shape
    mask = ~mask
    mask_np = mask.cpu().numpy().astype(bool)
    mask_np = morphology.remove_small_objects(mask_np, 20, 2)
    mask_np = morphology.remove_small_holes(mask_np, 10, 2)
    for idx in range(b):
        buffer = np.pad(mask_np[idx, 0, :, :], ((3, 3), (3, 3)), 'constant')
        buffer = morphology.binary_closing(buffer, morphology.disk(3))
        mask_np[idx, 0, :, :] = buffer[3:-3, 3:-3]
    mask_np = 1-mask_np
    mask_np = mask_np.astype(float)

    return torch.from_numpy(mask_np).float().to(device)
