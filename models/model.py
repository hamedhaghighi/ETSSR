import os
from torch.utils.data import Subset
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/home/oem/Documents/PhD_proj/iPASSR')
sys.path.append('/home/haghig_h@WMGDS.WMG.WARWICK.AC.UK/Documents/StereoSR')
from utils import disparity_alignment
from models.CoSwinTransformer import CoSwinAttn
from models.SwinTransformer import SwinAttn
from models.BaseModel import BaseModel
from timm.models.layers import trunc_normal_
from dataset import toNdarray, toTensor
import dataset

class Net(BaseModel):
    def __init__(self, upscale_factor, img_size, model, input_channel=3, w_size=8, device='cpu'):
        super(Net, self).__init__()
        self.input_channel = 20 if 'seperate' in model else input_channel
        self.upscale_factor = upscale_factor
        self.model = model
        self.w_size = w_size
        self.init_feature = nn.Conv2d(self.input_channel, 64, 3, 1, 1, bias=True)
        if 'CP' in self.model:
            self.condition_feature = nn.Sequential(nn.Conv2d(4, 64, 3, 1, 1, bias=True), nn.LeakyReLU(0.1, inplace=True), nn.Conv2d(64, 64, 3, 1, 1, bias=True))
        self.n_RDB = 3 if 'MDB' in model else 3
        self.deep_feature = RDG(G0=64, C=4, G=24, n_RDB=self.n_RDB, type='P') if 'MDB' in model else RDG(G0=64, C=4, G=24, n_RDB=self.n_RDB, type='N')
        depths = [2]
        num_heads = [4]
        if 'pam' in model :
            if 'light' in model:
                self.swin = SwinAttn(img_size=img_size, window_size=10, depths=[1], embed_dim=64, num_heads=num_heads, mlp_ratio=2)
                self.pam = LightPAM(64, self.n_RDB, w_size)
            else:
                self.pam = PAM(64, self.n_RDB)
        elif any(x in model for x in ['coswin', 'late_fusion']):
            self.swin = SwinAttn(img_size=img_size, window_size=w_size, depths=[1], embed_dim=64, num_heads=num_heads, mlp_ratio=2)
            self.coswin = CoSwinAttn(img_size=img_size, window_size=w_size, depths=[1], embed_dim=64, num_heads=num_heads, mlp_ratio=2)
        elif any(x in model for x in ['seperate', 'independent']):
            self.swin = SwinAttn(img_size=img_size, window_size=w_size, depths=depths, embed_dim=64, num_heads=num_heads, mlp_ratio=2)
        self.f_RDB = RDB(G0=128, C=4, G=32)
        self.CAlayer = CALayer(128)
        self.fusion = nn.Sequential(self.f_RDB, self.CAlayer, nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=True))
        self.reconstruct = RDG(G0=64, C=4, G=24, n_RDB=self.n_RDB, type='P') if 'MDB' in model else RDG(G0=64, C=4, G=24, n_RDB=self.n_RDB, type='N')
        self.upscale = nn.Sequential(nn.Conv2d(64, 3, 3, 1, 1, bias=True), nn.Conv2d(3, 3 * upscale_factor ** 2, 1, 1, 0, bias=True), nn.PixelShuffle(upscale_factor))
        #self.apply(self._init_weights)

    def forward(self, x_left, x_right, is_training = 0):
        if not 'pam' in self.model:
            x_left, mod_pad_h, mod_pad_w = self.check_image_size(x_left)
            x_right, _, _ = self.check_image_size(x_right)
        b, c, h, w = x_left.shape
        if c > 3:
            d_left = x_left[:, 3]
            d_right = x_right[:, 3]

        x_left_upscale = F.interpolate(x_left[:, :3], scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        x_right_upscale = F.interpolate(x_right[:, :3], scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        if 'seperate' in self.model:
            coords_b, coords_h, r2l_w, l2r_w = disparity_alignment(d_left, d_right, b, h, w)
            x_left_selected , x_right_selected = x_left[coords_b, :, coords_h, l2r_w].permute(0, 3, 1, 2) ,x_right[coords_b, :, coords_h, r2l_w].permute(0, 3, 1, 2)
            x_left, x_right = torch.cat([x_left, x_left_selected], dim = 1), torch.cat([x_right, x_right_selected], dim = 1)
        
        CF_left = self.condition_feature(x_left[:, 3:]) if 'CP' in self.model else None
        CF_right = self.condition_feature(x_right[:, 3:]) if 'CP' in self.model else None

        buffer_left = self.init_feature(x_left[:, :self.input_channel])
        buffer_right = self.init_feature(x_right[:, :self.input_channel])
        buffer_left, catfea_left = self.deep_feature(buffer_left, CF_left)
        buffer_right, catfea_right = self.deep_feature(buffer_right, CF_right)
        if 'pam' in self.model:
            if 'light' in self.model:
                buffer_leftT, buffer_rightT = self.swin(buffer_left), self.swin(buffer_right)
                buffer_leftT, buffer_rightT = self.pam(buffer_leftT, buffer_rightT, catfea_left, catfea_right)
            else:
                buffer_leftT, buffer_rightT = self.pam(buffer_left, buffer_right, catfea_left, catfea_right)
        elif 'coswin' in self.model:
            buffer_leftT, buffer_rightT = self.swin(buffer_left), self.swin(buffer_right)
            buffer_leftT, buffer_rightT = self.coswin(buffer_leftT, buffer_rightT, d_left, d_right)
            # if 'coswin_wo_d' in self.model:
            #     buffer_leftT, buffer_rightT = self.coswin(buffer_leftT, buffer_rightT)
            # else:
            #     buffer_leftT, buffer_rightT = self.coswin(buffer_leftT, buffer_rightT, d_left, d_right)            
        elif any(x in self.model for x in ['seperate', 'independent']):
            buffer_leftT, buffer_rightT = self.swin(buffer_left), self.swin(buffer_right)
        if 'mine_late_fusion' in self.model:
            buffer_leftF, buffer_rightF = buffer_left, buffer_right
        else:
            buffer_leftF, buffer_rightF = self.fusion(torch.cat([buffer_left, buffer_leftT], dim=1)), self.fusion(torch.cat([buffer_right, buffer_rightT], dim=1))
        
        buffer_leftF, _ = self.reconstruct(buffer_leftF, CF_left)
        buffer_rightF, _ = self.reconstruct(buffer_rightF, CF_right)
        if 'late_fusion' in self.model:
            buffer_leftF, buffer_rightF = self.swin(buffer_leftF), self.swin(buffer_rightF)
            buffer_leftF, buffer_rightF = self.coswin(buffer_leftF, buffer_rightF, d_left, d_right)
        out_left = self.upscale(buffer_leftF) + x_left_upscale
        out_right = self.upscale(buffer_rightF) + x_right_upscale
        mod_h = -mod_pad_h * self.upscale_factor if ((not 'pam' in self.model) and mod_pad_h != 0) else None
        mod_w = -mod_pad_w * self.upscale_factor if ((not 'pam' in self.model) and mod_pad_w != 0) else None
        return out_left[..., :mod_h, :mod_w], out_right[..., :mod_h, :mod_w]


    def flop(self, H, W):
        N = H * W
        flop = 0
        flop += 2 * ((self.input_channel * 9 + 1) * N * 64) # adding 1 for bias
        flop += 2 * self.deep_feature.flop(N)
        if 'pam' in self.model:
            flop += self.pam.flop(H, W)
        elif 'coswin' in self.model:
            flop += self.swin.flops(H, W) + self.coswin.flops(H, W)
        elif any(x in self.model for x in ['seperate', 'independent']):
            flop += self.swin.flops(H, W)
        flop += 2 * (self.f_RDB.flop(N) + self.CAlayer.flop(N) + N * (128 + 1) * 64)
        flop += 2 * self.reconstruct.flop(N)
        flop += 2 * (N * (64 + 1) * 64 * (self.upscale_factor ** 2) + N * 3 * (64 * 9 + 1))
        flop += 2 * (N * (64 + 1) * 64 * (self.upscale_factor ** 2))
        return flop


    def check_image_size(self, x):
            _, _, h, w = x.size()
            mod_pad_h = (self.w_size - h % self.w_size) % self.w_size
            mod_pad_w = (self.w_size - w % self.w_size) % self.w_size
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
            return x, mod_pad_h, mod_pad_w
 
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)




class one_conv(nn.Module):
    def __init__(self, G0, G):
        super(one_conv, self).__init__()
        self.G0, self.G = G0, G
        self.conv = nn.Conv2d(G0, G, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        output = self.relu(self.conv(x))
        return torch.cat((x, output), dim=1)

    def flop(self, N):
        flop = N * (self.G0 * 9 + 1) * self.G
        return flop


class RDB(nn.Module):
    def __init__(self, G0, C, G):
        super(RDB, self).__init__()
        self.G0, self.G, self.C = G0, G , C
        self.convs = []
        for i in range(C):
            self.convs.append(one_conv(G0+i*G, G))
        self.conv = nn.Sequential(*self.convs)
        self.LFF = nn.Conv2d(G0 + C*G, G0, kernel_size=1, stride=1, padding=0, bias=True)
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
    def __init__(self, G0, C, G, n_RDB, type='N'):
        super(RDG, self).__init__()
        self.n_RDB = n_RDB
        self.G0 = G0
        self.RDBs = []
        for i in range(n_RDB):
            if type == 'N':
                self.RDBs.append(RDB(G0, C, G))
            elif type == 'P':
                self.RDBs.append(PRDB(G0, C, G))
        self.RDB = nn.ModuleList(self.RDBs)
        self.conv = nn.Conv2d(G0*(n_RDB+1), G0, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, condition=None):
        buffer = x
        temp = []
        for i in range(self.n_RDB):
            buffer = self.RDB[i](buffer)
            temp.append(buffer)
        buffer_cat = torch.cat(temp, dim=1)
        cat_condition = torch.cat([buffer_cat, condition], dim=1) if condition is not None else buffer_cat
        out = self.conv(cat_condition)
        return out, buffer_cat

    def flop(self, N):
        flop = 0
        flop += len(self.RDBs) * self.RDBs[0].flop(N)
        flop += N * (self.n_RDB * self.G0 + 1) * self.G0
        return flop


class P_one_conv(nn.Module):
    def __init__(self, G0, G):
        super(P_one_conv, self).__init__()
        self.G0, self.G = G0, G
        self.conv = nn.Conv2d(G0, G, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))
        

    def flop(self, N):
        flop = N * (self.G0 * 9 + 1) * self.G
        return flop



class MDB(nn.Module):
    def __init__(self, G0, G):
        super(MDB, self).__init__()
        self.G , self.G0 = G , G0
        self.point_wise = nn.Conv2d(G0, G, kernel_size=1, bias=True)
        self.CA = CALayer(G)
        self.conv_1 = nn.Conv2d(G0, G, kernel_size=1, bias=True)
        self.conv_2 = P_one_conv(G, G)

    def forward(self, x):
        out_ca = self.CA(self.point_wise(x))
        out_conv_1 = self.conv_1(x)
        out_conv_2 = self.conv_2(out_conv_1)
        out = torch.cat([x, out_conv_2 + out_ca], dim=1)
        return out
    
    def flop(self, N):
        flop = 0
        flop += self.CA.flop(N)
        flop += 2 * self.G0 * self.G * N
        flop += self.conv_2.flop(N)
        return flop

class PRDB(nn.Module):
    def __init__(self, G0, C, G):
        super(PRDB, self).__init__()
        self.G0, self.G, self.C= G0, G, C
        self.MDBs = nn.ModuleList([MDB(G0 + i * G, G) for i in range(C)])
       
        self.LFF = nn.Conv2d(G0 + C * G, G0, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):

        for i in range(self.C):
            x = self.MDBs[i](x)   

        out = self.LFF(x)
        return out

    def flop(self, N):
        flop = 0
        for mdb in self.MDBs:
            flop += mdb.flop(N)
        flop += N * ((2 * (self.G0 + 2 * self.C *self.G) + 1) * self.G0)
        return flop





class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel = channel
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel//16, 1, padding=0, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(channel//16, channel, 1, padding=0, bias=True),
                nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
    
    def flop(self, N):
        flop = 0
        flop += N * self.channel
        flop += (self.channel + 1) * self.channel//16
        flop += (self.channel//16 + 1) * self.channel
        flop += N * self.channel
        return flop

class ResB(nn.Module):
    def __init__(self, channels):
        self.channel = channels
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
        )
    def forward(self,x):
        out = self.body(x)
        return out + x

    def flop(self, N):
        flop = 2 * N * self.channel * (self.channel * 9 + 1)
        return flop


class PAM(nn.Module):
    def __init__(self, channels, n_RDB):
        super(PAM, self).__init__()
        self.channel = channels
        self.n_RDB = n_RDB
        self.bq = nn.Conv2d(n_RDB*channels, channels, 1, 1, 0, groups=4, bias=True)
        self.bs = nn.Conv2d(n_RDB*channels, channels, 1, 1, 0, groups=4, bias=True)
        self.softmax = nn.Softmax(-1)
        self.rb = ResB(n_RDB * channels)
        self.bn = nn.BatchNorm2d(n_RDB * channels)

    def __call__(self, x_left, x_right, catfea_left, catfea_right, is_training = 0):
        b, c0, h0, w0 = x_left.shape
        Q = self.bq(self.rb(self.bn(catfea_left)))
        b, c, h, w = Q.shape
        Q = Q - torch.mean(Q, 3).unsqueeze(3).repeat(1, 1, 1, w)
        K = self.bs(self.rb(self.bn(catfea_right)))
        K = K - torch.mean(K, 3).unsqueeze(3).repeat(1, 1, 1, w)

        score = torch.bmm(Q.permute(0, 2, 3, 1).contiguous().view(-1, w, c),                    # (B*H) * Wl * C
                          K.permute(0, 2, 1, 3).contiguous().view(-1, c, w))                    # (B*H) * C * Wr
        # (B*H) * Wl * Wr
        M_right_to_left = self.softmax(score)
        # (B*H) * Wr * Wl
        M_left_to_right = self.softmax(score.permute(0, 2, 1))

        M_right_to_left_relaxed = M_Relax(M_right_to_left, num_pixels=2)
        V_left = torch.bmm(M_right_to_left_relaxed.contiguous().view(-1, w).unsqueeze(1),
                           M_left_to_right.permute(
                               0, 2, 1).contiguous().view(-1, w).unsqueeze(2)
                           ).detach().contiguous().view(b, 1, h, w)  # (B*H*Wr) * Wl * 1
        M_left_to_right_relaxed = M_Relax(M_left_to_right, num_pixels=2)
        V_right = torch.bmm(M_left_to_right_relaxed.contiguous().view(-1, w).unsqueeze(1),  # (B*H*Wl) * 1 * Wr
                            M_right_to_left.permute(
                                0, 2, 1).contiguous().view(-1, w).unsqueeze(2)
                            ).detach().contiguous().view(b, 1, h, w)   # (B*H*Wr) * Wl * 1

        V_left_tanh = torch.tanh(5 * V_left)
        V_right_tanh = torch.tanh(5 * V_right)

        x_leftT = torch.bmm(M_right_to_left, x_right.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                            ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2)  # B, C0, H0, W0
        x_rightT = torch.bmm(M_left_to_right, x_left.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                             ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2)  # B, C0, H0, W0
        out_left = x_left * (1 - V_left_tanh.repeat(1, c0, 1, 1)) + \
            x_leftT * V_left_tanh.repeat(1, c0, 1, 1)
        out_right = x_right * (1 - V_right_tanh.repeat(1, c0, 1, 1)) + \
            x_rightT * V_right_tanh.repeat(1, c0, 1, 1)

        if is_training == 1:
            return out_left, out_right, \
                (M_right_to_left.contiguous().view(b, h, w, w), M_left_to_right.contiguous().view(b, h, w, w)),\
                (V_left_tanh, V_right_tanh)
        if is_training == 0:
            return out_left, out_right

    def flop(self, H, W):
        N = H * W
        flop = 0
        flop += 2 * self.rb.flop(N)
        flop += 2 * N * self.channel * (self.n_RDB * self.channel + 1) * (1/4)
        flop += H * (W**2) * self.channel
        # flop += 2 * H * W
        flop += 2 * H * (W**2) * self.channel
        return flop


class SelfAttn(nn.Module):

    def __init__(self, channels, w_size):
        super(SelfAttn, self).__init__()
        self.channel = channels
        self.w_size = w_size
        self.qkv = nn.Conv2d(channels, 3 * channels, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(-1)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        w_size = self.w_size
        b, c, h, w = x.shape
        qkv = self.qkv(self.bn(x)).reshape(b, 3, c, h, w).permute(1, 0, 2, 3, 4)
        Q , K , V = qkv[0], qkv[1], qkv[2]
        score = torch.bmm(Q.permute(0, 2, 3, 1).contiguous().view(-1, w_size, c),                    # (B*H) * Wl * C
                          K.permute(0, 2, 1, 3).contiguous().view(-1, c, w_size))                    # (B*H) * C * Wr
        # (B*H) * Wl * Wr
        attn = self.softmax(score)
        # (B*H) * Wr * Wl

        xT = torch.bmm(attn, V.permute(0, 2, 3, 1).contiguous().view(-1, w_size, c)).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)  # B, C0, H0, W0
        out = xT + x

        return out


    def flop(self, H, W):
            w_size = self.w_size
            N = H * W
            flop = 0
            flop +=  N * self.channel * (self.channel + 1) * 3
            flop += H * W//w_size * (w_size**2) * self.channel
            # flop += 2 * H * W
            flop +=  H * W//w_size * (w_size**2) * self.channel
            return flop

class LightPAM(nn.Module):

    def __init__(self, channels, n_RDB, w_size):
        super(LightPAM, self).__init__()
        self.channel = channels
        self.w_size = w_size
        self.n_RDB = n_RDB
        self.n_RDB = 1
        self.g = 1
        # self.s_attn = SelfAttn(channels, w_size)
        self.bq = nn.Conv2d(self.n_RDB * channels, channels, 1, 1, 0, groups=self.g, bias=True)
        self.bs = nn.Conv2d(self.n_RDB * channels, channels, 1, 1, 0, groups=self.g, bias=True)
        self.softmax = nn.Softmax(-1)
        self.rb = ResB(self.n_RDB * channels)
        self.bn = nn.BatchNorm2d(self.n_RDB * channels)

    def __call__(self, x_left, x_right, catfea_left, catfea_right, is_training=0):
        w_size = self.w_size
        b, c0, h0, w0 = x_left.shape
        # x_left, x_right = self.s_attn(x_left), self.s_attn(x_right)
        Q = self.bq(self.rb(self.bn(x_left)))
        b, c, h, w = Q.shape
        Q = Q - torch.mean(Q, 3).unsqueeze(3).repeat(1, 1, 1, w)
        K = self.bs(self.rb(self.bn(x_right)))
        K = K - torch.mean(K, 3).unsqueeze(3).repeat(1, 1, 1, w)

        score = torch.bmm(Q.permute(0, 2, 3, 1).contiguous().view(-1, w_size, c),                    # (B*H) * Wl * C
                          K.permute(0, 2, 1, 3).contiguous().view(-1, c, w_size))                    # (B*H) * C * Wr
        # (B*H) * Wl * Wr
        M_right_to_left = self.softmax(score)
        # (B*H) * Wr * Wl
        M_left_to_right = self.softmax(score.permute(0, 2, 1))

        M_right_to_left_relaxed = M_Relax(M_right_to_left, num_pixels=2)
        V_left = torch.bmm(M_right_to_left_relaxed.contiguous().view(-1, w_size).unsqueeze(1),
                           M_left_to_right.permute(
                               0, 2, 1).contiguous().view(-1, w_size).unsqueeze(2)
                           ).detach().contiguous().view(b, 1, h, w)  # (B*H*Wr) * Wl * 1
        M_left_to_right_relaxed = M_Relax(M_left_to_right, num_pixels=2)
        V_right = torch.bmm(M_left_to_right_relaxed.contiguous().view(-1, w_size).unsqueeze(1),  # (B*H*Wl) * 1 * Wr
                            M_right_to_left.permute(
                                0, 2, 1).contiguous().view(-1, w_size).unsqueeze(2)
                            ).detach().contiguous().view(b, 1, h, w)   # (B*H*Wr) * Wl * 1

        V_left_tanh = torch.tanh(5 * V_left)
        V_right_tanh = torch.tanh(5 * V_right)

        x_leftT = torch.bmm(M_right_to_left, x_right.permute(0, 2, 3, 1).contiguous().view(-1, w_size, c0)
                            ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2)  # B, C0, H0, W0
        x_rightT = torch.bmm(M_left_to_right, x_left.permute(0, 2, 3, 1).contiguous().view(-1, w_size, c0)
                             ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2)  # B, C0, H0, W0
        out_left = x_left * (1 - V_left_tanh.repeat(1, c0, 1, 1)) + \
            x_leftT * V_left_tanh.repeat(1, c0, 1, 1)
        out_right = x_right * (1 - V_right_tanh.repeat(1, c0, 1, 1)) + \
            x_rightT * V_right_tanh.repeat(1, c0, 1, 1)

        if is_training == 1:
            return out_left, out_right, \
                (M_right_to_left.contiguous().view(b, h, w, w), M_left_to_right.contiguous().view(b, h, w, w)),\
                (V_left_tanh, V_right_tanh)
        if is_training == 0:
            return out_left, out_right

    def flop(self, H, W):
        w_size = self.w_size
        N = H * W
        flop = 0
        # flop+= 2 * self.s_attn.flop(H,W)
        flop += 3 * self.rb.flop(N)
        flop += 3 * N * self.channel * (self.n_RDB * self.channel + 1) * (1/self.g)
        flop += H * W//w_size * (w_size**2) * self.channel
        # flop += 2 * H * W
        flop += 2 * H * W//w_size * (w_size**2) * self.channel
        return flop


def M_Relax(M, num_pixels):
    _, u, v = M.shape
    M_list = []
    M_list.append(M.unsqueeze(1))
    for i in range(num_pixels):
        pad = nn.ZeroPad2d(padding=(0, 0, i+1, 0))
        pad_M = pad(M[:, :-1-i, :])
        M_list.append(pad_M.unsqueeze(1))
    for i in range(num_pixels):
        pad = nn.ZeroPad2d(padding=(0, 0, 0, i+1))
        pad_M = pad(M[:, i+1:, :])
        M_list.append(pad_M.unsqueeze(1))
    M_relaxed = torch.sum(torch.cat(M_list, 1), dim=1)
    return M_relaxed


if __name__ == "__main__":
    # from utils import disparity_alignment
    # from StreoSwinSR import CoSwinAttn
    # from SwinTransformer import SwinAttn
    H, W, C = 360, 640, 3
    net = Net(upscale_factor=4, model='MDB_lightpam', img_size=tuple([H, W]), input_channel=C, w_size=40).cuda()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    net.train(False)
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
    print('   FLOPS: %.2fG' % (net.flop(H, W) / 1e9))
    x = torch.clamp(torch.randn((1, 7, H, W)) , min=0.0).cuda()
    input_names = ['x_left', 'x_right']
    input_names = ['x_left', 'x_right']
    exc_time = 0.0
    n_itr = 10
    with torch.no_grad():
        for _ in range(10):
            _, _ = net(x, x, 0)
        for _ in range(n_itr):
            starter.record()
            _, _ = net(x, x, 0)
            ender.record()
            torch.cuda.synchronize()
            elps = starter.elapsed_time(ender)
            exc_time += elps
            print('################## total: ', elps /
                    1000, ' #######################')

    print('exec time: ', exc_time / n_itr / 1000)
