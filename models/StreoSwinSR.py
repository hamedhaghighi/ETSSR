from tkinter import N
from turtle import window_width
from scipy.fftpack import shift
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/oem/Documents/PhD_proj/iPASSR')
sys.path.append('/home/haghig_h@WMGDS.WMG.WARWICK.AC.UK/Documents/StereoSR')
from timm.models.layers import trunc_normal_


class Net(nn.Module):
    def __init__(self, upscale_factor, img_size, model, input_channel = 3, w_size = 8, embed_dim =64):
        super(Net, self).__init__()
        if 'light' in model:
            from models.LightSwinTransformer import SwinAttn
            from models.LightCoSwinTransformer import SwinAttnInterleaved, CoSwinAttn
        else:
            from models.SwinTransformer import SwinAttn
            from models.CoSwinTransformer import SwinAttnInterleaved, CoSwinAttn

        self.model = model
        self.w_size = w_size
        self.input_channel = input_channel
        self.upscale_factor = upscale_factor
        self.img_size = img_size
        self.condition = False
        if 'CP' in self.model:
            self.condition_feature = nn.Sequential(nn.Conv2d(4, 64, 3, 1, 1, bias=True), nn.LeakyReLU(0.1, inplace=True), nn.Conv2d(64, 64, 3, 1, 1, bias=True))
            self.condition = True

        self.init_feature = nn.Conv2d(input_channel, embed_dim, 3, 1, 1, bias=True)
        depths = [4]
        num_heads = [4]
        if 'swin_interleaved' in self.model:
            self.deep_feature = SwinAttnInterleaved(img_size=img_size, window_size=w_size, depths=depths, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=2)
        else:
            self.deep_feature = SwinAttn(img_size=img_size, window_size=w_size, depths=depths, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=2)
            if 'swin_pam' in self.model:
                self.co_feature = PAM(embed_dim)
            elif 'all_swin' in self.model:
                self.co_feature = CoSwinAttn(img_size=img_size, window_size=w_size, depths=[2], embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=2)
            elif 'indp' in self.model:
                self.swin = SwinAttn(img_size=img_size, window_size=w_size, depths=[2], embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=2)
            self.CAlayer = CALayer(embed_dim * 2)
            self.fusion = nn.Sequential(self.CAlayer, nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, stride=1, padding=0, bias=True))
            self.reconstruct = SwinAttn(img_size=img_size, window_size=w_size, depths=depths, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=2)
        self.upscale = nn.Sequential(nn.Conv2d(64, 64 * upscale_factor ** 2, 1, 1, 0, bias=True), nn.PixelShuffle(upscale_factor), nn.Conv2d(64, 3, 3, 1, 1, bias=True))

        self.apply(self._init_weights)

    def forward(self, x_left, x_right):
        b, c, h, w = x_left.shape
        x_left, mod_pad_h, mod_pad_w = self.check_image_size(x_left)
        x_right, _, _  = self.check_image_size(x_right)
        if c > 3:
            d_left = x_left[:, 3]
            d_right = x_right[:, 3]
        x_left_upscale = F.interpolate(x_left[:, :3], scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        x_right_upscale = F.interpolate(x_right[:, :3], scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)

        CF_left = self.condition_feature(x_left[:, 3:]) if 'CP' in self.model else None
        CF_right = self.condition_feature(x_right[:, 3:]) if 'CP' in self.model else None

        buffer_left = self.init_feature(x_left[:, :self.input_channel])
        buffer_right = self.init_feature(x_right[:, :self.input_channel])
        if 'swin_interleaved' in self.model:
            buffer_leftF, buffer_rightF = self.deep_feature(buffer_left, buffer_right, d_left, d_right)
        else:
            buffer_left = self.deep_feature(buffer_left, CF_left)
            buffer_right = self.deep_feature(buffer_right, CF_right)
            if 'swin_pam' in self.model:
                buffer_leftT, buffer_rightT = self.co_feature(buffer_left, buffer_right)
            elif 'all_swin' in self.model:
                if 'wo_d' in self.model:
                    buffer_leftT, buffer_rightT = self.co_feature(buffer_left, buffer_right)
                else:
                    buffer_leftT, buffer_rightT = self.co_feature(buffer_left, buffer_right, d_left, d_right)
            elif 'indp' in self.model:
                buffer_leftT, buffer_rightT = self.swin(buffer_left), self.swin(buffer_right)
            buffer_leftF = self.fusion(torch.cat([buffer_left, buffer_leftT], dim=1))
            buffer_rightF = self.fusion(torch.cat([buffer_right, buffer_rightT], dim=1))
            buffer_leftF = self.reconstruct(buffer_leftF, CF_left)
            buffer_rightF = self.reconstruct(buffer_rightF, CF_right)
        out_left = self.upscale(buffer_leftF) + x_left_upscale
        out_right = self.upscale(buffer_rightF) + x_right_upscale
        mod_h = -mod_pad_h * self.upscale_factor if mod_pad_h != 0 else None
        mod_w = -mod_pad_w * self.upscale_factor if mod_pad_w != 0 else None
        return out_left[..., :mod_h, :mod_w], out_right[..., :mod_h, :mod_w]

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_losses(self):
        loss_dict = {k: getattr(self, 'loss_' + k).data.cpu() for k in self.loss_names}
        return loss_dict

    def calc_loss(self, LR_left, LR_right, HR_left, HR_right, cfg):
        self.loss_names = ['SR']
        scale = cfg.scale_factor
        alpha = cfg.alpha
        criterion_L1 = torch.nn.L1Loss().to(cfg.device)
        SR_left, SR_right = self.forward(LR_left, LR_right)
        ''' SR Loss '''
        self.loss_SR = criterion_L1(SR_left, HR_left) + criterion_L1(SR_right, HR_right)
        return self.loss_SR

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.w_size - h % self.w_size) % self.w_size
        mod_pad_w = (self.w_size - w % self.w_size) % self.w_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x, mod_pad_h, mod_pad_w

    def flop(self, H, W):
        N = H * W
        flop = 0
        if 'CP' in self.model:
            flop += 2 * (N * (4 * 9 + 1) * 64 + N * (64 * 9 + 1) * 64)
        # init feature
        flop += 2 * ((self.input_channel * 9 + 1) * N * 64) # adding 1 for bias
        if 'swin_interleaved' in self.model:
            flop += self.deep_feature.flops(H, W)
        else:
            flop += 2 * self.deep_feature.flops(H, W)
            if 'swin_pam' in self.model:
                flop += self.co_feature.flop(H, W)
            elif 'all_swin' in self.model:
                flop += self.co_feature.flops(H, W)
            flop += 2 * self.deep_feature.flops(H, W)
            flop += 2 * (self.CAlayer.flop(N) + N * (128 + 1) * 64)
            flop += 2 * self.reconstruct.flops(H, W)
        flop += 2 * (N * (64 + 1) * 64 * (self.upscale_factor ** 2) + N * (self.upscale_factor ** 2) * (64 * 9 + 1) * 3)
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
    def __call__(self,x):
        out = self.body(x)
        return out + x

    def flop(self, N):
        flop = 2 * N * self.channel * (self.channel * 9 + 1)
        return flop



class PAM(nn.Module):
    def __init__(self, channels, device='cpu'):
        super(PAM, self).__init__()
        self.device = device
        self.channel = channels
        self.Q = nn.Conv2d(channels, channels, 1, 1, 0, groups=4, bias=True)
        self.K = nn.Conv2d(channels, channels, 1, 1, 0, groups=4, bias=True)
        self.softmax = nn.Softmax(-1)
        self.rb = ResB(channels)
        self.bn = nn.BatchNorm2d(channels)
        self.window_centre_table = None


    def __call__(self, x_left, x_right):
        # Building matching indexes and patch around that using disparity
        b, c, h, w = x_left.shape

        Q = self.Q(self.rb(self.bn(x_left)))
        K = self.K(self.rb(self.bn(x_right)))

        Q = Q - torch.mean(Q, 3).unsqueeze(3).repeat(1, 1, 1, w)
        K = K - torch.mean(K, 3).unsqueeze(3).repeat(1, 1, 1, w)

        score = Q.permute(0, 2, 3, 1).contiguous().view(-1, w, c)@ K.permute(0, 2, 1, 3).contiguous().view(-1, c, w)
        M_right_to_left = self.softmax(score)                                                   # (B*H) * Wl * Wr
        M_left_to_right = self.softmax(score.permute(0, 2, 1))                                  # (B*H) * Wr * Wl

        M_right_to_left_relaxed = M_Relax(M_right_to_left, num_pixels=2)
        V_left = (M_right_to_left_relaxed.contiguous().view(-1, w).unsqueeze(1) @ M_left_to_right.permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2)).detach().contiguous().view(b, 1, h, w)  # (B*H*Wr) * Wl * 1
        M_left_to_right_relaxed = M_Relax(M_left_to_right, num_pixels=2)
        V_right = (M_left_to_right_relaxed.contiguous().view(-1, w).unsqueeze(1) @  # (B*H*Wl) * 1 * Wr
                            M_right_to_left.permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2)
                                  ).detach().contiguous().view(b, 1, h, w)   # (B*H*Wr) * Wl * 1

        V_left_tanh = torch.tanh(5 * V_left)
        V_right_tanh = torch.tanh(5 * V_right)

        x_leftT = (M_right_to_left @ x_right.permute(0, 2, 3, 1).contiguous().view(-1, w, c)).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)                           #  B, C0, H0, W0
        x_rightT = (M_left_to_right @ x_left.permute(0, 2, 3, 1).contiguous().view(-1, w, c)).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)                              #  B, C0, H0, W0
        out_left = x_left * (1 - V_left_tanh.repeat(1, c, 1, 1)) + x_leftT * V_left_tanh.repeat(1, c, 1, 1)
        out_right = x_right * (1 - V_right_tanh.repeat(1, c, 1, 1)) +  x_rightT * V_right_tanh.repeat(1, c, 1, 1)
        return out_left, out_right

    def flop(self, H, W):
        N = H * W
        flop = 0
        flop += 2 * self.rb.flop(N)
        flop += 2 * N * self.channel * (self.channel + 1)
        flop += H * (W**2) * self.channel
        # flop += 2 * H * W
        flop += 2 * H * (W**2) * self.channel
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
    input_shape = (360, 640)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    net = Net(upscale_factor=4, img_size=(360, 645), model='all_swin_CP', input_channel=3, w_size=15).cuda()
    net.train(False)
    total = sum([param.nelement() for param in net.parameters()])
    x = torch.clamp(torch.randn((1, 7, input_shape[0], input_shape[1])) , min=0.0).cuda()
    print('   Number of params: %.2fM' % (total / 1e6))
    print('   FLOPS: %.2fG' % (net.flop(input_shape[0], input_shape[1]) / 1e9))
    exc_time = 0.0
    n_itr = 10
    for _ in range(10):
        _, _ = net(x, x)
    with torch.no_grad():
        for _ in range(n_itr):
            starter.record()
            _, _ = net(x, x)
            ender.record()
            torch.cuda.synchronize()
            elps = starter.elapsed_time(ender)
            exc_time += elps
            print('################## total: ', elps / 1000, ' #######################')

    print('exec time: ', exc_time / n_itr / 1000)
