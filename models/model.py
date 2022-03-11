from tkinter import N
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from torchvision import transforms

class Net(nn.Module):
    def __init__(self, upscale_factor, input_channel = 3, w_size = 8, device='cpu'):
        super(Net, self).__init__()
        self.input_channel = input_channel
        self.upscale_factor = upscale_factor
        self.init_feature = nn.Conv2d(input_channel, 64, 3, 1, 1, bias=True)
        self.deep_feature = RDG(G0=64, C=4, G=24, n_RDB=4)
        self.pam = PAM(64 ,w_size, device)
        self.f_RDB = RDB(G0=128, C=4, G=32)
        self.CAlayer = CALayer(128)
        self.fusion = nn.Sequential(self.f_RDB, self.CAlayer, nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=True))
        self.reconstruct = RDG(G0=64, C=4, G=24, n_RDB=4)
        self.upscale = nn.Sequential(nn.Conv2d(64, 64 * upscale_factor ** 2, 1, 1, 0, bias=True), nn.PixelShuffle(upscale_factor), nn.Conv2d(64, 3, 3, 1, 1, bias=True))


    def forward(self, x_left, x_right, is_training = 0):
        b, c, h, w = x_left.shape
        if c > 3:
            d_left = x_left[:, 3]
            d_right = x_right[:, 3]

        x_left_upscale = F.interpolate(x_left[:, :3], scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        x_right_upscale = F.interpolate(x_right[:, :3], scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        buffer_left = self.init_feature(x_left)
        buffer_right = self.init_feature(x_right)
        buffer_left, catfea_left = self.deep_feature(buffer_left)
        buffer_right, catfea_right = self.deep_feature(buffer_right)
        buffer_leftT, buffer_rightT = self.pam(buffer_left, buffer_right, catfea_left, catfea_right, d_left, d_right)

        buffer_leftF = self.fusion(torch.cat([buffer_left, buffer_leftT], dim=1))
        buffer_rightF = self.fusion(torch.cat([buffer_right, buffer_rightT], dim=1))
        buffer_leftF, _ = self.reconstruct(buffer_leftF)
        buffer_rightF, _ = self.reconstruct(buffer_rightF)
        out_left = self.upscale(buffer_leftF) + x_left_upscale
        out_right = self.upscale(buffer_rightF) + x_right_upscale
        return out_left, out_right

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

    def flop(self, H, W):
        N = H * W
        flop = 0
        flop += 2 * ((self.input_channel * 9 + 1) * N * 64) # adding 1 for bias
        flop += 2 * self.deep_feature.flop(N)
        flop += self.pam.flop(H, W)
        flop += 2 * (self.f_RDB.flop(N) + self.CAlayer.flop(N) + N * (128 + 1) * 64)
        flop += 2 * self.reconstruct.flop(N)
        flop += 2 * (N * (64 + 1) * 64 * (self.upscale_factor ** 2) + (N**self.upscale_factor) * 3 * (64 * 9 + 1))
        return flop



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
    def __init__(self, G0, C, G, n_RDB):
        super(RDG, self).__init__()
        self.n_RDB = n_RDB
        self.G0 = G0
        self.RDBs = []
        for i in range(n_RDB):
            self.RDBs.append(RDB(G0, C, G))
        self.RDB = nn.Sequential(*self.RDBs)
        self.conv = nn.Conv2d(G0*n_RDB, G0, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        buffer = x
        temp = []
        for i in range(self.n_RDB):
            buffer = self.RDB[i](buffer)
            temp.append(buffer)
        buffer_cat = torch.cat(temp, dim=1)
        out = self.conv(buffer_cat)
        return out, buffer_cat

    def flop(self, N):
        flop = 0
        flop += len(self.RDBs) * self.RDBs[0].flop(N)
        flop += N * (self.n_RDB * self.G0 + 1) * self.G0
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
    def __init__(self, channels, w_size = 8, device='cpu'):
        super(PAM, self).__init__()
        self.device = device
        self.w_size = w_size
        self.channel = channels
        self.bq = nn.Conv2d(4*channels, channels, 1, 1, 0, groups=4, bias=True)
        self.bs = nn.Conv2d(4*channels, channels, 1, 1, 0, groups=4, bias=True)
        self.softmax = nn.Softmax(-1)
        self.rb = ResB(4 * channels)
        self.bn = nn.BatchNorm2d(4 * channels)
        self.window_centre_table = None


    
    def patchify(self, tensor, b, c, h, w):
        w_size = self.w_size
        return tensor.reshape(b, c, h // w_size, w_size, w // w_size, w_size).permute(0, 2, 4, 3, 5, 1) # B C H//w_size W//wsize wsize wsize

    def unpatchify(self, tensor, b, c, h, w):
        w_size = self.w_size
        return tensor.reshape(b, h // w_size, w // w_size, w_size, w_size, c).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

    def __call__(self, x_left, x_right, catfea_left, catfea_right, d_left, d_right):
        # Building matching indexes and patch around that using disparity
        w_size = self.w_size
        b, c, h, w = x_left.shape
        coords_b, coords_h, coords_w = torch.meshgrid([torch.arange(b), torch.arange(h), torch.arange(w)], indexing='ij') # H, W
        m_left = ((coords_w.to(self.device) - d_left.long() ) >= 0).unsqueeze(1).int() # B , H , W
        m_right = ((coords_w.to(self.device) + d_right.long()) <= w - 1).unsqueeze(1).int() # B , H , W
        r2l_w = (torch.clamp(coords_w - d_left.long().cpu(), min=0) )
        l2r_w = (torch.clamp(coords_w + d_right.long().cpu(), max=w - 1))

        Q = self.bq(self.rb(self.bn(catfea_left))) # B C H W
        # Q = Q - torch.mean(Q, 3).unsqueeze(3).repeat(1, 1, 1, w)
        K = self.bs(self.rb(self.bn(catfea_right)))  # B C H W
        # K = K - torch.mean(K, 3).unsqueeze(3).repeat(1, 1, 1, w)
        # B , C , W// , H//, w_size w_size
        Q_selected = self.patchify(Q[coords_b, :, coords_h, l2r_w], b, c, h, w)
        K_selected = self.patchify(K[coords_b, :, coords_h, r2l_w], b, c, h, w)  # B , C , W//wsize , H//wsize, w_size, w_size
        Q_selected = Q_selected - Q_selected.mean((4, 5))[..., None, None]
        K_selected = K_selected - K_selected.mean((4, 5))[..., None, None]
        score_r2l = self.patchify(Q, b, c, h, w).reshape(-1, w_size * w_size, c) @ K_selected.permute(0, 1, 2, 5, 3, 4).reshape(-1, c, w_size * w_size)
        score_l2r = self.patchify(K, b, c, h, w).reshape(-1, w_size * w_size, c) @ Q_selected.permute(0, 1, 2, 5, 3, 4).reshape(-1, c, w_size * w_size)
        # (B*H) * Wl * Wr
        Mr2l = self.softmax(score_r2l)  # B*C*H//*W//, wsize * wsize, wsize * wsize
        Ml2r = self.softmax(score_l2r)  
        ## masks
        Mr2l_relaxed = M_Relax(Mr2l, num_pixels=2)
        Ml2r_relaxed = M_Relax(Ml2r, num_pixels=2)
        V_left = Mr2l_relaxed.reshape(-1, 1, w_size*w_size) @ Ml2r.permute(0, 2, 1).reshape(-1, w_size*w_size, 1)
        V_left = self.unpatchify(V_left.squeeze().reshape(-1, w_size, w_size, 1) , b, 1, h, w).detach()
        V_right = Ml2r_relaxed.reshape(-1, 1, w_size*w_size) @ Mr2l.permute(0, 2, 1).reshape(-1, w_size*w_size, 1)
        V_right = self.unpatchify(V_right.squeeze().reshape(-1, w_size, w_size, 1) , b, 1, h, w).detach()
        V_left = torch.tanh(5 * V_left)
        V_right = torch.tanh(5 * V_right)

        x_right_selected = self.patchify(x_right[coords_b, :, coords_h, r2l_w], b, c, h, w) # B, C, H//, W//, wsize, wsize
        x_left_selected = self.patchify(x_left[coords_b, :, coords_h, l2r_w], b, c, h, w) # B, C, H, W, wsize, wsize
        x_leftT = Mr2l @ x_right_selected.reshape(-1, w_size * w_size, c)
        x_rightT = Ml2r @ x_left_selected.reshape(-1, w_size * w_size, c)
        x_leftT = self.unpatchify(x_leftT, b, c, h, w)  # B, C, H , W
        x_rightT = self.unpatchify(x_rightT, b, c, h, w)  # B, C, H , W
        out_left = x_left  + x_leftT * m_left
        out_right = x_right + x_rightT * m_right
        return out_left, out_right

    def flop(self, H, W):
        N = H * W
        flop = 0
        flop += 2 * N * 4 * self.channel
        flop += 2 * self.rb.flop(N)
        flop += 2 * N * self.channel * (4 * self.channel + 1)
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
    net = Net(upscale_factor=2, input_channel=10)
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
    print('   FLOPS: %.2fG' % (net.flop(30 , 90) / 1e9))
    x = torch.randn((2, 10, 10, 30))
    y = net(x, x , 0)
