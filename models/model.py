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
            d_right = x_left[:, 3]

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


    def calculate_window_center_table(self, h, w):
        w_size = self.w_size
        table_h = torch.zeros(h + w_size, w + w_size, self.w_size, self.w_size).long()
        table_w = torch.zeros(h + w_size, w + w_size, self.w_size, self.w_size).long()
        coords_h, coords_w = torch.meshgrid([torch.arange(h + w_size//2), torch.arange(w + w_size//2)], indexing='ij')
        coords_h, coords_w = coords_h, coords_w
        for i in range(w_size//2, h):
            for j in range(w_size//2, w):
                table_h[i , j] = coords_h[i - w_size//2: i + w_size//2, j - w_size//2: j + w_size//2]
                table_w[i , j] = coords_w[i - w_size//2: i + w_size//2, j - w_size//2: j + w_size//2]
        self.window_centre_table = (table_h, table_w)

    def select_patch(self, tensor, coords_h, coords_w, b_size):
        w_size = self.w_size
        tensor_padded = F.pad(tensor, (w_size//2, w_size//2, w_size//2, w_size//2))
        tensor_selected = torch.cat([tensor_padded[i: i+1, :,  coords_h[i], coords_w[i]] for i in range(b_size)])  # B , C, H , W , wsize, wsize
        return tensor_selected

    def __call__(self, x_left, x_right, catfea_left, catfea_right, d_left, d_right):
        # Building matching indexes and patch around that using disparity
        w_size = self.w_size
        b, c, h, w = x_left.shape
        coords_h, coords_w = torch.meshgrid([torch.arange(h), torch.arange(w)], indexing='ij') # H, W
        coords_h, coords_w = coords_h.repeat(b ,1, 1) + w_size//2 , coords_w.repeat(b, 1, 1) # B , H , W
        V_left = ((coords_w.to(self.device) - d_left.long() ) >= 0).unsqueeze(1).int() # B , H , W
        V_Right = ((coords_w.to(self.device) + d_right.long()) <= w - 1).unsqueeze(1).int() # B , H , W
        r2l_w = (torch.clamp(coords_w - d_left.long().cpu(), min=0) + w_size//2)
        l2r_w = (torch.clamp(coords_w + d_right.long().cpu(), max=w - 1) + w_size//2)
        if self.window_centre_table is None:
            self.calculate_window_center_table(h, w)
        table_h, table_w = self.window_centre_table 
        Wr2l_h, Wr2l_w = table_h[coords_h, r2l_w], table_w[coords_h, r2l_w] # B , H , W ,wsize, wsize
        Wl2r_h, Wl2r_w = table_h[coords_h, l2r_w], table_w[coords_h, l2r_w]

        Q = self.bq(self.rb(self.bn(catfea_left))) # B C H W
        # Q = Q - torch.mean(Q, 3).unsqueeze(3).repeat(1, 1, 1, w)
        K = self.bs(self.rb(self.bn(catfea_right)))  # B C H W
        # K = K - torch.mean(K, 3).unsqueeze(3).repeat(1, 1, 1, w)
        Q_selected = self.select_patch(Q, Wl2r_h, Wl2r_w, b) # B, C, H, W, wsize, wsize
        K_selected = self.select_patch(K, Wr2l_h, Wr2l_w, b) # B, C, H , W ,wsize, wsize
        Q_selected = Q_selected - Q_selected.mean((4, 5))[..., None, None]
        K_selected = K_selected - K_selected.mean((4, 5))[..., None, None]
        score_r2l = Q.permute(0, 2, 3, 1).reshape(-1, 1, c) @ K_selected.permute(0, 2, 3, 1, 4, 5).reshape(-1, c, w_size * w_size)
        score_l2r = K.permute(0, 2, 3, 1).reshape(-1, 1, c) @ Q_selected.permute(0, 2, 3, 1, 4, 5).reshape(-1, c, w_size * w_size)
        # (B*H) * Wl * Wr
        Mr2l = self.softmax(score_r2l)  # B*C*H*W, 1 , wsize * wsize
        Ml2r = self.softmax(score_l2r)                     

        x_right_selected = self.select_patch(x_right, Wr2l_h, Wr2l_w, b) # B, C, H, W, wsize, wsize
        x_left_selected = self.select_patch(x_left, Wr2l_h, Wr2l_w, b) # B, C, H, W, wsize, wsize
        x_leftT = Mr2l @ x_right_selected.permute(0, 2, 3, 4, 5, 1).reshape(-1, w_size * w_size, c)
        x_rightT = Ml2r @ x_left_selected.permute(0, 2, 3, 4, 5, 1).reshape(-1, w_size * w_size, c)
        x_leftT = x_leftT.reshape(b, h, w, c).permute(0, 3, 1, 2) # B, C, H , W
        x_rightT = x_rightT.reshape(b, h, w, c).permute(0, 3, 1, 2) # B, C, H , W
        out_left = x_left + x_leftT * V_left.repeat(1, c, 1, 1)
        out_right = x_right +  x_rightT * V_Right.repeat(1, c, 1, 1)
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
