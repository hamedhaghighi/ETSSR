from tkinter import N
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from torchvision import transforms
import torch.utils.checkpoint as checkpoint
from models.SwinTransformer import SwinAttn, CoSwinAttnBlock
from timm.models.layers import trunc_normal_

class Net(nn.Module):
    def __init__(self, upscale_factor, img_size, model, input_channel = 3, w_size = 8,device='cpu'):
        super(Net, self).__init__()
        self.model = model
        self.input_channel = input_channel
        self.upscale_factor = upscale_factor
        self.init_feature = nn.Conv2d(input_channel, 64, 3, 1, 1, bias=True)
        self.deep_feature= SwinAttn(img_size=img_size, window_size=w_size, depths=[
                          6, 6, 6, 6], embed_dim=64, num_heads=[8, 8, 8, 8], mlp_ratio=2)
        if model == 'swin_pam':
            self.co_feature = PAM(64)
        elif model == 'all_swin':
            self.co_feature = CoSwinAttn(img_size=img_size, window_size=w_size, depths=[6, 6, 6, 6], embed_dim=64, num_heads=[8, 8, 8, 8], mlp_ratio=2)
        self.CAlayer = CALayer(128)
        self.fusion = nn.Sequential(self.CAlayer, nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=True))
        self.reconstruct = SwinAttn(img_size=img_size, window_size=w_size, depths=[
            6, 6, 6, 6], embed_dim=64, num_heads=[8, 8, 8, 8], mlp_ratio=2)
        self.upscale = nn.Sequential(nn.Conv2d(64, 64 * upscale_factor ** 2, 1, 1, 0, bias=True), nn.PixelShuffle(upscale_factor), nn.Conv2d(64, 3, 3, 1, 1, bias=True))
        self.apply(self._init_weights)

    def forward(self, x_left, x_right, is_training = 0):
        b, c, h, w = x_left.shape
        if c > 3:
            d_left = x_left[:, 3]
            d_right = x_right[:, 3]

        x_left_upscale = F.interpolate(x_left[:, :3], scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        x_right_upscale = F.interpolate(x_right[:, :3], scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        buffer_left = self.init_feature(x_left)
        buffer_right = self.init_feature(x_right)
        buffer_left = self.deep_feature(buffer_left)
        buffer_right = self.deep_feature(buffer_right)
        if self.model == 'swin_pam':
            buffer_leftT, buffer_rightT = self.co_feature(buffer_left, buffer_right)
        elif self.model == 'all_swin':
            buffer_leftT, buffer_rightT = self.co_feature(buffer_left, buffer_right, d_left, d_right)

        buffer_leftF = self.fusion(torch.cat([buffer_left, buffer_leftT], dim=1))
        buffer_rightF = self.fusion(torch.cat([buffer_right, buffer_rightT], dim=1))
        buffer_leftF = self.reconstruct(buffer_leftF)
        buffer_rightF = self.reconstruct(buffer_rightF)
        out_left = self.upscale(buffer_leftF) + x_left_upscale
        out_right = self.upscale(buffer_rightF) + x_right_upscale
        return out_left, out_right

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
        mod_pad_h = (self.window_size - h %
                     self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w %
                     self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

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


class CoRSTB(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(CoRSTB, self).__init__()

        self.use_checkpoint = use_checkpoint
        self.dim = dim
        self.input_resolution = input_resolution
        self.blocks = nn.ModuleList([
            CoSwinAttnBlock(dim=dim, input_resolution=input_resolution,
                          num_heads=num_heads, window_size=window_size,
                          shift_size=0 if (
                              i % 2 == 0) else window_size // 2,
                          mlp_ratio=mlp_ratio,
                          qkv_bias=qkv_bias,
                          drop=drop,
                          drop_path=drop_path[i] if isinstance(
                              drop_path, list) else drop_path,
                          norm_layer=norm_layer)
            for i in range(depth)])

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x_left, x_right, d_left, d_right, x_size):
        out_left = x_left
        out_right = x_right
        for blk in self.blocks:
            if self.use_checkpoint:
                x_left, x_right = checkpoint.checkpoint(blk, x_left, x_right, d_left, d_right, x_size)
            else:
                x_left, x_right = blk(x_left, x_right, d_left, d_right, x_size)
        x_left = x_left.transpose(1, 2).view(-1, self.dim, x_size[0], x_size[1])
        x_right = x_right.transpose(1, 2).view(-1, self.dim, x_size[0], x_size[1])
        x_left = self.conv(x_left)
        x_right = self.conv(x_right)
        x_left = x_left.flatten(2).transpose(1, 2)
        x_right = x_right.flatten(2).transpose(1, 2)
        return x_left + out_left, x_right + out_right

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class CoSwinAttn(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, resi_connection='1conv',
                 **kwargs):
        super(CoSwinAttn, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        self.window_size = window_size

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        num_patches = img_size[0] * img_size[1]
        patches_resolution = [img_size[0], img_size[1]]
        self.patches_resolution = patches_resolution
        self.pre_norm = norm_layer(embed_dim)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = CoRSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias,
                         drop_path=dpr[sum(depths[:i_layer]):sum(
                             depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection

                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        


    

    def forward(self, x_left, x_right, d_left, d_right):

        x_size = (x_left.shape[2], x_left.shape[3])
        x_left = x_left.flatten(2).transpose(1, 2)
        x_right = x_right.flatten(2).transpose(1, 2)
        x_left = self.pre_norm(x_left)
        x_right = self.pre_norm(x_right)
        for layer in self.layers:
            x_left, x_right = layer(x_left, x_right, d_left, d_right, x_size)

        x_left = self.norm(x_left)  # B L C
        x_right = self.norm(x_right)
        B, HW, C = x_left.shape
        x_left = x_left.transpose(1, 2).view(B, C, x_size[0], x_size[1])
        x_right = x_right.transpose(1, 2).view(B, C, x_size[0], x_size[1])
        return x_left, x_right


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

        x_leftT = (M_right_to_left @ x_right.permute(0, 2, 3, 1).contiguous().view(-1, w, c)
                            ).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)                           #  B, C0, H0, W0
        x_rightT = (M_left_to_right @ x_left.permute(0, 2, 3, 1).contiguous().view(-1, w, c)
                            ).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)                              #  B, C0, H0, W0
        out_left = x_left * (1 - V_left_tanh.repeat(1, c, 1, 1)) + x_leftT * V_left_tanh.repeat(1, c, 1, 1)
        out_right = x_right * (1 - V_right_tanh.repeat(1, c, 1, 1)) +  x_rightT * V_right_tanh.repeat(1, c, 1, 1)
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
    input_shape = (10, 30)
    net = Net(upscale_factor=2, img_size=input_shape, model='swin_pam', input_channel=10, w_size=10)
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
    # print('   FLOPS: %.2fG' % (net.flop(30 , 90) / 1e9))
    x = torch.clamp(torch.randn((2, 10, input_shape[0], input_shape[1])) , min=0.0)
    y_l, y_r = net(x, x , 0)
    print (y_l.shape)
