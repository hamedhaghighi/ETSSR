from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
sys.path.append('/home/oem/Documents/PhD_proj/iPASSR')
sys.path.append('/home/haghig_h@WMGDS.WMG.WARWICK.AC.UK/Documents/StereoSR')
from utils import disparity_alignment
from models.CoSwinTransformer import CoSwinAttn
from models.SwinTransformer import SwinAttn
from dataset import toNdarray
import math

class Net(nn.Module):
    def __init__(self, upscale_factor, img_size, model, input_channel=3, w_size=8, device='cpu'):
        super(Net, self).__init__()
        self.input_channel = 20 if 'seperate' in model else input_channel
        self.upscale_factor = upscale_factor
        self.f_uf = float(self.upscale_factor)
        self.model = model
        self.w_size = w_size
        self.init_feature = nn.Conv2d(self.input_channel, 64, 3, 1, 1, bias=True)
        depths = [1 ,1]
        num_heads = [4, 4]
        self.deep_feature = SwinAttn(img_size=img_size, window_size=w_size, depths=depths, embed_dim=64, num_heads=num_heads, mlp_ratio=2)
        # self.co_feature = SwinAttn(img_size=img_size, window_size=w_size, depths=depths, embed_dim=64, num_heads=num_heads, mlp_ratio=2)
        self.co_feature = CoSwinAttn(img_size=img_size, window_size=w_size, depths=depths, embed_dim=64, num_heads=num_heads, mlp_ratio=2)
        self.f_RDB = RDB(G0=128, C=4, G=32)
        self.CAlayer = CALayer(128)
        self.fusion = nn.Sequential(self.f_RDB, self.CAlayer, nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=True))
        self.reconstruct = SwinAttn(img_size=img_size, window_size=w_size, depths=depths, embed_dim=64, num_heads=num_heads, mlp_ratio=2)
        self.upscale = nn.Sequential(nn.Conv2d(64, 3, 3, 1, 1, bias=True), nn.Conv2d(3, 3 * upscale_factor ** 2, 1, 1, 0, bias=True), nn.PixelShuffle(upscale_factor))
        self.apply(self._init_weights)


    def forward(self, x_left, x_right):
        # if not 'pam' in self.model:
        #     x_left, mod_pad_h, mod_pad_w = self.check_image_size(x_left)
        #     x_right, _, _ = self.check_image_size(x_right)
        # else:
        #     mod_pad_h, mod_pad_w = 0, 0
        # b, c, h, w = x_left.shape
        d_left = x_left[:, 3]
        d_right = x_right[:, 3]

        output_size = [int(math.floor(float(x_left.size(i + 2)) * self.f_uf)) for i in range(2)]
        x_left_upscale= torch._C._nn.upsample_bilinear2d(x_left[:, :3], output_size, False, self.f_uf)
        x_right_upscale= torch._C._nn.upsample_bilinear2d(x_right[:, :3], output_size, False, self.f_uf)
        # x_left_upscale = F.interpolate(x_left[:, :3], scale_factor=self.f_uf, mode='bilinear', align_corners=False)
        # x_right_upscale = F.interpolate(x_right[:, :3], scale_factor=self.f_uf, mode='bilinear', align_corners=False)
        buffer_left = self.init_feature(x_left)
        buffer_right = self.init_feature(x_right)
        buffer_left = self.deep_feature(buffer_left)
        buffer_right = self.deep_feature(buffer_right)
        # if 'pam' in self.model:
        #     buffer_leftT, buffer_rightT = self.pam(buffer_left, buffer_right, catfea_left, catfea_right)
        # elif 'coswin' in self.model:
        # buffer_leftT, buffer_rightT = self.co_feature(buffer_left), self.co_feature(buffer_right)
        buffer_leftT, buffer_rightT = self.co_feature(buffer_left, buffer_right, d_left, d_right)                
        # elif 'seperate' in self.model or 'independent' in self.model:
        # buffer_leftT, buffer_rightT = self.swin(buffer_left), self.swin(buffer_right)
        buffer_leftF, buffer_rightF = self.fusion(torch.cat([buffer_left, buffer_leftT], dim=1)), self.fusion(torch.cat([buffer_right, buffer_rightT], dim=1))
        
        buffer_leftF = self.reconstruct(buffer_leftF)
        buffer_rightF = self.reconstruct(buffer_rightF)
        # if 'late_fusion' in self.model:
        #     buffer_leftF, buffer_rightF = self.swin(buffer_leftF), self.swin(buffer_rightF)
        #     buffer_leftF, buffer_rightF = self.coswin(buffer_leftF, buffer_rightF, d_left, d_right)
        out_left = self.upscale(buffer_leftF) + x_left_upscale
        out_right = self.upscale(buffer_rightF) + x_right_upscale
        # out_left = self.upscale(buffer_leftF)
        # out_right = self.upscale(buffer_rightF)
        return out_left, out_right
        # mod_h = -mod_pad_h * self.upscale_factor if (not 'pam' in self.model and mod_pad_h != 0) else None
        # mod_w = -mod_pad_w * self.upscale_factor if (not 'pam' in self.model and mod_pad_w != 0) else None
        # return out_left[..., :mod_h, :mod_w], out_right[..., :mod_h, :mod_w]

    def flop(self, H, W):
        N = H * W
        flop = 0
        flop += 2 * ((self.input_channel * 9 + 1) * N * 64) # adding 1 for bias
        flop += 2 * self.deep_feature.flops(H, W)
        flop += self.co_feature.flops(H, W)
        flop += 2 * (self.f_RDB.flop(N) + self.CAlayer.flop(N) + N * (128 + 1) * 64)
        flop += 2 * self.reconstruct.flops(H, W)
        flop += 2 * (N * (64 + 1) * 64 * (self.upscale_factor ** 2) + N * 3 * (64 * 9 + 1))
        flop += 2 * (N * (64 + 1) * 64 * (self.upscale_factor ** 2))
        return flop

    def remainder(self, x: int ,y: int):
        return x - (x//y) * y    

    def check_image_size(self, x):

        _, _, h, w = x.size()
        mod_pad_h = self.remainder(self.w_size - self.remainder(h, self.w_size), self.w_size)
        mod_pad_w = self.remainder(self.w_size - self.remainder(w, self.w_size), self.w_size)
        # mod_pad_w = (self.w_size - w % self.w_size) % self.w_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant')
        return x, mod_pad_h, mod_pad_w

    def _init_weights(self, m):
        if any(isinstance(m , mm) for mm in [nn.Conv2d, nn.Linear, nn.Conv1d, nn.LayerNorm, nn.BatchNorm2d]):
            nn.init.constant_(m.weight, 0.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


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
        self.RDB = nn.Sequential(*self.RDBs)
        self.conv = nn.Conv2d(G0*n_RDB, G0, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        buffer = x
        temp = []
        for rdb in self.RDB:
            buffer =rdb(buffer)
            temp.append(buffer)
        buffer_cat = torch.cat(temp, dim=1)
        out = self.conv(buffer_cat)
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

        for _, mdb in enumerate(self.MDBs):
            x = mdb(x)   

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
        # y = torch.mean(x, dim=(2, 3), keepdim=True)
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


def benchmark(model, x, y):
    exc_time = 0.0
    n_itr = 100
    left, right = None, None
    for _ in range(10):
        _, _ = model(x, y)
    for _ in range(n_itr):
        # with torch.cuda.amp.autocast():
        starter.record()
        left, right = model(x, y)
        ender.record()
        torch.cuda.synchronize()
        elps = starter.elapsed_time(ender)
        exc_time += elps
        # print('################## total: ', elps /
        #         1000, ' #######################')
    print('exec time: ', exc_time / n_itr / 1000)
    return left, right


if __name__ == "__main__":
    # from utils import disparity_alignment
    # from StreoSwinSR import CoSwinAttn
    # from SwinTransformer import SwinAttn
    B, H, W, C = 1, 176, 640, 7
    import onnx
    import torch_tensorrt
    import torch_tensorrt.logging as torchtrt_logging
    torchtrt_logging.set_reportable_log_level(torchtrt_logging.Level.Warning)


    net = Net(upscale_factor=4, model='MDB_coswin', img_size=tuple([H, W]), input_channel=C, w_size=8).cuda().eval()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
    print('   FLOPS: %.2fG' % (net.flop(H, W) / 1e9))
    x = torch.randn((B, C, H, W)).cuda()
    y = torch.randn((B, C, H, W)).cuda()
    input_names = ['x_left', 'x_right']
    output_names = ['sr_left', 'sr_right']
    with torch.no_grad():
        g_left , g_right = net(x, y)
        # torch.onnx.export(net, 
        #           (x, x),
        #           "MDB_coswin.onnx",
        #           verbose=False,
        #           input_names=input_names,
        #           output_names=output_names,
        #           export_params=True,opset_version=11
        #           )

        inputs = [torch_tensorrt.Input((B, C, H, W), dtype=torch.half), torch_tensorrt.Input((B, C, H, W), dtype=torch.half)]
        jit_model = torch.jit.script(net)
        trt_script_module = torch_tensorrt.compile(jit_model, inputs=inputs, enabled_precisions=[torch.half])
        # trt_script_module = torch_tensorrt.compile(jit_model, inputs=inputs)
        # benchmark(model_trt, x)
        benchmark(net, x, y)
        p_left, p_right = benchmark(trt_script_module, x.half(), y.half())
        print('error:', ((g_left - p_left).abs().mean().cpu().numpy() + (g_right - p_right).abs().mean().cpu().numpy())/2.0)
        # import pdb; pdb.set_trace()
        
