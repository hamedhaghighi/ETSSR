import torch
import torch.nn as nn
import torch.nn.functional as F
from extra.tritanlayernorm import LayerNorm
from models.model import Net
from model_zoo.PASSRnet import PASSRnet


if __name__ == "__main__":
    # from utils import disparity_alignment
    # from StreoSwinSR import CoSwinAttn
    # from SwinTransformer import SwinAttn
    model_name = 'PASSRnet'
    H, W, C = 360, 640, 3
    upscale_factor = 4
    if model_name == 'mine':
        net = Net(upscale_factor=4, model='MDB_coswin', img_size=tuple(
            [H, W]), input_channel=C, w_size=15).cuda()
    elif model_name == 'PASSRnet':
        net = PASSRnet(upscale_factor).cuda()
    
    starter, ender = torch.cuda.Event(
        enable_timing=True), torch.cuda.Event(enable_timing=True)
    net.train(False)
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
    print('   FLOPS: %.2fG' % (net.flop(H, W) / 1e9))
    x = torch.clamp(torch.randn((1, 3, H, W)), min=0.0).cuda()
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
