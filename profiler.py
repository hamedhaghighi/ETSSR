import torch
import torch.nn as nn
import torch.nn.functional as F
from model_selection import model_selection





if __name__ == "__main__":
    # from utils import disparity_alignment
    # from StreoSwinSR import CoSwinAttn
    # from SwinTransformer import SwinAttn
    model_name = 'EDSR'
    H, W, C = 360, 640, 3
    upscale_factor = 4
    w_size = 15
    net = model_selection(model_name, upscale_factor, H, W, C, w_size)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    net.train(False)
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
    print('   FLOPS: %.2fG' % (net.flop(H, W) / 1e9))
    x = torch.clamp(torch.randn((1, 3, H, W)), min=0.0).cuda()
    exc_time = 0.0
    n_itr = 10
    with torch.no_grad():
        for _ in range(10):
            _, _ = net(x, x)
        for _ in range(n_itr):
            starter.record()
            _, _ = net(x, x)
            ender.record()
            torch.cuda.synchronize()
            elps = starter.elapsed_time(ender)
            exc_time += elps
            print('################## total: ', elps /
                  1000, ' #######################')

    print('exec time: ', exc_time / n_itr / 1000)
