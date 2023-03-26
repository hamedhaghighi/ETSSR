import torch

from model_selection import model_selection
from utils import check_input_size

if __name__ == "__main__":
    
    model_name = 'NAFSSR'
    H, W, C = 360, 640, 3
    input_size = (H, W)
    w_size = 15
    upscale_factor = 4
    # input_size = check_input_size(input_size, w_size)
    net = model_selection(
        model_name,
        upscale_factor,
        input_size[0],
        input_size[1],
        C,
        w_size)
    starter, ender = torch.cuda.Event(
        enable_timing=True), torch.cuda.Event(
        enable_timing=True)
    net.train(False)
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
    print('   FLOPS: %.2fG' % (net.flop(H, W) / 1e9))
