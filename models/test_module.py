import sys
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = nn.Sequential(*[nn.Conv2d(64, 64, 3, 1, 1, bias=True) for _ in range(30)])

    def forward(self, x):
        return self.layer(x)


if __name__ == "__main__":
    H, W, C = 360, 640, 3
    from torch2trt import torch2trt
    import torch_tensorrt

    net = Net().cuda()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    net.train(False)
    x = torch.clamp(torch.randn((1, 64, H, W)) , min=0.0).cuda()
    with torch.no_grad():
        model_trt = torch2trt(net, [x], max_batch_size=1)
        script_model = torch.jit.script(net)
        # benchmark(script_model, x)
        inputs = [torch_tensorrt.Input((1, 64, H, W), dtype=torch.float)]
        trt_script_module = torch_tensorrt.compile(script_model, inputs=inputs)
        exc_time = 0.0
        n_itr = 100
        for _ in range(10):
            # a = model_trt(x)
            b = net(x)
            # print('output error :', (((torch.abs(a - c)).mean() + (torch.abs(b - d)).mean() ).cpu().numpy() /2.0))
            # assert torch.equal(a, c) and torch.equal(b, d)
        for _ in range(n_itr):
            # with torch.cuda.amp.autocast():
            starter.record()
            _ = trt_script_module(x)
            # _ = model_trt(x)
            # _ = net(x)
            ender.record()
            torch.cuda.synchronize()
            elps = starter.elapsed_time(ender)
            exc_time += elps
            print('################## total: ', elps /
                    1000, ' #######################')
        print('exec time: ', exc_time / n_itr / 1000)
