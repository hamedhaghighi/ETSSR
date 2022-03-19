import torch
import torch.nn as nn
import torch.nn.functional as F
from extra.tritanlayernorm import LayerNorm
starter, ender = torch.cuda.Event(
    enable_timing=True), torch.cuda.Event(enable_timing=True)

cv = nn.Conv2d(64, 64, 1, 1).cuda()
x = torch.randn(1 , 64, 32 , 96).cuda()
with torch.no_grad():
    for _ in range(100):
        starter.record()
        _ = cv(x)
        ender.record()
        torch.cuda.synchronize()
        print('conv time:', starter.elapsed_time(ender) / 1000)

    for _ in range(100):
        starter.record()
        q = x.view(1, 64, 4, 8, 12, 8).permute(0, 2, 4, 3, 5, 1).contiguous().view(1*48, 64, 64)
        v = x.view(1, 64, 4, 8, 12, 8).permute(0, 2, 4, 3, 5, 1).contiguous().view(1*48, 64, 64)
        ender.record()
        torch.cuda.synchronize()
        print('matmul time:', starter.elapsed_time(ender) / 1000)
        attn = q @ v

    w_shape = (64)
    weight = torch.rand(w_shape, dtype=torch.float, device='cuda', requires_grad=True)
    bias = torch.rand(w_shape, dtype=torch.float, device='cuda', requires_grad=True)
    layer_norm = LayerNorm.apply
    # forward pass
    x = torch.randn(48 , 64, 64).cuda()
    norm = nn.LayerNorm(64).cuda()
    for _ in range(100):
        starter.record()
        # _ = bnorm(x)
        y_tri = layer_norm(x, w_shape, weight, bias, 1e-5)
        ender.record()
        torch.cuda.synchronize()
        print('norm time:', starter.elapsed_time(ender) / 1000)

