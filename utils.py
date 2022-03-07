import torch

def check_disparity(HR_left, HR_right, LR_left, LR_right):
    b, c, h, w = LR_left.shape
    d_left = LR_left[0, 3]
    d_right = LR_right[0, 3]
    coords_h, coords_w = torch.meshgrid(
        [torch.arange(h), torch.arange(w)], indexing='ij')
    r2l_w = torch.clamp(coords_w - d_left.long(), min=0)
    l2r_w = torch.clamp(coords_w + d_right.long(), max=w - 1)
    selected_r = LR_right[0, :, coords_h, r2l_w]
    selected_l = LR_left[0, :, coords_h, l2r_w]
    import matplotlib.pyplot as plt
    plt.figure(0)
    plt.imshow(selected_l[:3].permute(1, 2, 0).numpy())
    plt.figure(1)
    plt.imshow(LR_left[0, :3].permute(1, 2, 0).numpy())
    plt.figure(2)
    plt.imshow(LR_right[0, :3].permute(1, 2, 0).numpy())
    plt.show()
    import pdb
    pdb.set_trace()
