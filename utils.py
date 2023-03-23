import torch


def check_disparity(LR_left, LR_right):
    b, c, h, w = LR_left.shape
    d_left = LR_left[0, 3]
    d_right = LR_right[0, 3]
    coords_h, coords_w = torch.meshgrid(
        [torch.arange(h), torch.arange(w)], indexing='ij')
    r2l_w = torch.clamp(coords_w.float() + 0.5 - d_left, min=0).long()
    l2r_w = torch.clamp(coords_w.float() + 0.5 + d_right, max=w - 1).long()
    selected_r = LR_right[0, :, coords_h, r2l_w]
    selected_l = LR_left[0, :, coords_h, l2r_w]
    import matplotlib.pyplot as plt
    plt.figure(0)
    plt.imshow(LR_left[0, :3].permute(1, 2, 0).numpy())
    plt.figure(1)
    plt.imshow(selected_r[:3].permute(1, 2, 0).numpy())
    plt.figure(2)
    plt.imshow(LR_right[0, :3].permute(1, 2, 0).numpy())
    plt.show()
    import pdb
    pdb.set_trace()


def select_patch(self, tensor, coords_h, coords_w, b_size):
    w_size = self.w_size
    tensor_padded = F.pad(
        tensor,
        (w_size // 2,
         w_size // 2,
         w_size // 2,
         w_size // 2))
    tensor_selected = torch.cat([tensor_padded[i: i + 1, :, coords_h[i], coords_w[i]]
                                for i in range(b_size)])  # B , C, H , W , wsize, wsize
    return tensor_selected


def check_input_size(input_resolution, w_size):
    H, W = input_resolution
    mod_pad_h = w_size - H % w_size
    mod_pad_w = w_size - W % w_size
    return tuple([H + mod_pad_h, W + mod_pad_w])


def disparity_alignment(d_left, d_right, b, h, w):
    coords_b, coords_h, coords_w = torch.meshgrid(
        [torch.arange(b), torch.arange(h), torch.arange(w)], indexing='ij')  # H, W
    # m_left = ((coords_w.to(self.device).float() + 0.5 - d_left ) >= 0).unsqueeze(1).float() # B , H , W
    # m_right = ((coords_w.to(self.device).float() + 0.5 + d_right.long()) <=
    # w - 1).unsqueeze(1).float() # B , H , W
    r2l_w = torch.clamp(
        coords_w.float() + 0.5 - d_left.cpu(),
        min=0).long() if d_left is not None else None
    l2r_w = torch.clamp(
        coords_w.float() + 0.5 + d_right.cpu(),
        max=w - 1).long() if d_right is not None else None
    return coords_b, coords_h, r2l_w, l2r_w
