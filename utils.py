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


def calculate_window_center_table(self, h, w):
        w_size = self.w_size
        table_h = torch.zeros(h + w_size, w + w_size,
                              self.w_size, self.w_size).long()
        table_w = torch.zeros(h + w_size, w + w_size,
                              self.w_size, self.w_size).long()
        coords_h, coords_w = torch.meshgrid(
            [torch.arange(h + w_size//2), torch.arange(w + w_size//2)], indexing='ij')
        coords_h, coords_w = coords_h, coords_w
        for i in range(w_size//2, h):
            for j in range(w_size//2, w):
                table_h[i, j] = coords_h[i - w_size//2: i +
                    w_size//2, j - w_size//2: j + w_size//2]
                table_w[i, j] = coords_w[i - w_size//2: i +
                    w_size//2, j - w_size//2: j + w_size//2]
        self.window_centre_table = (table_h, table_w)

def select_patch(self, tensor, coords_h, coords_w, b_size):
        w_size = self.w_size
        tensor_padded = F.pad(tensor, (w_size//2, w_size//2, w_size//2, w_size//2))
        tensor_selected = torch.cat([tensor_padded[i: i+1, :,  coords_h[i], coords_w[i]] for i in range(b_size)])  # B , C, H , W , wsize, wsize
        return tensor_selected
