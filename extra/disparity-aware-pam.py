class PAM(nn.Module):
    def __init__(self, channels, w_size=8, device='cpu'):
        super(PAM, self).__init__()
        self.device = device
        self.w_size = w_size
        self.channel = channels
        self.bq = nn.Conv2d(4*channels, channels, 1, 1, 0, groups=4, bias=True)
        self.bs = nn.Conv2d(4*channels, channels, 1, 1, 0, groups=4, bias=True)
        self.softmax = nn.Softmax(-1)
        self.rb = ResB(4 * channels)
        self.bn = nn.BatchNorm2d(4 * channels)
        self.window_centre_table = None

    def patchify(self, tensor, b, c, h, w):
        w_size = self.w_size
        # B C H//w_size W//wsize wsize wsize
        return tensor.reshape(b, c, h // w_size, w_size, w // w_size, w_size).permute(0, 2, 4, 3, 5, 1)

    def unpatchify(self, tensor, b, c, h, w):
        w_size = self.w_size
        return tensor.reshape(b, h // w_size, w // w_size, w_size, w_size, c).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

    def __call__(self, x_left, x_right, catfea_left, catfea_right, d_left, d_right):
        # Building matching indexes and patch around that using disparity
        w_size = self.w_size
        b, c, h, w = x_left.shape
        coords_b, coords_h, coords_w = torch.meshgrid(
            [torch.arange(b), torch.arange(h), torch.arange(w)], indexing='ij')  # H, W
        # m_left = ((coords_w.to(self.device).float() + 0.5 - d_left ) >= 0).unsqueeze(1).float() # B , H , W
        # m_right = ((coords_w.to(self.device).float() + 0.5 + d_right.long()) <= w - 1).unsqueeze(1).float() # B , H , W
        r2l_w = torch.clamp(coords_w.float() + 0.5 -
                            d_left.cpu(), min=0).long()
        l2r_w = torch.clamp(coords_w.float() + 0.5 +
                            d_right.cpu(), max=w - 1).long()

        Q = self.bq(self.rb(self.bn(catfea_left)))  # B C H W
        # Q = Q - torch.mean(Q, 3).unsqueeze(3).repeat(1, 1, 1, w)
        K = self.bs(self.rb(self.bn(catfea_right)))  # B C H W
        # K = K - torch.mean(K, 3).unsqueeze(3).repeat(1, 1, 1, w)
        # B , C , W// , H//, w_size w_size

        Q_selected = self.patchify(Q[coords_b, :, coords_h, l2r_w], b, c, h, w)
        # B , C , W//wsize , H//wsize, w_size, w_size
        K_selected = self.patchify(K[coords_b, :, coords_h, r2l_w], b, c, h, w)
        Q, K = self.patchify(Q, b, c, h, w), self.patchify(K, b, c, h, w)
        Q, K = Q - Q.mean((4, 5))[..., None, None], K - \
            K.mean((4, 5))[..., None, None]
        score_r2l = Q.reshape(-1, w_size * w_size, c) @ K_selected.permute(0,
                                                                           1, 2, 5, 3, 4).reshape(-1, c, w_size * w_size)
        score_l2r = K.reshape(-1, w_size * w_size, c) @ Q_selected.permute(0,
                                                                           1, 2, 5, 3, 4).reshape(-1, c, w_size * w_size)
        # (B*H) * Wl * Wr
        # B*C*H//*W//, wsize * wsize, wsize * wsize
        Mr2l = self.softmax(score_r2l)
        Ml2r = self.softmax(score_l2r)
        ## masks
        Mr2l_relaxed = M_Relax(Mr2l, num_pixels=2)
        Ml2r_relaxed = M_Relax(Ml2r, num_pixels=2)
        V_left = Mr2l_relaxed.reshape(-1, 1, w_size*w_size) @ Ml2r.permute(
            0, 2, 1).reshape(-1, w_size*w_size, 1)
        V_left = self.unpatchify(
            V_left.squeeze().reshape(-1, w_size, w_size, 1), b, 1, h, w).detach()
        V_right = Ml2r_relaxed.reshape(-1, 1, w_size*w_size) @ Mr2l.permute(
            0, 2, 1).reshape(-1, w_size*w_size, 1)
        V_right = self.unpatchify(
            V_right.squeeze().reshape(-1, w_size, w_size, 1), b, 1, h, w).detach()
        V_left = torch.tanh(5 * V_left)
        V_right = torch.tanh(5 * V_right)

        x_right_selected = self.patchify(
            x_right[coords_b, :, coords_h, r2l_w], b, c, h, w)  # B, C, H//, W//, wsize, wsize
        x_left_selected = self.patchify(
            x_left[coords_b, :, coords_h, l2r_w], b, c, h, w)  # B, C, H, W, wsize, wsize
        x_leftT = Mr2l @ x_right_selected.reshape(-1, w_size * w_size, c)
        x_rightT = Ml2r @ x_left_selected.reshape(-1, w_size * w_size, c)
        x_leftT = self.unpatchify(x_leftT, b, c, h, w)  # B, C, H , W
        x_rightT = self.unpatchify(x_rightT, b, c, h, w)  # B, C, H , W
        out_left = x_left * (1 - V_left) + x_leftT * V_left
        out_right = x_right * (1 - V_right) + x_rightT * V_right
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
