import torch.nn as nn
import torch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from utils import disparity_alignment
from models.SwinTransformer import RSTB, Mlp
import torch.utils.checkpoint as checkpoint


def window_partition(x, window_size):

    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous(
    ).view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class CoWindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros(
            (2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - \
            1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x_selected, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None

        """
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, 1, self.num_heads, c //
                              self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(x_selected).reshape(b, n, 2, self.num_heads,
                                         c//self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q[0], kv[0], kv[1]  # b , nh, n, c//nh

        # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        # attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(b // nW, nW, self.num_heads, n,
                             n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        xT = (attn @ v).transpose(1, 2).reshape(b, n, c)
        xT = self.proj(xT)
        xT = self.proj_drop(xT)
        return xT, attn

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class CoSwinAttnBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                    mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = CoWindowAttention(dim, window_size=to_2tuple(
            self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, proj_drop=drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.fuse = nn.Linear(2 * dim, dim)
        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # nW, window_size, window_size, 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x_left, x_right, d_left, d_right, x_size):
        h, w = x_size
        b, n, c = x_left.shape
        ww = self.window_size * self.window_size
        coords_b, coords_h, r2l_w, l2r_w = disparity_alignment(d_left, d_right, b, h, w)
        # assert L == H * W, "input feature has wrong size"
        shortcut_left , shortcut_right = x_left, x_right

        def norm_view_selection(x, w_ind):
            x = self.norm1(x)
            x = x.view(b, h, w, c)
            x_selected = x[coords_b, coords_h, w_ind].clone() if w_ind is not None else x
            return x, x_selected
            
        x_left, x_left_selected = norm_view_selection(x_left, l2r_w)
        x_right, x_right_selected = norm_view_selection(x_right, r2l_w)

        def shift_partition_view(x):
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) if self.shift_size > 0 else x
            x = window_partition(x, self.window_size).view(-1, self.window_size * self.window_size, c)
            return x
        x_left_windows , x_right_windows = shift_partition_view(x_left), shift_partition_view(x_right)
        x_left_selected_windows , x_right_selected_windows = shift_partition_view(x_left_selected), shift_partition_view(x_right_selected)
        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_mask = self.attn_mask if self.input_resolution == x_size else self.calculate_mask(x_size).to(x_left.device)
        x_leftT, attn_r2l = self.attn(x_left_windows, x_right_selected_windows, mask=attn_mask)
        x_rightT, attn_l2r = self.attn(x_right_windows, x_left_selected_windows, mask=attn_mask)

        def create_apply_mask(attn1, attn2):
            M_relaxed = self.M_Relax(attn1, num_pixels=2)
            msk = M_relaxed.reshape(-1, 1, ww) @ attn2.permute(0, 1, 3, 2).reshape(-1, ww, 1)
            msk = msk.squeeze().reshape(-1, self.num_heads, ww, 1).permute(0, 2, 1, 3).detach()  # b nh
            return torch.tanh(5 * msk)
            
        ## masks
        m_left= create_apply_mask(attn_r2l, attn_l2r)
        m_right = create_apply_mask(attn_l2r, attn_r2l)

        
        def matmul_mask(x, m):
            return (x.reshape(x.shape[0], ww, self.num_heads, c//self.num_heads) * m).reshape(x.shape[0], ww, c)

        # out_left = matmul_mask(x_left_windows, (1 - m_left)) + matmul_mask(x_leftT, m_left)
        # out_right = matmul_mask(x_right_windows, (1 - m_right)) + matmul_mask(x_rightT, m_right)
        out_left = self.fuse(torch.cat([x_left_windows, x_leftT], dim=-1))
        out_right = self.fuse(torch.cat([x_right_windows, x_rightT], dim=-1))

        def view_reverse_drop_mlp(x, shortcut):
            x = x.view(-1, self.window_size, self.window_size, c)
            x = window_reverse(x, self.window_size, h, w)  # B H' W' C
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)) if self.shift_size > 0 else x
            x = x.view(b, n, c)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

        x_left , x_right = view_reverse_drop_mlp(out_left, shortcut_left), view_reverse_drop_mlp(out_right, shortcut_right)

        return x_left, x_right

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


    def M_Relax(self, M, num_pixels):
        M_list = []
        M_list.append(M.unsqueeze(1))
        for i in range(num_pixels):
            pad = nn.ZeroPad2d(padding=(0, 0, i+1, 0))
            pad_M = pad(M[:, :, :-1-i, :])
            M_list.append(pad_M.unsqueeze(1))
        for i in range(num_pixels):
            pad = nn.ZeroPad2d(padding=(0, 0, 0, i+1))
            pad_M = pad(M[:, :, i+1:, :])
            M_list.append(pad_M.unsqueeze(1))
        M_relaxed = torch.sum(torch.cat(M_list, 1), dim=1)
        return M_relaxed

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += 2 * self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += 2 * nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 4 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += 2 * self.dim * H * W
        return flops


class CoRSTB(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):
        super(CoRSTB, self).__init__()

        self.use_checkpoint = use_checkpoint
        self.dim = dim
        self.input_resolution = input_resolution
        self.blocks = nn.ModuleList([
            CoSwinAttnBlock(dim=dim, input_resolution=input_resolution,
                            num_heads=num_heads, window_size=window_size,
                            shift_size=0 if (
                                i % 2 == 0) else window_size // 2,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            drop=drop,
                            drop_path=drop_path[i] if isinstance(
                                drop_path, list) else drop_path,
                            norm_layer=norm_layer)
            for i in range(depth)])

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x_left, x_right, d_left, d_right, x_size):
        out_left = x_left
        out_right = x_right
        for blk in self.blocks:
            if self.use_checkpoint:
                x_left, x_right = checkpoint.checkpoint(
                    blk, x_left, x_right, d_left, d_right, x_size)
            else:
                x_left, x_right = blk(x_left, x_right, d_left, d_right, x_size)
        x_left = x_left.transpose(
            1, 2).view(-1, self.dim, x_size[0], x_size[1])
        x_right = x_right.transpose(
            1, 2).view(-1, self.dim, x_size[0], x_size[1])
        x_left = self.conv(x_left)
        x_right = self.conv(x_right)
        x_left = x_left.flatten(2).transpose(1, 2)
        x_right = x_right.flatten(2).transpose(1, 2)
        return x_left + out_left, x_right + out_right

    def flops(self):
        flops = 0
        for block in self.blocks:
            flops += block.flops()
        H, W = self.input_resolution
        flops += 2 * H * W * self.dim * self.dim * 9

        return flops


class CoSwinAttn(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, resi_connection='1conv',
                 **kwargs):
        super(CoSwinAttn, self).__init__()
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        self.window_size = window_size

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        patches_resolution = [img_size[0], img_size[1]]
        self.patches_resolution = patches_resolution
        self.pre_norm = norm_layer(embed_dim)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = CoRSTB(dim=embed_dim,
                           input_resolution=(patches_resolution[0],
                                             patches_resolution[1]),
                           depth=depths[i_layer],
                           num_heads=num_heads[i_layer],
                           window_size=window_size,
                           mlp_ratio=self.mlp_ratio,
                           qkv_bias=qkv_bias,
                           drop_path=dpr[sum(depths[:i_layer]):sum(
                               depths[:i_layer + 1])],  # no impact on SR results
                           norm_layer=norm_layer,
                           use_checkpoint=use_checkpoint
                           )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

    def forward(self, x_left, x_right, d_left=None, d_right=None):

        x_size = (x_left.shape[2], x_left.shape[3])
        x_left = x_left.flatten(2).transpose(1, 2)
        x_right = x_right.flatten(2).transpose(1, 2)
        # x_left = self.pre_norm(x_left)
        # x_right = self.pre_norm(x_right)
        for layer in self.layers:
            x_left, x_right = layer(x_left, x_right, d_left, d_right, x_size)

        # x_left = self.norm(x_left)  # B L C
        # x_right = self.norm(x_right)
        B, HW, C = x_left.shape
        x_left = x_left.transpose(1, 2).view(B, C, x_size[0], x_size[1])
        x_right = x_right.transpose(1, 2).view(B, C, x_size[0], x_size[1])
        return x_left, x_right

    def flops(self):
        flops = 0
        for layer in self.layers:
            flops += layer.flops()
        H, W = self.patches_resolution
        flops += 4 * H * W * self.embed_dim
        return flops


class SwinAttnInterleaved(nn.Module):
    def __init__(self, img_size=64, in_chans=3,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False):
        super(SwinAttnInterleaved, self).__init__()
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        self.window_size = window_size

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        patches_resolution = [img_size[0], img_size[1]]
        self.patches_resolution = patches_resolution
        self.pre_norm = norm_layer(embed_dim)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.disjoint_layers = nn.ModuleList()
        self.merge_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias,
                         drop_path=dpr[sum(depths[:i_layer]):sum(
                             depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint
                         )
            self.disjoint_layers.append(layer)

        for i_layer in range(self.num_layers):
            layer = CoRSTB(dim=embed_dim,
                           input_resolution=(patches_resolution[0],
                                             patches_resolution[1]),
                           depth=depths[i_layer],
                           num_heads=num_heads[i_layer],
                           window_size=window_size,
                           mlp_ratio=self.mlp_ratio,
                           qkv_bias=qkv_bias,
                           drop_path=dpr[sum(depths[:i_layer]):sum(
                               depths[:i_layer + 1])],  # no impact on SR results
                           norm_layer=norm_layer,
                           use_checkpoint=use_checkpoint,
                           )
            self.merge_layers.append(layer)
        self.norm = norm_layer(self.num_features)

    def forward(self, x_left, x_right, d_left, d_right):

        x_size = (x_left.shape[2], x_left.shape[3])
        x_left = x_left.flatten(2).transpose(1, 2)
        x_right = x_right.flatten(2).transpose(1, 2)
        x_left = self.pre_norm(x_left)
        x_right = self.pre_norm(x_right)
        for dlayer, mlayer in zip(self.disjoint_layers, self.merge_layers):
            x_left, x_right = dlayer(x_left, x_size), dlayer(x_right, x_size)
            x_left, x_right = mlayer(x_left, x_right, d_left, d_right, x_size)

        x_left = self.norm(x_left)  # B L C
        x_right = self.norm(x_right)
        B, HW, C = x_left.shape
        x_left = x_left.transpose(1, 2).view(B, C, x_size[0], x_size[1])
        x_right = x_right.transpose(1, 2).view(B, C, x_size[0], x_size[1])
        return x_left, x_right

    def flops(self):
        flops = 0
        for layer in self.disjoint_layers:
            flops += 2 * layer.flops()
        for layer in self.merge_layers:
            flops += layer.flops()
        H, W = self.patches_resolution
        flops += 4 * H * W * self.embed_dim
        return flops
