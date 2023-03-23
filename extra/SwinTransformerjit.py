# -----------------------------------------------------------------------------------
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
# -----------------------------------------------------------------------------------

from typing import Tuple

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size: int):

    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous(
    ).view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros(
            (2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the
        # window
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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask.numel() > 0:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

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


class SwinAttnBlock(nn.Module):

    def __init__(
            self,
            dim,
            input_resolution,
            num_heads,
            window_size=7,
            shift_size=0,
            mlp_ratio=4.,
            qkv_bias=True,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't
            # partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(
                self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_drop=drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = torch.empty(1)

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

    def forward(self, x, x_size: Tuple[int, int]):
        H, W = x_size[0], x_size[1]
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        # nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size)
        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        # if self.input_resolution == x_size:
        # nW*B, window_size*window_size, C
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        # else:
        # attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1,
                                         self.window_size, self.window_size, C)
        shifted_x = window_reverse(
            attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(
                self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = self.norm2(x)
        x = x + self.drop_path(self.mlp(x))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class RSTB(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):
        super(RSTB, self).__init__()

        self.use_checkpoint = use_checkpoint
        self.dim = dim
        self.input_resolution = input_resolution
        self.blocks = nn.ModuleList([
            SwinAttnBlock(dim=dim, input_resolution=input_resolution,
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

    def forward(self, x, x_size: Tuple[int, int]):
        out = x
        for blk in self.blocks:
            # if self.use_checkpoint:
            #     x = checkpoint.checkpoint(blk, x, x_size)
            # else:
            x = blk(x, x_size)
        x = x.transpose(1, 2).view(-1, self.dim, x_size[0], x_size[1])
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        return x + out

    def flops(self):
        flops = 0
        for block in self.blocks:
            flops += block.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9

        return flops


class SwinAttn(nn.Module):

    def __init__(
            self,
            img_size=64,
            in_chans=3,
            embed_dim=96,
            depths=[
                6,
                6,
                6,
                6],
            num_heads=[
                6,
                6,
                6,
                6],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
            use_checkpoint=False):
        super(SwinAttn, self).__init__()
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        self.window_size = window_size

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        patches_resolution = [img_size[0], img_size[1]]
        self.patches_resolution = patches_resolution
        self.pre_norm = norm_layer(embed_dim)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(
                0,
                drop_path_rate,
                sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
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
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

    def forward(self, x):

        x_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        x = self.pre_norm(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)  # B L C
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, C, x_size[0], x_size[1])
        return x

    def flops(self):
        flops = 0
        for layer in self.layers:
            flops += layer.flops()
        H, W = self.patches_resolution
        flops += 2 * H * W * self.embed_dim
        return flops


class CoWindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros(
            (2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the
        # window
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
        kv = self.kv(x_selected).reshape(
            b,
            n,
            2,
            self.num_heads,
            c //
            self.num_heads).permute(
            2,
            0,
            3,
            1,
            4)
        q, k, v = q[0], kv[0], kv[1]  # b , nh, n, c//nh

        # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        attn = attn + relative_position_bias.unsqueeze(0)

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

    def __init__(
            self,
            dim,
            input_resolution,
            num_heads,
            window_size=7,
            shift_size=0,
            mlp_ratio=4.,
            qkv_bias=True,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't
            # partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = CoWindowAttention(
            dim,
            window_size=to_2tuple(
                self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_drop=drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

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

        coords_b, coords_h, coords_w = torch.meshgrid(
            [torch.arange(b), torch.arange(h), torch.arange(w)], indexing='ij')  # H, W
        # m_left = ((coords_w.to(self.device).float() + 0.5 - d_left ) >= 0).unsqueeze(1).float() # B , H , W
        # m_right = ((coords_w.to(self.device).float() + 0.5 + d_right.long())
        # <= w - 1).unsqueeze(1).float() # B , H , W
        r2l_w = torch.clamp(coords_w.float() + 0.5 -
                            d_left.cpu(), min=0).long()
        l2r_w = torch.clamp(coords_w.float() + 0.5 +
                            d_right.cpu(), max=w - 1).long()
        # assert L == H * W, "input feature has wrong size"
        shortcut_left, shortcut_right = x_left, x_right
        x_left, x_right = self.norm1(x_left), self.norm1(x_right)
        x_left, x_right = x_left.view(b, h, w, c), x_right.view(b, h, w, c)

        x_left_selected = x_left[coords_b, coords_h, l2r_w].clone()
        x_right_selected = x_right[coords_b, coords_h, r2l_w].clone()
        # cyclic shift
        if self.shift_size > 0:
            shifted_x_left = torch.roll(
                x_left, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_x_right = torch.roll(
                x_right, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_x_left_selected = torch.roll(
                x_left_selected, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_x_right_selected = torch.roll(
                x_right_selected, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x_left = x_left
            shifted_x_right = x_right
            shifted_x_left_selected = x_left_selected
            shifted_x_right_selected = x_right_selected

        # partition windows
        # nW*B, window_size, window_size, C
        x_left_windows = window_partition(shifted_x_left, self.window_size)
        x_right_windows = window_partition(shifted_x_right, self.window_size)
        x_left_selected_windows = window_partition(
            shifted_x_left_selected, self.window_size)
        x_right_selected_windows = window_partition(
            shifted_x_right_selected, self.window_size)
        # nW*B, window_size*window_size, C
        x_left_windows = x_left_windows.view(-1,
                                             self.window_size * self.window_size, c)
        x_right_windows = x_right_windows.view(-1,
                                               self.window_size * self.window_size, c)
        x_left_selected_windows = x_left_selected_windows.view(
            -1, self.window_size * self.window_size, c)
        x_right_selected_windows = x_right_selected_windows.view(
            -1, self.window_size * self.window_size, c)

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are
        # the multiple of window size
        attn_mask = self.attn_mask if self.input_resolution == x_size else self.calculate_mask(
            x_size).to(x_left.device)
        x_leftT, attn_r2l = self.attn(
            x_left_windows, x_right_selected_windows, mask=attn_mask)
        x_rightT, attn_l2r = self.attn(
            x_right_windows, x_left_selected_windows, mask=attn_mask)

        # masks
        ww = self.window_size * self.window_size
        Mr2l_relaxed = self.M_Relax(attn_r2l, num_pixels=2)
        Ml2r_relaxed = self.M_Relax(attn_l2r, num_pixels=2)
        m_left = Mr2l_relaxed.reshape(-1,
                                      1,
                                      ww) @ attn_l2r.permute(0,
                                                             1,
                                                             3,
                                                             2).reshape(-1,
                                                                        ww,
                                                                        1)
        m_right = Ml2r_relaxed.reshape(-1,
                                       1,
                                       ww) @ attn_r2l.permute(0,
                                                              1,
                                                              3,
                                                              2).reshape(-1,
                                                                         ww,
                                                                         1)
        m_left = m_left.squeeze().reshape(-1, self.num_heads, ww,
                                          1).permute(0, 2, 1, 3).detach()  # b nh
        m_right = m_right.squeeze().reshape(-1, self.num_heads, ww,
                                            1).permute(0, 2, 1, 3).detach()
        m_left = torch.tanh(5 * m_left)
        m_right = torch.tanh(5 * m_right)

        def matmul_mask(x, m):
            return (x.reshape(x.shape[0], ww, self.num_heads, c //
                    self.num_heads) * m).reshape(x.shape[0], ww, c)

        out_left = matmul_mask(x_left_windows, (1 - m_left)
                               ) + matmul_mask(x_leftT, m_left)
        out_right = matmul_mask(
            x_right_windows, (1 - m_right)) + matmul_mask(x_rightT, m_right)

        # merge windows
        out_left = out_left.view(-1, self.window_size, self.window_size, c)
        out_right = out_right.view(-1, self.window_size, self.window_size, c)
        shifted_x_left = window_reverse(
            out_left, self.window_size, h, w)  # B H' W' C
        shifted_x_right = window_reverse(
            out_right, self.window_size, h, w)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x_left = torch.roll(shifted_x_left, shifts=(
                self.shift_size, self.shift_size), dims=(1, 2))
            x_right = torch.roll(shifted_x_right, shifts=(
                self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_left = shifted_x_left
            x_right = shifted_x_right

        x_left = x_left.view(b, n, c)
        x_right = x_right.view(b, n, c)

        # FFN
        x_left = shortcut_left + self.drop_path(x_left)
        x_right = shortcut_right + self.drop_path(x_right)
        x_left = x_left + self.drop_path(self.mlp(self.norm2(x_left)))
        x_right = x_right + self.drop_path(self.mlp(self.norm2(x_right)))

        return x_left, x_right

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def M_Relax(self, M, num_pixels):
        M_list = []
        M_list.append(M.unsqueeze(1))
        for i in range(num_pixels):
            pad = nn.ZeroPad2d(padding=(0, 0, i + 1, 0))
            pad_M = pad(M[:, :, :-1 - i, :])
            M_list.append(pad_M.unsqueeze(1))
        for i in range(num_pixels):
            pad = nn.ZeroPad2d(padding=(0, 0, 0, i + 1))
            pad_M = pad(M[:, :, i + 1:, :])
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


if __name__ == '__main__':
    upscale = 2
    window_size = 8
    height = (60 // upscale // window_size + 1) * window_size
    width = (180 // upscale // window_size + 1) * window_size
    model = SwinAttn(
        upscale=2, img_size=(
            height, width), window_size=window_size, depths=[
            6, 6, 6, 6], embed_dim=60, num_heads=[
                6, 6, 6, 6], mlp_ratio=2)
    print('Input Height:', height, 'Width: ', width)
    # print ('FLOPS: ', model.flops() / 1e9)
    x = torch.randn((1, 60, height, width))
    x = model(x)
    print(x.shape)
