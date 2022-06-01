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
        # x_left = self.pre_norm(x_left)
        # x_right = self.pre_norm(x_right)
        for dlayer, mlayer in zip(self.disjoint_layers, self.merge_layers):
            x_left, x_right = dlayer(x_left, x_size), dlayer(x_right, x_size)
            x_left, x_right = mlayer(x_left, x_right, d_left, d_right, x_size)

        x_left = self.norm(x_left)  # B L C
        x_right = self.norm(x_right)
        B, HW, C = x_left.shape
        # x_left = x_left.transpose(1, 2).view(B, C, x_size[0], x_size[1])
        # x_right = x_right.transpose(1, 2).view(B, C, x_size[0], x_size[1])
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
