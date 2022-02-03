import math
import torch.nn as nn
import torch
import torch.nn.functional as F


class DownsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, dim, kernel_size=5, n_heads=None, hidden_dim=None):
        n_shortcut = int((n_inputs + n_outputs) // 2) + 1
        super(DownsamplingBlock, self).__init__()
        self.kernel_size = kernel_size
        # CONV 1
        self.pre_shortcut_convs = nn.Conv1d(n_inputs, n_shortcut, self.kernel_size, padding="same")# padding="same"
        self.shortcut_convs = nn.Conv1d(n_shortcut, n_shortcut, 1, padding="same")#padding="same"
        self.post_shortcut_convs = nn.Conv1d(n_shortcut, n_outputs, self.kernel_size, padding="same")#, padding="same"
        self.down = nn.Conv1d(n_outputs, n_outputs, 3, 2, padding=1)#nn.MaxPool1d(2)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.res_conv = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else nn.Identity()
        if n_heads is None and hidden_dim is None:
            self.attention = SelfAttention(dim, n_shortcut)
        elif n_heads is None and hidden_dim:
            self.attention = SelfAttention(dim, n_shortcut, hidden_dim=hidden_dim)
        elif n_heads and hidden_dim is None:
            self.attention = SelfAttention(dim, n_shortcut, num_heads=n_heads)
        elif n_heads and hidden_dim:
            self.attention = SelfAttention(dim, n_shortcut, hidden_dim=hidden_dim, num_heads=n_heads)


    def forward(self, x):
        # PREPARING SHORTCUT FEATURES
        shortcut = self.pre_shortcut_convs(x)
        shortcut = self.layer_norm1(shortcut)
        shortcut = F.mish(shortcut)
        shortcut = self.shortcut_convs(shortcut)
        shortcut = self.attention(shortcut)
        # shortcut = torch.cat([h, shortcut], dim=1)
        # shortcut = self.shortcut_convs(shortcut)
        # shortcut = F.mish(shortcut)
        # PREPARING FOR DOWNSAMPLING
        shortcut = self.post_shortcut_convs(shortcut)
        shortcut = self.layer_norm2(shortcut)
        h = F.mish(shortcut)
        h = h + self.res_conv(x)
        # DOWNSAMPLING
        out = self.down(h)
        return h, out


class UpsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, dim, kernel_size=5, up_dim=None, n_heads=None, hidden_dim=None):
        n_shortcut = int((n_inputs + n_outputs) // 2)
        super(UpsamplingBlock, self).__init__()
        self.kernel_size = kernel_size

        # CONV 1
        self.pre_shortcut_convs = nn.Conv1d(n_inputs, n_shortcut, self.kernel_size, padding="same")#padding="same"
        self.shortcut_convs = nn.Conv1d(n_shortcut, n_shortcut, self.kernel_size, padding="same")
        self.post_shortcut_convs = nn.Conv1d(n_shortcut, n_outputs, self.kernel_size, padding="same")#padding="same"
        #self.up = nn.Upsample(scale_factor=2)
        if up_dim is None:
            self.up = nn.ConvTranspose1d(n_inputs // 2, n_inputs // 2, kernel_size=4, stride=2, padding=1)#padding=1
        else:
            self.up = nn.ConvTranspose1d(up_dim, up_dim, kernel_size=4, stride=2, padding=1)
        self.layer_norm1 = nn.LayerNorm(2*dim)
        self.layer_norm2 = nn.LayerNorm(2*dim)
        self.res_conv = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else nn.Identity()
        if n_heads is None and hidden_dim is None:
            self.attention = SelfAttention(2*dim, n_shortcut)
        elif n_heads is None and hidden_dim:
            self.attention = SelfAttention(2*dim, n_shortcut, hidden_dim=hidden_dim)
        elif n_heads and hidden_dim is None:
            self.attention = SelfAttention(2*dim, n_shortcut, num_heads=n_heads)
        elif n_heads and hidden_dim:
            self.attention = SelfAttention(2*dim, n_shortcut, hidden_dim=hidden_dim, num_heads=n_heads)

        self.time_emb = TimeEmbedding(2*dim)

    def forward(self, x, h, t):
        x = self.up(x) 
        initial_x = x
        t = self.time_emb(t)[:,None,:].repeat(1, x.shape[1], 1)
        x = x + t
        if h is None:
            h = x
        shortcut = torch.cat([x, h], dim=1)
        # PREPARING SHORTCUT FEATURES
        shortcut = self.pre_shortcut_convs(shortcut)
        shortcut = self.layer_norm1(shortcut)
        shortcut = F.mish(shortcut)
        shortcut = self.shortcut_convs(shortcut)
        shortcut = self.attention(shortcut)
        # shortcut = torch.cat([h, shortcut], dim=1)
        # PREPARING FOR DOWNSAMPLING
        out = self.post_shortcut_convs(shortcut)
        out = self.layer_norm2(out)
        out = F.mish(out)
        out = out + self.res_conv(torch.cat([initial_x, h], dim=1))
        return out


class SelfAttention(nn.Module):
    def __init__(self, vector_size, dim, hidden_dim=16, num_heads=4):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.to_qkv = nn.Conv1d(dim, 3*hidden_dim, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)
        self.attention = nn.MultiheadAttention(vector_size, num_heads=num_heads, batch_first=True)
        self.normalization = nn.LayerNorm(vector_size)

    def forward(self, x):
        x_norm = self.normalization(x)
        qkv = self.to_qkv(x_norm)
        q = qkv[:,:self.hidden_dim, :]
        k = qkv[:,self.hidden_dim:2*self.hidden_dim, :]
        v = qkv[:,2*self.hidden_dim:, :]
        h, _ = self.attention(q, k, v)
        h = self.to_out(h)
        h = h + x
        return h


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super(TimeEmbedding, self).__init__()
        if dim <= 32:
            dim_latent = 2*dim
        else:
            dim_latent = dim
        self.fc1 = nn.Linear(dim, dim_latent)
        self.fc2 = nn.Linear(dim_latent, dim)
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = 10.0 / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        #out = t.repeat(self.dim,1).transpose(1,0)#.unsqueeze(1)
        #out = out.to(dtype = torch.float)
        out = self.fc1(emb)
        out = F.mish(out)
        out = self.fc2(out)
        return out


class ECGunet(nn.Module):
    def __init__(self, resolution=256, kernel_size=3, num_levels=4):
        super(ECGunet, self).__init__()

        self.num_levels = num_levels
        self.kernel_size = kernel_size

        # Only odd filter kernels allowed
        assert(kernel_size % 2 == 1)
        self.downsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()

        self.downsampling_blocks.append(
            DownsamplingBlock(n_inputs=1, n_outputs=4, dim=resolution, kernel_size=kernel_size + 2))
        self.downsampling_blocks.append(
            DownsamplingBlock(n_inputs=4, n_outputs=4, dim=resolution // 2, kernel_size=kernel_size))
        for i in range(2, self.num_levels - 1):
             current_resolution = resolution >> i
             self.downsampling_blocks.append(
                DownsamplingBlock(n_inputs=2**i, n_outputs=2**(i+1), dim=current_resolution, kernel_size=kernel_size))

        self.upsampling_blocks.append(
            UpsamplingBlock(n_inputs=2**num_levels, n_outputs=2**(num_levels-2), dim=current_resolution // 2, kernel_size=kernel_size) )
        for i in reversed(range(1, self.num_levels - 2)):
             current_resolution = resolution >> (i + 1)
             self.upsampling_blocks.append(
                UpsamplingBlock(n_inputs=2**(i+2), n_outputs=2**i, dim=current_resolution, kernel_size=kernel_size))
        current_resolution = resolution // 2
        self.upsampling_blocks.append(
                UpsamplingBlock(n_inputs=6, n_outputs=4, dim=current_resolution, kernel_size=kernel_size, up_dim=2))

        self.time_emb = TimeEmbedding(resolution >> (self.num_levels - 1))
        self.bottleneck_conv1 = nn.Conv1d(2**(self.num_levels - 1), 2**(self.num_levels - 1), kernel_size=3, padding="same")
        self.bottleneck_conv1_2 = nn.Conv1d(2**(self.num_levels - 1), 2**(self.num_levels - 1), kernel_size=3, padding="same")
        self.bottleneck_conv2 = nn.Conv1d(2**(self.num_levels - 1), 2**(self.num_levels - 1), kernel_size=3, padding="same")
        self.attention_block = SelfAttention(resolution >> (self.num_levels - 1), 2**(self.num_levels - 1))
        self.bottleneck_layer_norm1 = nn.LayerNorm(resolution >> (self.num_levels - 1) )
        self.bottleneck_layer_norm2 = nn.LayerNorm(resolution >> (self.num_levels - 1) )
        self.output_conv = nn.Sequential(nn.Conv1d(4, 4, 3, padding="same"), nn.Mish(),
                                         nn.Conv1d(4, 1, 1, padding="same"))

    def forward(self, x, t):
        '''
        '''
        shortcuts = []
        out = x

        # DOWNSAMPLING BLOCKS
        for block in self.downsampling_blocks:
            h, out = block(out)
            shortcuts.append(h)
        del shortcuts[-1]
        #out = self.downsampling_blocks[-1](out)

        # BOTTLENECK CONVOLUTION
        old_out = out
        tt = self.time_emb(t)[:,None,:].repeat(1, out.shape[1], 1)
        out = out + tt
        out = self.bottleneck_conv1(out)
        out = self.bottleneck_layer_norm1(out)
        out = F.mish(out)
        out = self.bottleneck_conv1_2(out)
        self_attention = self.attention_block(out)
        out = self.bottleneck_conv2(self_attention)
        out = self.bottleneck_layer_norm2(out)
        out = F.mish(out) + old_out

        # UPSAMPLING BLOCKS
        out = self.upsampling_blocks[0](out, None, t)
        for idx, block in enumerate(self.upsampling_blocks[1:]):
            out = block(out, shortcuts[-1-idx], t)

        # OUTPUT CONV
        out = self.output_conv(out)
        return out


class ECGunetChannels(nn.Module):
    def __init__(self, resolution=512, kernel_size=3, num_levels=4, n_channels=12):
        super(ECGunetChannels, self).__init__()

        self.num_levels = num_levels
        self.kernel_size = kernel_size

        # Only odd filter kernels allowed
        assert(kernel_size % 2 == 1)
        self.downsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()

        self.downsampling_blocks.append(
            DownsamplingBlock(n_inputs=n_channels, n_outputs=2*n_channels, 
            dim=resolution, kernel_size=kernel_size + 2, n_heads=8, hidden_dim=8))
        self.downsampling_blocks.append(
            DownsamplingBlock(n_inputs=2*n_channels, n_outputs=4*n_channels,
            dim=resolution // 2, kernel_size=kernel_size))
        for i in range(2, self.num_levels - 1):
             current_resolution = resolution >> i
             self.downsampling_blocks.append(
                DownsamplingBlock(n_inputs=n_channels * 2**i, n_outputs=n_channels * 2**(i+1), dim=current_resolution, kernel_size=kernel_size))

        self.upsampling_blocks.append(
            UpsamplingBlock(n_inputs=n_channels * 2**num_levels, n_outputs=n_channels * 2**(num_levels-2), dim=current_resolution // 2, kernel_size=kernel_size) )
        for i in reversed(range(1, self.num_levels - 2)):
             current_resolution = resolution >> (i + 1)
             self.upsampling_blocks.append(
                UpsamplingBlock(n_inputs=n_channels * 2**(i+2), n_outputs=n_channels * 2**i, dim=current_resolution, kernel_size=kernel_size))
        current_resolution = resolution // 2
        self.upsampling_blocks.append(
                UpsamplingBlock(n_inputs=4 * n_channels, n_outputs=2 * n_channels,
                dim=current_resolution, kernel_size=kernel_size, n_heads=8))#, up_dim=2))

        self.time_emb = TimeEmbedding(resolution >> (self.num_levels - 1))
        
        self.bottleneck_conv1 = nn.Conv1d(n_channels * 2**(self.num_levels - 1), n_channels * 2**(self.num_levels - 1), kernel_size=3, padding="same")
        self.bottleneck_conv1_2 = nn.Conv1d(n_channels * 2**(self.num_levels - 1), n_channels * 2**(self.num_levels - 1), kernel_size=3, padding="same")
        self.bottleneck_conv2 = nn.Conv1d(n_channels * 2**(self.num_levels - 1), n_channels * 2**(self.num_levels - 1), kernel_size=3, padding="same")
        self.attention_block = SelfAttention(resolution >> (self.num_levels - 1), n_channels * 2**(self.num_levels - 1))
        self.bottleneck_layer_norm1 = nn.LayerNorm(resolution >> (self.num_levels - 1) )
        self.bottleneck_layer_norm2 = nn.LayerNorm(resolution >> (self.num_levels - 1) )
        
        self.output_conv = nn.Sequential(nn.Conv1d(2 * n_channels, n_channels, 3, padding="same"), nn.Mish(),
                                         nn.Conv1d(n_channels, n_channels, 1, padding="same"))

    def forward(self, x, t):
        '''
        '''
        shortcuts = []
        out = x

        # DOWNSAMPLING BLOCKS
        for block in self.downsampling_blocks:
            h, out = block(out)
            shortcuts.append(h)
        del shortcuts[-1]
        #out = self.downsampling_blocks[-1](out)

        # BOTTLENECK CONVOLUTION
        old_out = out
        tt = self.time_emb(t)[:,None,:].repeat(1, out.shape[1], 1)
        out = out + tt
        out = self.bottleneck_conv1(out)
        out = self.bottleneck_layer_norm1(out)
        out = F.mish(out)
        out = self.bottleneck_conv1_2(out)
        self_attention = self.attention_block(out)
        out = self.bottleneck_conv2(self_attention)
        out = self.bottleneck_layer_norm2(out)
        out = F.mish(out) + old_out / math.sqrt(2) #residiual connection normalization

        # UPSAMPLING BLOCKS
        out = self.upsampling_blocks[0](out, None, t)
        for idx, block in enumerate(self.upsampling_blocks[1:]):
            out = block(out, shortcuts[-1-idx], t)

        # OUTPUT CONV
        out = self.output_conv(out)
        return out


