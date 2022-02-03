import math
import torch.nn as nn
import torch
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim, n_channels=1, dim_latent=64):
        super(TimeEmbedding, self).__init__()
        self.n_channels = n_channels
        if 2*dim <= dim_latent:
            dim_latent = 2*dim
        else:
            dim_latent = dim_latent
        self.fc1 = nn.Linear(dim, dim_latent)
        self.fc2 = nn.Linear(dim_latent, dim)
        if n_channels > 1:
            self.out_conv = nn.Conv1d(1, n_channels, 1)
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = 10.0 / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        out = self.fc1(emb)
        out = F.mish(out)
        out = self.fc2(out)
        if self.n_channels > 1:
            out = self.out_conv(out)
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


class Block(nn.Module):
    def __init__(self, n_inputs, n_outputs, dim, kernel_size=5, n_heads=None, hidden_dim=None):
        super(Block, self).__init__()
        n_shortcut = int((n_inputs + n_outputs) // 2) + 1
        self.pre_shortcut_convs = nn.Conv1d(n_inputs, n_shortcut, kernel_size, padding="same")# padding="same"
        self.shortcut_convs = nn.Conv1d(n_shortcut, n_shortcut, 1, padding="same")#padding="same"
        self.post_shortcut_convs = nn.Conv1d(n_shortcut, n_outputs, kernel_size, padding="same")#, padding="same"
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
            self.attention = SelfAttention(dim, n_shortcut)
        self.time_emb = TimeEmbedding(2*dim, n_channels)

    def forward(self, x, t, condition):
        initial_x = x
        t = self.time_emb(t)#[:,None,:].repeat(1, x.shape[1], 1)
        x = x + t
        shortcut = self.pre_shortcut_convs(x)
        shortcut = self.layer_norm1(shortcut)
        shortcut = F.mish(shortcut)
        shortcut = self.shortcut_convs(shortcut)
        shortcut = self.attention(shortcut)

        #contional_emb = self.cond_emb(condition)
        #shortcut = shortcut + contional_emb

        out = self.post_shortcut_convs(shortcut)
        out = self.layer_norm2(out)
        out = F.mish(out)
        out = (out + self.res_conv(initial_x)) / math.sqrt(2.0)
        return out


class DownsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, dim, kernel_size=5, n_heads=None, hidden_dim=None):
        super(DownsamplingBlock, self).__init__()
        self.down = nn.Conv1d(n_outputs, n_outputs, 3, 2, padding=1)#nn.MaxPool1d(2)
        self.block = Block(n_inputs, n_outputs, dim, kernel_size=5, n_heads=n_heads, hidden_dim=hidden_dim)

    def forward(self, x, t, condition):
        h = self.block(x, t, condition)
        # DOWNSAMPLING
        out = self.down(h)
        return h, out


class UpsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, dim, kernel_size=5, up_dim=None, n_heads=None, hidden_dim=None):
        super(UpsamplingBlock, self).__init__()
        self.block = Block(n_inputs, n_outputs, dim, kernel_size=5, n_heads=n_heads, hidden_dim=hidden_dim)

        if up_dim is None:
            self.up = nn.ConvTranspose1d(n_inputs // 2, n_inputs // 2, kernel_size=4, stride=2, padding=1)#padding=1
        else:
            self.up = nn.ConvTranspose1d(up_dim, up_dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x, h, t, condition):
        x = self.up(x) 
        x = torch.cat([x, h], dim=1)
        out = self.block(x, t, condition)
        return out


class BottleneckNet(nn.Module):
    def __init__(self, resolution, n_channels, kernel_size=3, n_heads=None, hidden_dim=None):
        super(BottleneckNet, self).__init__()
        self.time_emb = TimeEmbedding(resolution)        
        self.bottleneck_conv1 = nn.Conv1d(n_channels, n_channels , kernel_size=kernel_size, padding="same")
        self.bottleneck_conv1_2 = nn.Conv1d(n_channels, n_channels , kernel_size=kernel_size, padding="same")
        self.bottleneck_conv2 = nn.Conv1d(n_channels, n_channels, kernel_size=kernel_size, padding="same")
        self.attention_block = SelfAttention(resolution, n_channels)
        self.bottleneck_layer_norm1 = nn.LayerNorm(resolution)
        self.bottleneck_layer_norm2 = nn.LayerNorm(resolution)

        if n_heads is None and hidden_dim is None:
            self.attention = SelfAttention(resolution, n_shortcut)
        elif n_heads is None and hidden_dim:
            self.attention = SelfAttention(resolution, n_shortcut, hidden_dim=hidden_dim)
        elif n_heads and hidden_dim is None:
            self.attention = SelfAttention(resolution, n_shortcut, num_heads=n_heads)
        elif n_heads and hidden_dim:
            self.attention = SelfAttention(resolution, n_shortcut)

        self.time_emb = TimeEmbedding(resolution, n_channels)

    def forward(self, x, t, condition):
        out = x
        tt = self.time_emb(t)#[:,None,:].repeat(1, out.shape[1], 1)
        out = out + tt
        out = self.bottleneck_conv1(out)
        out = self.bottleneck_layer_norm1(out)
        out = F.mish(out)
        out = self.bottleneck_conv1_2(out)
        self_attention = self.attention_block(out)
        out = self.bottleneck_conv2(self_attention)
        out = self.bottleneck_layer_norm2(out)
        out = F.mish(out)
        out = (x + out) / math.sqrt(2)
        return out


