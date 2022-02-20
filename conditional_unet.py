import math
import torch.nn as nn
import torch
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim, number_of_diffusions, n_channels=1, dim_embed=64, dim_latent=128):
        super(TimeEmbedding, self).__init__()

        self.number_of_diffusions = number_of_diffusions
        self.n_channels = n_channels
        self.fc1 = nn.Linear(dim_embed, dim_latent)
        self.fc2 = nn.Linear(dim_latent, dim)
        self.conv1 = nn.Conv1d(1, n_channels, 1)
        self.conv_out = nn.Conv1d(n_channels, n_channels, 1)
        self.dim_embed = dim_embed
        self.embeddings = nn.Parameter(self.embed_table())
        self.embeddings.requires_grad = False

    def embed_table(self):
        t = torch.arange(self.number_of_diffusions) + 1 
        half_dim = self.dim_embed // 2
        emb = 10.0 / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def forward(self, t):
        emb = self.embeddings[t, :]
        out = self.fc1(emb)
        out = F.mish(out)
        out = self.fc2(out)
        if self.n_channels > 1:
            out = out.unsqueeze(1)
        x = F.mish(self.conv1(F.mish(out)))
        out = out.repeat(1, self.n_channels, 1)
        out = self.conv_out(x + out)
        return out


# class ECGCondition(nn.Module):
#     def __init__(self, resolution, n_channels=1):
#         super(ECGCondition, self).__init__()    

#     def forward(self, x):
        


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
    def __init__(self, n_inputs, n_outputs, dim, number_of_diffusions,
                 kernel_size=5, n_heads=None, hidden_dim=None):
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
            self.attention = SelfAttention(dim, n_shortcut, hidden_dim=hidden_dim, num_heads=n_heads)
        
        self.time_emb = TimeEmbedding(dim, number_of_diffusions, n_inputs)

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
        out = (out + self.res_conv(initial_x))# / math.sqrt(2.0)
        return out


class DownsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, dim, number_of_diffusions, 
                 kernel_size=5, n_heads=None, hidden_dim=None):
        super(DownsamplingBlock, self).__init__()
        self.down = nn.Conv1d(n_outputs, n_outputs, 3, 2, padding=1)#nn.MaxPool1d(2)
        self.block = Block(n_inputs, n_outputs, dim, number_of_diffusions, kernel_size=kernel_size, n_heads=n_heads, hidden_dim=hidden_dim)

    def forward(self, x, t, condition):
        h = self.block(x, t, condition)
        # DOWNSAMPLING
        out = self.down(h)
        return h, out


class UpsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, dim, number_of_diffusions,
                 kernel_size=5, up_dim=None, n_heads=None, hidden_dim=None):
        super(UpsamplingBlock, self).__init__()
        self.block = Block(n_inputs, n_outputs, 2*dim, number_of_diffusions, kernel_size=kernel_size, n_heads=n_heads, hidden_dim=hidden_dim)

        if up_dim is None:
            self.up = nn.ConvTranspose1d(n_inputs // 2, n_inputs // 2, kernel_size=4, stride=2, padding=1)#padding=1
        else:
            self.up = nn.ConvTranspose1d(up_dim, up_dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x, h, t, condition):
        x = self.up(x) 
        if h is not None:
            x = torch.cat([x, h], dim=1)
        out = self.block(x, t, condition)
        return out


class BottleneckNet(nn.Module):
    def __init__(self, resolution, n_channels, number_of_diffusions,
                 kernel_size=3, n_heads=None, hidden_dim=None):
        super(BottleneckNet, self).__init__()
        self.time_emb = TimeEmbedding(resolution, number_of_diffusions, n_channels)        
        self.bottleneck_conv1 = nn.Conv1d(n_channels, n_channels , kernel_size=kernel_size, padding="same")
        self.bottleneck_conv1_2 = nn.Conv1d(n_channels, n_channels , kernel_size=kernel_size, padding="same")
        self.bottleneck_conv2 = nn.Conv1d(n_channels, n_channels, kernel_size=kernel_size, padding="same")
        self.attention_block = SelfAttention(resolution, n_channels)
        self.bottleneck_layer_norm1 = nn.LayerNorm(resolution)
        self.bottleneck_layer_norm2 = nn.LayerNorm(resolution)

        if n_heads is None and hidden_dim is None:
            self.attention = SelfAttention(resolution, n_channels)
        elif n_heads is None and hidden_dim:
            self.attention = SelfAttention(resolution, n_channels, hidden_dim=hidden_dim)
        elif n_heads and hidden_dim is None:
            self.attention = SelfAttention(resolution, n_channels, num_heads=n_heads)
        elif n_heads and hidden_dim:
            self.attention = SelfAttention(resolution, n_channels, num_heads=n_heads, hidden_dim=hidden_dim)

    def forward(self, x, t, condition):
        out = x
        tt = self.time_emb(t)#[:,None,:].repeat(1, out.shape[1], 1)
        out = out + tt

        out = self.bottleneck_conv1(out)
        out = self.bottleneck_layer_norm1(out)
        out = F.mish(out)
        out = self.bottleneck_conv1_2(out)
        out = self.attention_block(out)

        #contional_emb = self.cond_emb(condition)
        #out = out + contional_emb

        out = self.bottleneck_conv2(out)
        out = self.bottleneck_layer_norm2(out)
        out = F.mish(out)
        
        out = (x + out) #/ math.sqrt(2)
        return out


class ECGconditional(nn.Module):
    def __init__(self, number_of_diffusions, resolution=512, kernel_size=3, num_levels=4, n_channels=12):
        super(ECGconditional, self).__init__()

        self.num_levels = num_levels
        self.kernel_size = kernel_size
        self.number_of_diffusions = number_of_diffusions
        input_resolution_list = []
        input_channels_list = []
        output_channels_list = []
        n_heads_list = [8, 8, 4, 4, 4, 4, 8, 8]
        n_hidden_state_list = [16, 16, 16, 32, 32, 16, 16, 16]
        
        kernel_size_list = 2*(num_levels - 1) * [kernel_size]
        kernel_size_list[0] = kernel_size + 2
        kernel_size_list[1] = kernel_size + 2
        kernel_size_list[-1] = kernel_size + 2
        kernel_size_list[-2] = kernel_size + 2
        
        for i in range(num_levels - 1):
            input_channels_list.append(n_channels * 2**i)  
        for i in range(num_levels - 1):    
            x = 2 * input_channels_list[num_levels - i - 2]
            input_channels_list.append(x)

        for i in range(num_levels):
            input_resolution_list.append(resolution >> i)
        input_resolution_list = input_resolution_list + list(reversed(input_resolution_list[1:-1])) 
                
        for i in range(num_levels - 1):
            output_channels_list.append(2 * input_channels_list[i]) 
        for i in range(num_levels - 1):    
            x = output_channels_list[num_levels - i - 2] // 2 
            output_channels_list.append(x)

        for i in range(num_levels - 2):
            k = 2*(num_levels - 1) - i - 1
            input_channels_list[k] += output_channels_list[i]

        # Only odd filter kernels allowed
        assert(kernel_size % 2 == 1)
        self.downsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()

        for i in range(self.num_levels - 1):
            self.downsampling_blocks.append(
                DownsamplingBlock(n_inputs=input_channels_list[i], n_outputs=output_channels_list[i],
                                  dim=input_resolution_list[i], number_of_diffusions=number_of_diffusions, kernel_size=kernel_size,
                                  n_heads=n_heads_list[i], hidden_dim=n_hidden_state_list[i]))

        self.bottelneck = BottleneckNet(resolution=input_resolution_list[num_levels - 1], n_channels=input_channels_list[num_levels],
                                        number_of_diffusions=number_of_diffusions, n_heads=4, hidden_dim=32)

        i = self.num_levels - 1
        self.upsampling_blocks.append(
            UpsamplingBlock(n_inputs=input_channels_list[i], n_outputs=output_channels_list[i],
                                  dim=input_resolution_list[i], number_of_diffusions=number_of_diffusions,
                                  kernel_size=kernel_size, up_dim=input_channels_list[i],
                                  n_heads=n_heads_list[i], hidden_dim=n_hidden_state_list[i]))
        for i in range(self.num_levels, 2*self.num_levels - 2):
            self.upsampling_blocks.append(
                UpsamplingBlock(n_inputs=input_channels_list[i], n_outputs=output_channels_list[i],
                                  dim=input_resolution_list[i], number_of_diffusions=number_of_diffusions,
                                  kernel_size=kernel_size, n_heads=n_heads_list[i], hidden_dim=n_hidden_state_list[i]))


        self.output_conv = nn.Sequential(nn.Conv1d(output_channels_list[-1], n_channels, 3, padding="same"), nn.Mish(),
                                         nn.Conv1d(n_channels, n_channels, 1, padding="same"))

    def forward(self, x, t, condition=None):
        '''
        '''
        shortcuts = []
        out = x

        # DOWNSAMPLING BLOCKS
        for block in self.downsampling_blocks:
            h, out = block(out, t, None)
            shortcuts.append(h)
        del shortcuts[-1]
        #out = self.downsampling_blocks[-1](out)

        # BOTTLENECK CONVOLUTION
        out = self.bottelneck(out, t, None) 
        #print(out.shape)      

        # UPSAMPLING BLOCKS
        out = self.upsampling_blocks[0](out, None, t, None)
        #print(out.shape)
        for idx, block in enumerate(self.upsampling_blocks[1:]):
            out = block(out, shortcuts[-1-idx], t, None)
            #print(out.shape)

        # OUTPUT CONV
        out = self.output_conv(out)
        return out


# if __name__ == "__main__":
#     net = ECGconditional(1000, kernel_size=5, num_levels=5)
#     from ecg_data.ecg_dataset import ECGdataset
#     dataset = ECGdataset("/ayb/vol1/kruzhilov/datasets/ecg/Dx_164873001_401_PTBXL.npy")
#     from torch.utils.data import DataLoader
#     dataloader = DataLoader(dataset, batch_size=16)
#     for ecg in dataloader:
#         t = torch.randperm(1000 - 1)[:ecg.shape[0]] + 1
#         mu_estim = net(ecg, t)
#         break