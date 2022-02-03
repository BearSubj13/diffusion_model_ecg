import matplotlib.pyplot as plt 
import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
#from einops import rearrange
from torch.utils.data import DataLoader
import torch_optimizer
from torch.utils.tensorboard import SummaryWriter

from ecg_data.ecg_dataset import ECGdataset
from unet1d import ECGunet, ECGunetChannels


class GaussianDiffusion:
    def __init__(self, time_steps=1000, n_channels=12):
        super(GaussianDiffusion, self).__init__()
        self.time_steps = time_steps
        self.number_of_channels = n_channels
        #self.betta = torch.linspace(0.00001, 0.0001, time_steps)
        self.betta = self.cosine_beta_schedule(time_steps)
        self.alpha = 1 - self.betta
        self.alpha_cumprod = torch.cumprod(self.alpha, axis=0)
        self.alpha_cumprod_sqrt = torch.sqrt(self.alpha_cumprod)
        self.alpha_cumprod_sqrt_recip = 1 / torch.sqrt(self.alpha_cumprod)
        self.alpha_cumprod_sqrt_recip_minus_one = torch.sqrt(1 / self.alpha_cumprod - 1)
        self.one_minus_alpha_cumprod = 1 - self.alpha_cumprod 
        self.one_minus_alpha_cumprod_sqrt = torch.sqrt(self.one_minus_alpha_cumprod)

    def cosine_beta_schedule(self, timesteps, s = 0.008):
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = timesteps + 1
        x = torch.linspace(0, steps, steps)
        alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, min = 0.00001, max = 0.999)
        return betas

    def diffuse(self, x0, t, noise=None):
        """
        x0 - [batch_size*ecg_size=64] - original ecg
        t < time_steps [batch_size] - time spets for diffusion
        """
        assert x0.shape[0] == t.shape[0]
        if noise is None:
            noise = torch.normal(0, 1, size=[t.shape[0], self.number_of_channels, x0.shape[-1]])
        alpha_t = self.alpha_cumprod_sqrt[t]
        alpha_t = alpha_t.repeat(x0.shape[-1], self.number_of_channels, 1).transpose(0,2)
        #alpha_t = alpha_t.unsqueeze(1).repeat(1, self.number_of_channels, 1)
        part1 = alpha_t * x0
        one_minus_alpha_t = self.one_minus_alpha_cumprod_sqrt[t]
        one_minus_alpha_t = one_minus_alpha_t.repeat(x0.shape[-1], self.number_of_channels, 1).transpose(0,2)
        part2 = one_minus_alpha_t * noise
        xt = part1 + part2
        return xt, noise

    def mean_posterior(self, x0, t, xt=None):
        """
        x0 - points
        t - list of time, int
        """
        alpha_cum_prod_sqrt_t_1 = self.alpha_cumprod_sqrt[t - 1]
        alpha_sqrt_t = torch.sqrt(self.alpha[t])
        one_minus_alpha_cumprod_t_1 = self.one_minus_alpha_cumprod[t - 1]
        one_minus_alpha_cumprod_t = self.one_minus_alpha_cumprod[t]
       
        alpha_cum_prod_sqrt_t_1 = alpha_cum_prod_sqrt_t_1.repeat(x0.shape[-1], self.number_of_channels, 1).transpose(0,2)
        alpha_sqrt_t = alpha_sqrt_t.repeat(x0.shape[-1], self.number_of_channels, 1).transpose(0,2)
        one_minus_alpha_cumprod_t_1 = one_minus_alpha_cumprod_t_1.repeat(x0.shape[-1], self.number_of_channels, 1).transpose(0,2)
        one_minus_alpha_cumprod_t = one_minus_alpha_cumprod_t.repeat(x0.shape[-1], self.number_of_channels, 1).transpose(0,2)
        betta_t = self.betta[t].repeat(x0.shape[-1], self.number_of_channels, 1).transpose(0,2)

        if xt is None:
            xt, _ = self.diffuse(x0, t)
        part1 = alpha_cum_prod_sqrt_t_1 * betta_t * x0
        part2 = alpha_sqrt_t * one_minus_alpha_cumprod_t_1 * xt
        mu = (part1 + part2) / one_minus_alpha_cumprod_t
        return xt, mu

    def predict_start_from_noise(self, x_t, t, noise):
        coeff_xt = self.alpha_cumprod_sqrt_recip[t]
        coeff_xt = coeff_xt.repeat(x_t.shape[2], 1).transpose(0,1)
        coeff_xt = coeff_xt.to(x_t.device)
        coeff_xt = coeff_xt.unsqueeze(1)
        coeff_noise = self.alpha_cumprod_sqrt_recip_minus_one[t]
        coeff_noise = coeff_noise.repeat(x_t.shape[2], 1).transpose(0,1)
        coeff_noise = coeff_noise.to(x_t.device)
        coeff_noise = coeff_noise.unsqueeze(1)
        x0 = coeff_xt * x_t - coeff_noise * noise
        return x0

    def backward_pass(self, input):
        k_steps = 1200
        #for i in range(k_steps):
        xi = torch.randn_like(input)
        #xi = self.diffuse(x0, t)
        for i in reversed(range(1, k_steps)):
            t = i*torch.ones(input.shape[0], dtype=torch.long)
            _, mu = self.mean_posterior(input, t, xi)
            delta = mu - xi
            if i % 200 == 0:
                print(i, delta.abs().mean())
            xi = mu
        plt.plot(xi[0,:])

    def generation_from_net(self, net, number_of_points, device, dim=512):
        net.eval()
        xi = torch.randn(number_of_points, self.number_of_channels, dim)
        xi = xi.to(device)
        for i in reversed(range(1, self.time_steps)):
            t = i*torch.ones(number_of_points, dtype=torch.long)
            t = t.to(device)
            with torch.no_grad():
                noise = torch.sqrt(self.betta[i]) * torch.randn(number_of_points, self.number_of_channels, dim)
                noise = noise.to(device)
                delta_mu = net(xi, t)
                xi = xi + delta_mu + noise
                #xi = xi.clamp(min=-3.0, max=3.0)
        xi = xi.cpu()
        return xi

    def generation_from_net_epsilon(self, net, number_of_points, dim=256):
        net.eval()
        xi = torch.randn(number_of_points, dim).unsqueeze(1)
        xi = xi.to(device)
        for i in reversed(range(1, self.time_steps)):
            t = i*torch.ones(number_of_points, dtype=torch.long)
            t = t.to(device)
            with torch.no_grad():
                noise = torch.sqrt(self.betta[i]) * torch.randn(number_of_points, dim).unsqueeze(1)
                noise = noise.to(device)
                epsilon = net(xi, t)
                x0 = self.predict_start_from_noise(xi, t, epsilon)
                _, mu = self.mean_posterior(x0.cpu().squeeze(1), t.cpu(), xi.cpu().squeeze(1))
                mu = mu.to(device).unsqueeze(1)
                # coeff = self.betta[t] / self.one_minus_alpha_cumprod_sqrt[t]
                # coeff = coeff.repeat(dim, 1).transpose(0,1).unsqueeze(1)
                # coeff = coeff.to(device)
                # epsilon = coeff * epsilon
                # coeff = 1 / torch.sqrt(self.alpha[t])
                # coeff = coeff.repeat(dim, 1).transpose(0,1).unsqueeze(1)
                # coeff = coeff.to(device)
                #xi = coeff * (xi - epsilon) + noise
                xi = mu + noise
                xi = xi.clamp(min=-3.0, max=3.0)
        xi = xi.cpu()
        return xi


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# class MuNet(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         time_embd_dim = 100
#         # self.time_mlp = nn.Sequential(SinusoidalPosEmb(time_embd_dim), nn.Linear(time_embd_dim, time_embd_dim * 2),
#         #                               Mish(), nn.Linear(time_embd_dim * 2, time_embd_dim)
#         #                              )
#         self.time_mlp = nn.Sequential(nn.Linear(time_embd_dim, time_embd_dim * 2),
#                                       Mish(), nn.Linear(time_embd_dim * 2, time_embd_dim)
#                                      )
#         self.conv_block = nn.Sequential(nn.Conv1d(1,1,kernel_size=3,  padding="same"), Mish(),
#                                         nn.Conv1d(1,1,kernel_size=3,  padding="same"), Mish(),
#                                         nn.Conv1d(1,1,kernel_size=3,  padding="same"), Mish()
#         )
#         self.attention_block = nn.MultiheadAttention(time_embd_dim // 5, 1)
#         self.block1 = nn.Sequential(nn.Linear(2*time_embd_dim, 2*time_embd_dim), Mish(),
#                                           #nn.Linear(2*time_embd_dim, 2*time_embd_dim), Mish(),
#                                           nn.Linear(2*time_embd_dim, time_embd_dim), Mish(),
#                                           nn.Linear(time_embd_dim, time_embd_dim))
#         self.delete_me_layer = nn.Sequential(nn.Linear(dim, dim), Mish(), nn.Linear(dim, dim), Mish(), nn.Linear(dim, time_embd_dim), Mish())                                          
#         self.output_block = nn.Sequential(nn.Linear(2*time_embd_dim, dim), Mish(), nn.Linear(dim, dim))
#         #self.delete_me_layer = nn.Sequential(nn.Linear(2, 10), Mish(), nn.Linear(10, 50), Mish(), nn.Linear(50, time_embd_dim))
#         #self.fc_list = nn.ModuleList([nn.Linear(3, 10), nn.Linear(10, 100), nn.Linear(100, 100),  nn.Linear(100, 100), nn.Linear(100, 100), nn.Linear(100, 10)])
#         #self.fc_last = nn.Linear(10, 2)

    def forward(self, x, t):    
        #t_emb = t.repeat(x.shape[0], 1)    
        #t_emb = self.time_mlp(t)
        x = self.delete_me_layer(x)
        t = t.repeat(x.shape[1], 1).transpose(1,0).type(torch.FloatTensor).to(device)
        t_emb = self.time_mlp(t)
        #x = x + t_emb
        x = torch.cat([x, t_emb], dim=1)
        x = self.block1(x)
        xr = x.reshape([x.shape[0], 5, -1])
        h, _ = self.attention_block(xr, xr, xr)
        h = h.reshape(x.shape)
        x = torch.cat([x, h], dim=1)
        x = self.output_block(x)
        return x


class WeightedMSE(nn.Module):
    def __init__(self, betta, device):
        super(WeightedMSE, self).__init__()
        self.weights = 1 / betta
        self.weights = self.weights.to(device)

    def forward(self, input, target, t):
        weights = self.weights[t]
        loss = torch.mean(weights * ((input - target)**2).mean(dim=[1,2]) )
        return loss


class WeightedMSEepsilon(nn.Module):
    def __init__(self, alpha, device):
        super(WeightedMSEepsilon, self).__init__()
        alpha_cumprod = torch.cumprod(alpha, axis=0)
        self.weights = (1 - alpha) / (alpha * (1 - alpha_cumprod))
        self.weights = self.weights.to(device)

    def forward(self, input, target, t):
        weights = self.weights[t]
        loss = torch.mean(weights * ((input - target)**2).sum(dim=1) )
        return loss


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def train_epoch_channels(dataloader, net, diffused_model, optimizer, loss_f, device, number_of_repetition=1):
    #device = net.dummy_param.divice
    loss_list = []
    for i in range(number_of_repetition):
        for ecg in dataloader:
            optimizer.zero_grad()
            #batch_size random int variables from 1 to Tmax-1
            t = torch.randperm(diffused_model.time_steps-2)[:ecg.shape[0]] + 1 #diffused_model.time_steps-2
            xt, mu = diffused_model.mean_posterior(ecg, t)
            xt = xt.to(device)
            t = t.to(device)
            mu = mu.to(device)
            delta = mu - xt
            mu_estim = net(xt, t)
            loss = loss_f(mu_estim, delta, t)#
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
    return sum(loss_list) / len(loss_list)


def train_epoch(dataloader, net, diffused_model, optimizer, loss_f, batch_size, device):
    #device = net.dummy_param.divice
    loss_list = []
    for ecg in dataloader:
        optimizer.zero_grad()
        #batch_size random int variables from 1 to Tmax-1
        t = torch.randperm(diffused_model.time_steps-2)[:ecg.shape[0]] + 1 #diffused_model.time_steps-2
        xt, mu = diffused_model.mean_posterior(ecg, t)
        xt = xt.to(device)
        t = t.to(device)
        mu = mu.to(device)
        delta = mu - xt
        mu_estim = net(xt.unsqueeze(1), t).squeeze(1)
        # t = t[:,None].repeat(1, ecg.shape[1])
        # t = torch.exp(t/500)
        # test_function = t * torch.sqrt(torch.abs(ecg.flip([-1]) + ecg)).to(device)
        loss = loss_f(mu_estim, delta, t)#
        #delta2 = mu_estim - xt
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
    return sum(loss_list) / len(loss_list)



def train_epoch_epsilon(dataloader, net, diffused_model, optimizer):
    device = net.bottleneck_conv1.bias.device
    loss_f = WeightedMSEepsilon(diffused_model.alpha, device)
    #loss_f = nn.MSELoss()
    loss_list = []
    for ecg in dataloader:
        optimizer.zero_grad()
        #batch_size random int variables from 1 to Tmax-1
        t = torch.randperm(diffused_model.time_steps-2)[:ecg.shape[0]] + 1 #diffused_model.time_steps-2
        xt, epsilon = diffused_model.diffuse(ecg, t)
        epsilon = epsilon.to(device)
        xt = xt.to(device)
        t = t.to(device)
        epsilon_pred = net(xt.unsqueeze(1), t).squeeze(1)
        loss = loss_f(epsilon_pred, epsilon, t)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
    return sum(loss_list) / len(loss_list)




if __name__ == "__main__":
    writer = SummaryWriter()
    dataset_path = "/ayb/vol1/kruzhilov/datasets/ecg/ningbo_128_small.npy"
    diffused_model = GaussianDiffusion(n_channels=2)
    
    # dataset = Ellips(100000)
    dataset = ECGdataset(dataset_path)

    #net = MuNet(256)
    net = ECGunetChannels(kernel_size=5, n_channels=2)
    device = "cuda:2"
    net = net.to(device)
    save_weights_path = "/ayb/vol1/kruzhilov/weights/diff_ecg/ecg2chan_cos.pth"
    net.load_state_dict(torch.load(save_weights_path))
    ema = EMA(beta=0.995)
    batch_size = 400
    epoch_number = 30000

    lr = 0.0001
    dataloader = DataLoader(dataset, batch_size=batch_size)

    #optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
    optimizer = torch_optimizer.Lamb(params=net.parameters(), lr=lr)
    loss_f = WeightedMSE(diffused_model.betta, device)
    for i in range(epoch_number):
        ema_net = deepcopy(net)
        mean_loss = train_epoch_channels(dataloader, net, diffused_model, optimizer, loss_f, device, number_of_repetition=50)
        writer.add_scalar("Loss", mean_loss, i)
        #mean_loss = train_epoch_epsilon(dataloader, net, diffused_model, optimizer)
        #ema.update_model_average(ema_net, net)
        print(i, mean_loss)
        if (i % 5 == 0) and i > 0:
          torch.save(net.state_dict(),save_weights_path) 
    torch.save(net.state_dict(), save_weights_path)

 


# %%
