#%%
import torch
import numpy as np
import matplotlib.pyplot as plt 
from unet1d import ECGunetChannels
from diffusion_model import GaussianDiffusion
from ecg_data.ecg_dataset import ECGdataset


if __name__ == "__main__":
    device = "cuda:2"
    n_channels = 2
    diffused_model = GaussianDiffusion(n_channels=n_channels)
    net = ECGunetChannels(kernel_size=5, n_channels=n_channels)
    save_weights_path = "/ayb/vol1/kruzhilov/weights/diff_ecg/ecg2chan_cos.pth"
    net.load_state_dict(torch.load(save_weights_path))
    net = net.to(device)
    xi = diffused_model.generation_from_net(net, 2, device, dim=512)
    xi = xi.clamp(-0.4, 1.0)
    print("-----------------------------------")
    fig = plt.figure()
    plt.plot(xi[0,0,:].squeeze(0))
    fig = plt.figure()
    plt.plot(xi[0,1,:].squeeze(0))

    #Dx_164890007.npy ningbo_128_small.npy
    # dataset_path = "/ayb/vol1/kruzhilov/datasets/ecg/Dx_164890007.npy"
    # dataset = ECGdataset(dataset_path)
    # ecg = dataset.__getitem__(26)
    # print("-----------------------------------")
    # fig = plt.figure()
    # plt.plot(ecg[0,:])
    # fig = plt.figure()
    # plt.plot(ecg[1,:])

# %%
