import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from types import SimpleNamespace
# from utils.Traj_UNe import *
from main import Uncond_UNet
from utils.config import args
from utils.utils import *
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('seq_len', type=int)
parser.add_argument('weights')
clargs = parser.parse_args()

temp = {}
for k, v in args.items():
    temp[k] = SimpleNamespace(**v)

config = SimpleNamespace(**temp)
config.data.traj_length = clargs.seq_len

unet = Uncond_UNet(config).cuda()
# load the model
weights = torch.load(clargs.weights)
unet.load_state_dict(weights)

n_steps = config.diffusion.num_diffusion_timesteps
beta = torch.linspace(config.diffusion.beta_start,
                          config.diffusion.beta_end, n_steps).cuda()
alpha = 1. - beta
alpha_bar = torch.cumprod(alpha, dim=0)
lr = 2e-4  # Explore this - might want it lower when training on the full dataset

eta=0.0
timesteps=100
skip = n_steps // timesteps
seq = range(0, n_steps, skip)

# load head information for guide trajectory generation
batchsize = 500
# head = np.load('heads.npy',
#                    allow_pickle=True)
# head = torch.from_numpy(head).float()
# dataloader = DataLoader(head, batch_size=batchsize, shuffle=True, num_workers=4)

Gen_traj = []
Gen_head = []
for i in tqdm(range(20)):
    # head = next(iter(dataloader))
    # lengths = head[:, 3]
    # lengths = lengths * len_std + len_mean
    # lengths = lengths.int()
    # tes = head[:,:6].numpy()
    # Gen_head.extend((tes*hstd+hmean))
    # head = head.cuda()
    # Start with random noise
    x = torch.randn(batchsize, 2, config.data.traj_length).cuda()
    ims = []
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        with torch.no_grad():
            pred_noise = unet(x, t)
            # print(pred_noise.shape)
            x = p_xt(x, pred_noise, t, next_t, beta, eta)
            if i % 10 == 0:
                ims.append(x.cpu().squeeze(0))
    trajs = ims[-1].cpu().numpy()
    trajs = trajs[:,:2,:]
    # resample the trajectory length
    for j in range(batchsize):
        # new_traj = resample_trajectory(trajs[j].T, lengths[j])
        # new_traj = new_traj #  * std + mean
        new_traj = trajs[j].T
        Gen_traj.append(new_traj)

result = np.array(Gen_traj, dtype=np.float32)
np.save(f'difftraj-synthetic-{clargs.dataset}-{clargs.seq_len}.npy', result)
