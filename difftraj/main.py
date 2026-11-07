import torch
import torch.nn as nn
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from types import SimpleNamespace
from utils.config import args
from utils.EMA import EMAHelper
from utils.Traj_UNet import *
from utils.utils import p_xt
from utils.logger import Logger, log_info
from pathlib import Path
import random
import shutil
import sys
sys.path.append("../")
from data import load_dataset_from_str, minmaxscale, rolling_window
import argparse
import wandb
import time
from tqdm import tqdm


# This code part from https://github.com/sunlin-ai/diffusion_tutorial
# This file is then updated using the solution from the issue 11 (https://github.com/Yasoz/DiffTraj/issues/11) to run in unconditional model and adapted to run with benchmark dataloaders.


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)


class Uncond_UNet(nn.Module):
    """
    This is the model without conditional information.

    Copied from: https://github.com/Yasoz/DiffTraj/issues/11
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.unet = Model(config)

    def forward(self, x, t, attr=None):
        """

        :param x: Noisy sample
        :param t: Timestep
        :param attr: Ignored, only for compatibility
        :return: Predicted noise
        """
        pred_noise = self.unet(x, t, extra_embed=None)
        return pred_noise


def main(config, logger, exp_dir):

    # Modified to return the noise itself as well
    def q_xt_x0(x0, t):
        mean = gather(alpha_bar, t)**0.5 * x0
        var = 1 - gather(alpha_bar, t)
        eps = torch.randn_like(x0).to(x0.device)
        return mean + (var**0.5) * eps, eps  # also returns noise

    wandb.init(
      project='tddpm',
      group='difftraj',
      config={'dataset': config.data.dataset, 'seq_len': config.data.traj_length,
              'seed': getattr(config, 'seed', None)}
      )

    # Create the model
    unet = Uncond_UNet(config)
    unet.cuda()
    # print(unet)

    traj = load_dataset_from_str(dataset_str=config.data.dataset, prefix='../data/')
    traj = minmaxscale(traj)

    seq_len = config.data.traj_length
    if config.data.dataset in ['geolife-cropped', 'porto', 'porto-25', 'porto-100', 'geolife-25', 'geolife-100', 'brest-100']:
      assert seq_len == 128
    else:
      traj = rolling_window(traj, seq_len)

    traj = traj[:, :, :2]
    # head = np.load('./xxxxxx',
    #                allow_pickle=True)

    traj = np.swapaxes(traj, 1, 2)
    traj = torch.from_numpy(traj).float()
    # head = torch.from_numpy(head).float()
    ###########################################################
    # The input shape of traj and head list as follows:
    # traj: [batch_size, 2, traj_length]   2: latitude and longitude
    # head: [batch_size, 8]   8: departure_time, trip_distance,  trip_time, trip_length, avg_dis, avg_speed, start_id, end_id
    ###########################################################
    # dataset = TensorDataset(traj, head)
    dataset = TensorDataset(traj)
    dataloader = DataLoader(dataset,
                            batch_size=config.training.batch_size,
                            shuffle=True,
                            num_workers=8)

    # Training params
    # Set up some parameters
    n_steps = config.diffusion.num_diffusion_timesteps
    beta = torch.linspace(config.diffusion.beta_start,
                          config.diffusion.beta_end, n_steps)
    beta = beta.cuda()
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    lr = 2e-4  # Explore this - might want it lower when training on the full dataset

    losses = []  # Store losses for later plotting
    # optimizer
    optim = torch.optim.AdamW(unet.parameters(), lr=lr)  # Optimizer

    # EMA
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(unet)
    else:
        ema_helper = None

    # new filefold for save model pt
    model_save = exp_dir / 'models' / (timestamp + '/')
    if not os.path.exists(model_save):
        os.makedirs(model_save)

    start_training = time.time()

    # config.training.n_epochs = 1
    for epoch in range(1, config.training.n_epochs + 1):
        logger.info("<----Epoch-{}---->".format(epoch))
        for _, (trainx, ) in enumerate(dataloader):
            x0 = trainx.cuda()
            # head = head.cuda()
            t = torch.randint(low=0, high=n_steps,
                              size=(len(x0) // 2 + 1, ))
            t = t.cuda()
            t = torch.cat([t, n_steps - t - 1], dim=0)[:len(x0)]
            # Get the noised images (xt) and the noise (our target)
            xt, noise = q_xt_x0(x0, t)
            # Run xt through the network to get its predictions
            pred_noise = unet(xt.float(), t)
            # Compare the predictions with the targets
            loss = F.mse_loss(noise.float(), pred_noise)
            # Store the loss for later viewing
            losses.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
            if config.model.ema:
                ema_helper.update(unet)
        if (epoch) % 10 == 0:
            m_path = model_save / f"unet_{epoch}.pt"
            torch.save(unet.state_dict(), m_path)
            seed_tag = f"seed{config.seed}_" if getattr(config, 'seed', None) is not None else ''
            m_path = exp_dir / 'results' / f"loss_{timestamp}_{seed_tag}{epoch}.npy"
            np.save(m_path, np.array(losses))

    stop_training = time.time()
    wandb.run.summary['training_time'] = stop_training - start_training

    start_sampling = time.time()

    eta=0.0
    timesteps=100
    skip = n_steps // timesteps
    seq = range(0, n_steps, skip)

    batchsize = 500

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
    stop_sampling = time.time()
    wandb.run.summary['sampling_time'] = stop_sampling - start_sampling

    seed_suffix = f'-seed{config.seed}' if getattr(config, 'seed', None) is not None else ''
    synthetic_data_path = f'difftraj-synthetic-{config.data.dataset}-{config.data.traj_length}{seed_suffix}.npy'
    np.save(synthetic_data_path, result)
    wandb.run.summary['synthetic_data_path'] = synthetic_data_path


if __name__ == "__main__":
    # Load configuration
    temp = {}
    for k, v in args.items():
        temp[k] = SimpleNamespace(**v)
    config = SimpleNamespace(**temp)

    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--dataset', type=str, help='Dataset name', required=True)
    parser.add_argument('--seq_len', type=int, help='Sequence length', required=True)
    parser.add_argument('--epochs', type=int, help='Number of training epochs', default=500)
    parser.add_argument('--seed', type=int, default=None, help='Random seed for torch/numpy/random; also appended to the synthetic output filename as -seed<N>')
    args = parser.parse_args()

    config.data.dataset = args.dataset
    config.data.traj_length = args.seq_len
    config.training.n_epochs = args.epochs
    config.seed = args.seed

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Seeded torch/numpy/random with {args.seed}")

    root_dir = Path(__name__).resolve().parents[0]
    result_name = '{}_steps={}_len={}_{}_bs={}'.format(
        config.data.dataset, config.diffusion.num_diffusion_timesteps,
        config.data.traj_length, config.diffusion.beta_end,
        config.training.batch_size)
    if args.seed is not None:
        result_name += f'_seed{args.seed}'
    exp_dir = root_dir / "DiffTraj" / result_name
    for d in ["results", "models", "logs","Files"]:
        os.makedirs(exp_dir / d, exist_ok=True)
    print("All files saved path ---->>", exp_dir)
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    files_save = exp_dir / 'Files' / (timestamp + '/')
    if not os.path.exists(files_save):
        os.makedirs(files_save)
    shutil.copy('./utils/config.py', files_save)
    shutil.copy('./utils/Traj_UNet.py', files_save)

    # filename = f"difftraj-synthetic-{dataset}-{seq_len}-experiment-{logger.experiment_id}.npy"

    logger = Logger(
        __name__,
        log_path=exp_dir / "logs" / (timestamp + '.log'),
        colorize=True,
    )
    log_info(config, logger)
    main(config, logger, exp_dir)
