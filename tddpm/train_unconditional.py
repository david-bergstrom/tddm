from common.data import SlidingDataset
from model import TransformerDenoiser

from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import GaussianDiffusion1D
from torch.utils.data import DataLoader, RandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from tqdm import tqdm
from common.preproc import minmaxscale

import argparse
import os
import random
import torch
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../')
from data import load_dataset_from_str 


def main():
    parser = argparse.ArgumentParser(
                        prog='train_unconditional',
                        description='Train the ablation without spatial prior (an unconditional diffusion model)')

    parser.add_argument('dataset', choices=['geolife-25', 'geolife-100', 'porto-25', 'porto-100', 'cabspotting-25', 'cabspotting-100', 'brest-100'])
    parser.add_argument('seq_len', choices=[24, 32, 64, 128, 256, 512, 1024], type=int, default=128)
    parser.add_argument('--seed', type=int, default=None, help='Random seed for torch/numpy/random; also appended to the checkpoint and sample names as -seed<N>')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Seeded torch/numpy/random with {args.seed}")

    return evaluate_parameters(args)


def evaluate_parameters(args):
    dataset_str = args.dataset
    seq_len = args.seq_len
    num_workers = 20
    batch_size = 128
    objective = 'pred_noise'
    beta_schedule = 'cosine'
    epochs = 100
    num_epochs = epochs
    number_warmup_epochs = 10
    max_plot_dim = 8
    noise_scale = 1
    # from hyperparameter search
    hidden_dim = 128
    learning_rate = 0.00017482789809553816
    n_heads = 2
    n_layers = 8
    timesteps = 500

    exp_desc = f'unconditional-tddpm-{dataset_str}-{seq_len}-{noise_scale}'
    if args.seed is not None:
        exp_desc += f'-seed{args.seed}'

    # Never overwrite an earlier run's weights or samples
    for path in (f'model_weights/{exp_desc}.pth', f'_out/{exp_desc}.npy'):
        if os.path.exists(path):
            raise SystemExit(f'{path} already exists; refusing to overwrite it')
    os.makedirs('model_weights', exist_ok=True)
    os.makedirs('_out', exist_ok=True)

    dataset = load_dataset_from_str(dataset_str, prefix='../data/')

    if isinstance(dataset, np.ndarray) and len(dataset.shape) == 2:
        dataset = np.expand_dims(dataset, axis=0)

    train_new = minmaxscale(dataset)

    n_dim = train_new[0][0].shape[-1]
    plot_dims = min(max_plot_dim, n_dim)

    dataset = SlidingDataset(train_new, seq_len)

    samples_per_epoch = min(10000, len(dataset))

    sampler = RandomSampler(dataset, replacement=False, num_samples=samples_per_epoch)

    train_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            sampler=sampler)

    model = TransformerDenoiser(
        dim_data=n_dim, 
        dim_embedding=hidden_dim,
        seq_len=seq_len,
        n_heads=n_heads,
        n_layers=n_layers
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = seq_len,
        timesteps = timesteps,
        objective = objective,
        beta_schedule = beta_schedule,
        noise_scale = noise_scale
    )

    diffusion.cuda()

    adam_betas = (0.9, 0.99)

    optimizer = torch.optim.Adam(diffusion.parameters(), learning_rate, adam_betas)

    train_scheduler = CosineAnnealingLR(optimizer, num_epochs)

    def warmup(current_step: int):
        return 1 / (10 ** (float(number_warmup_epochs - current_step)))

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup)

    lr_scheduler = SequentialLR(optimizer, [warmup_scheduler, train_scheduler], [number_warmup_epochs])

    def generate_samples(n_samples=100, sample_batch_size=batch_size):    
        diffusion.eval()

        all_samples = []

        n_batches = n_samples // sample_batch_size
        remainder = n_samples % sample_batch_size

        def sample(n_samples):
            samples = diffusion.sample(n_samples)
            # Convert to numpy
            samples = samples.cpu().numpy()

            return samples

        for i in range(n_batches):
            all_samples.append(sample(sample_batch_size))

        all_samples.append(sample(remainder))

        all_samples = np.concatenate(all_samples, axis=0)
        # print(all_samples.shape)
        
        diffusion.train()
        
        f, axs = plt.subplots(figsize=(6*plot_dims, 4), ncols=plot_dims, sharey=True, sharex=True, constrained_layout=True)
        
        if n_dim == 1:
            axs.plot(all_samples[:, 0, :].T)
        else:
            for i, ax in enumerate(axs):
                ax.plot(all_samples[:, i, :].T)

        return np.transpose(all_samples, (0, 2, 1)), f
    
    progress_bar = tqdm(range(epochs))

    for epoch in progress_bar:
        epoch_loss = 0
        for x in tqdm(train_loader):
            x = torch.transpose(x, 1, 2)
            x = x.cuda()

            loss, _ = diffusion(x)
            epoch_loss += loss.item()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        # epoch_loss
        epoch_loss /= len(train_loader)

        lr_scheduler.step()

        # writer.add_scalar('Loss/train', epoch_loss, epoch + 1)

    torch.save(
        {
        'model_state_dict': model.state_dict(), 
        'diffusion_state_dict': diffusion.state_dict(),
        'optim_state_dict': optimizer.state_dict()
        },
        f"model_weights/{exp_desc}.pth"
    )

    n_samples = min(len(dataset), 10000)
    all_samples, _ = generate_samples(n_samples=n_samples)
    np.save(f'_out/{exp_desc}.npy', all_samples)


if __name__ == '__main__':
    main()
