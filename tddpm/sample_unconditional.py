"""
Standalone sampler for unconditional TDDM weights.

Mirrors generate_samples() in train_unconditional.py so we can re-sample without
retraining when the auto-sample step at the end of training fails (e.g. missing
_out/ directory) or when we want different sample counts.

Usage:
    python sample_unconditional.py <dataset> [--n-samples N] [--batch-size B] [--output-path PATH]

Output filename matches synthetic_loader.py's expectation:
    unconditional-tddpm-{dataset}-128.npy
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.append('../')

from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import GaussianDiffusion1D
from model import TransformerDenoiser


def main():
    parser = argparse.ArgumentParser(description='Sample from a trained unconditional TDDM model.')
    parser.add_argument('dataset', choices=['geolife-25', 'geolife-100', 'porto-25', 'porto-100',
                                            'cabspotting-25', 'cabspotting-100', 'brest-100'])
    parser.add_argument('--n-samples', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--noise-scale', type=float, default=1)
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to .pth (default: model_weights/unconditional-tddpm-<dataset>-<seq_len>-<noise_scale>.pth)')
    parser.add_argument('--output-path', type=str, default=None,
                        help='Output .npy path (default: ../synthetic_data/unconditional-tddpm-<dataset>-<seq_len>.npy)')
    args = parser.parse_args()

    exp_desc = f'unconditional-tddpm-{args.dataset}-{args.seq_len}-{int(args.noise_scale)}'
    weights_path = args.weights or f'model_weights/{exp_desc}.pth'
    output_path = args.output_path or f'../synthetic_data/unconditional-tddpm-{args.dataset}-{args.seq_len}.npy'
    if os.path.exists(output_path):
        raise SystemExit(f'{output_path} already exists; refusing to overwrite it (use --output-path)')

    # Hyperparams must match train_unconditional.py
    hidden_dim = 128
    n_heads = 2
    n_layers = 8
    timesteps = 500

    n_dim = 2

    model = TransformerDenoiser(
        dim_data=n_dim,
        dim_embedding=hidden_dim,
        seq_len=args.seq_len,
        n_heads=n_heads,
        n_layers=n_layers,
    )
    diffusion = GaussianDiffusion1D(
        model,
        seq_length=args.seq_len,
        timesteps=timesteps,
        objective='pred_noise',
        beta_schedule='cosine',
        noise_scale=args.noise_scale,
    ).cuda()

    print(f"Loading weights from {weights_path}")
    ckpt = torch.load(weights_path, weights_only=False, map_location='cuda')
    diffusion.load_state_dict(ckpt['diffusion_state_dict'])
    diffusion.eval()

    print(f"Sampling {args.n_samples} trajectories (batch {args.batch_size})...")
    all_samples = []
    n_batches = args.n_samples // args.batch_size
    remainder = args.n_samples % args.batch_size

    with torch.no_grad():
        for i in range(n_batches):
            s = diffusion.sample(args.batch_size).cpu().numpy()
            all_samples.append(s)
            print(f"  batch {i+1}/{n_batches}", end='\r', flush=True)
        if remainder:
            all_samples.append(diffusion.sample(remainder).cpu().numpy())

    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = np.transpose(all_samples, (0, 2, 1))

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.save(output_path, all_samples)
    print(f"\nSaved {len(all_samples)} samples to {output_path}")


if __name__ == '__main__':
    main()
