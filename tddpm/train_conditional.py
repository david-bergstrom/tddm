from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import wandb
import numpy as np
import argparse
import os
import random
import re

os.makedirs('model_weights', exist_ok=True)

from denoising_diffusion_pytorch import GaussianDiffusion1D

from model import TransformerDenoiser
from dataset import ConditionalDataset


def main():
    parser = argparse.ArgumentParser(
                        prog='train_conditional',
                        description='Train TDDM on (spatial prior, trajectory) pairs from conditional-variant-generator.py')

    parser.add_argument('dataset', choices=[
        'porto-25', 'porto-100', 'geolife-25', 'geolife-100', 'cabspotting-25', 'cabspotting-100',
        'geolife-100-no-map-match', 'porto-100-no-map-match', 'cabspotting-100-no-map-match',
        # Corner variants for 4-corner generalization experiment
        'geolife-25-bl', 'geolife-25-br', 'geolife-25-tl', 'geolife-25-tr',
        'porto-25-bl', 'porto-25-br', 'porto-25-tl', 'porto-25-tr',
        'cabspotting-25-bl', 'cabspotting-25-br', 'cabspotting-25-tl', 'cabspotting-25-tr',
        # Maritime AIS
        'brest-100',
        'brest-25-bl', 'brest-25-br', 'brest-25-tl', 'brest-25-tr',
    ])
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--res', type=int, default=64, help='Heatmap resolution (default: 64)')
    parser.add_argument('--region-size', type=int, default=3000, help='Region size in meters (default: 3000)')
    parser.add_argument('--use-fp16', action='store_true', help='Use float16 precision for heatmaps to reduce RAM by 4x (negligible precision loss)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for torch/numpy/random; also appended to checkpoint names as -seed<N> (load with sample.py --run-id seed<N>)')
    args = parser.parse_args()

    if args.resume:
        # Checkpoint names encode the seed; resuming with a different --seed would
        # write the continued run to another run's filenames (a missing --seed
        # would overwrite the unseeded checkpoint).
        match = re.search(r'-seed(\d+)(?:-\d+)?\.pth$', args.resume)
        resume_seed = int(match.group(1)) if match else None
        if resume_seed != args.seed:
            parser.error(f"--resume checkpoint has seed {resume_seed} but --seed is {args.seed}; "
                         f"pass the matching --seed so outputs keep the same filenames")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Seeded torch/numpy/random with {args.seed}")

    num_workers = 10

    dataset_str = args.dataset
    n_dim = 2
    seq_len = 128
    save_n = 100000
    region_size = args.region_size
    heatmap_res = args.res

    batch_size = 128
    objective = 'pred_noise'
    reconstruction_loss_weight = 0
    beta_schedule = 'cosine'
    epochs = 100
    adam_betas = (0.9, 0.99)

    # From hyperparameter search
    hidden_dim = 128
    learning_rate = 0.00017482789809553816
    n_heads = 2
    n_layers = 8
    timesteps = 500

    windowed_subset_save_path = f'../data/preprocessed/'

    # Build filename based on resolution
    if heatmap_res != 64:
        res_suffix = f'-{heatmap_res}res'
    else:
        res_suffix = ''

    filename = f'{dataset_str}-{save_n}-conditional-128-{region_size}{res_suffix}.npz'

    dataset_path = os.path.join(windowed_subset_save_path, filename)
    print(f"Using dataset ({heatmap_res}x{heatmap_res}): {dataset_path}")
    if args.use_fp16:
        print("Using float16 precision for heatmaps (4x RAM reduction)")

    exp_str = f'conditional-{dataset_str}-{save_n}-{region_size}'
    if heatmap_res != 64:
        exp_str += f'-{heatmap_res}res'
    if args.seed is not None:
        exp_str += f'-seed{args.seed}'

    dataset = ConditionalDataset(dataset_path, use_fp16=args.use_fp16)

    def test_conditioned_generation(idx, n_samples=100):
        sample_batch_size = batch_size
        
        query_heatmap, something = dataset[idx]
        if len(query_heatmap.shape) == 2:
            query_heatmap = query_heatmap.unsqueeze(0)
        
        diffusion.eval()

        all_samples = []

        n_batches = n_samples // sample_batch_size
        remainder = n_samples % sample_batch_size

        def sample(n_samples):
            condition = query_heatmap.unsqueeze(0).expand(n_samples, 1, heatmap_res, heatmap_res).float().cuda()
            # Sample `n_samples` number of samples from the model
            samples = diffusion.sample_conditional(condition, n_samples)
            # Convert to numpy
            samples = samples.cpu().numpy()

            return samples

        for i in range(n_batches):
            all_samples.append(sample(sample_batch_size))

        all_samples.append(sample(remainder))

        all_samples = np.concatenate(all_samples, axis=0)
        
        diffusion.train()
        
        f, (ax1, ax2, ax3) = plt.subplots(figsize=(12, 4), ncols=3, sharey=True, sharex=True, constrained_layout=True)
        ax1.imshow(np.log(query_heatmap[0]), extent=[0, 1, 0, 1])
        ax1.plot(something[:, 1], 1-something[:, 0])
        ax2.plot(all_samples[:, 1, :].T, 1-all_samples[:, 0, :].T)
        ax3.imshow(np.log(query_heatmap[0]), extent=[0, 1, 0, 1])
        ax3.plot(all_samples[:, 1, :].T, 1-all_samples[:, 0, :].T, alpha=.50)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        return f

    # Create data loader for the data
    train_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=True)
    
    model = TransformerDenoiser(n_dim, hidden_dim, seq_len=seq_len, n_heads=n_heads, n_layers=n_layers, heatmap_resolution=heatmap_res)

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = seq_len,
        timesteps = timesteps,
        objective = objective,
        beta_schedule = beta_schedule,
        reconstruction_loss_weight = reconstruction_loss_weight
    )

    diffusion.cuda()

    optimizer = torch.optim.Adam(diffusion.parameters(), learning_rate, adam_betas)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Handle checkpoint resuming
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, weights_only=False)

        # Load model and optimizer states (required)
        model.load_state_dict(checkpoint['model_state_dict'])
        diffusion.load_state_dict(checkpoint['diffusion_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])

        # Load epoch number (try from checkpoint, fallback to parsing filename)
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        else:
            # Parse epoch from filename pattern: {exp_str}-{epoch}.pth
            match = re.search(r'-(\d+)\.pth$', args.resume)
            if match:
                start_epoch = int(match.group(1)) + 1
                print(f"Parsed epoch {start_epoch - 1} from filename")
            else:
                print("Warning: Could not determine epoch from checkpoint, starting from 0")

        # Load LR scheduler state (optional, fallback to stepping manually)
        if 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            print(f"Loaded LR scheduler state")
        else:
            # Fallback: step the scheduler to get to the right state
            for _ in range(start_epoch):
                lr_scheduler.step()
            print(f"Recreated LR scheduler state by stepping {start_epoch} times")

        # Load random states (optional, for reproducibility)
        if 'torch_rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_rng_state'])
            print("Loaded PyTorch RNG state")
        if 'numpy_rng_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_rng_state'])
            print("Loaded NumPy RNG state")
        if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
            print("Loaded CUDA RNG state")

        print(f"Resuming training from epoch {start_epoch}")

    progress_bar = tqdm(range(start_epoch, epochs), position=0, desc='Epochs:')

    wandb.init(
        project="tddpm",
        group=f'{dataset_str}',
        reinit=True,
        config={'seed': args.seed}
    )

    for epoch in progress_bar:
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_pbar = tqdm(total=len(dataset), position=1, desc='Iterating over train set', leave=False)
        for heatmap, x in train_loader:
            heatmap = heatmap.cuda()

            x = torch.transpose(x, 1, 2)
            x = x.cuda()

            loss, loss_d = diffusion(x, conditional = heatmap)
            loss.backward()

            epoch_loss += loss_d['ddpm_loss']

            optimizer.step()
            optimizer.zero_grad()

            epoch_pbar.update(len(x))

        epoch_loss /= len(train_loader)
        epoch_recon_loss /= len(train_loader)

        results = {
            'epoch_loss': epoch_loss,
            'lr': lr_scheduler.get_lr()[0]
        }

        if epoch % 10 == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'diffusion_state_dict': diffusion.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'epoch': epoch,
                'torch_rng_state': torch.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
            }
            # Save CUDA RNG state if available
            if torch.cuda.is_available():
                checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state_all()

            torch.save(checkpoint, f"model_weights/{exp_str}-{epoch}.pth")

            for i in [9359, 4204, 3604, 3076, 9904, 382, 352, 115, 272, 221, 777, 991, 602, 947, 114]:
                results[f'samples_{i}'] = test_conditioned_generation(i, n_samples=100)

        wandb.log(results)

        lr_scheduler.step()


    diffusion.cpu()

    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'diffusion_state_dict': diffusion.state_dict(),
        'optim_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'epoch': epochs - 1,
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
    }
    # Save CUDA RNG state if available
    if torch.cuda.is_available():
        final_checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state_all()

    torch.save(final_checkpoint, f"model_weights/{exp_str}.pth")


if __name__ == '__main__':
    main()
