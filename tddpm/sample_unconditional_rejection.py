"""
Rejection sampling script for unconditional trajectory diffusion models.

This script generates synthetic trajectories from an unconditional model and applies
rejection sampling based on a target dataset's spatial distribution. Samples are rejected
if they fall outside regions with non-zero density in the target dataset.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time
from tqdm import tqdm
import pickle

import sys
sys.path.append('../')
from data import load_dataset_from_str, minmaxscale
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import GaussianDiffusion1D
from model import TransformerDenoiser
from preproc import get_limit


def compute_heatmap_resolution(dataset_scaled, dataset_name, target_pixels_per_km=21.33):
    """
    Compute heatmap resolution dynamically based on the actual physical size of the dataset.

    This matches the resolution used in sample.py where 3km × 3km regions use 64×64 pixels,
    giving 21.33 pixels/km. For the full city dataset, we calculate the actual physical size
    and apply the same pixels/km ratio.

    Args:
        dataset_scaled: Trajectory dataset scaled to [0, 1] space
        dataset_name: Name of the dataset (e.g., 'geolife-100') to look up bounds
        target_pixels_per_km: Target resolution in pixels per kilometer (default: 21.33)

    Returns:
        (res_y, res_x): Heatmap resolution in pixels
    """
    print(f"  Computing dataset bounds in normalized space...")
    # Get dataset bounds (should be close to [0, 1])
    min_y, min_x = dataset_scaled.min(axis=(0, 1))
    max_y, max_x = dataset_scaled.max(axis=(0, 1))
    print(f"    Y range: [{min_y:.4f}, {max_y:.4f}]")
    print(f"    X range: [{min_x:.4f}, {max_x:.4f}]")

    # Get original lat/lon bounds from preproc
    limits = get_limit(dataset_name)
    min_lat, max_lat, min_lon, max_lon = limits
    print(f"  Original lat/lon bounds: lat=[{min_lat:.4f}, {max_lat:.4f}], lon=[{min_lon:.4f}, {max_lon:.4f}]")

    # Calculate physical size in kilometers using haversine approximation
    # 1 degree latitude ≈ 111 km
    lat_km = abs(max_lat - min_lat) * 111.0

    # 1 degree longitude ≈ 111 km × cos(latitude)
    # Use the average latitude for the cosine correction
    avg_lat = (min_lat + max_lat) / 2
    lon_km = abs(max_lon - min_lon) * 111.0 * np.cos(np.radians(avg_lat))

    print(f"  Physical size: {lat_km:.2f} km (lat) × {lon_km:.2f} km (lon)")

    # Compute resolution based on physical size and target pixels/km
    res_y = int(np.ceil(lat_km * target_pixels_per_km))
    res_x = int(np.ceil(lon_km * target_pixels_per_km))

    # Ensure minimum resolution
    res_y = max(res_y, 64)
    res_x = max(res_x, 64)

    # Compute normalized extent
    extent_y = max_y - min_y
    extent_x = max_x - min_x

    print(f"  Target resolution: {target_pixels_per_km:.2f} pixels/km")
    print(f"  Computed heatmap resolution: {res_y} × {res_x} pixels")
    print(f"  Effective resolution: {res_y/lat_km:.2f} × {res_x/lon_km:.2f} pixels/km")
    print(f"  This matches the {target_pixels_per_km:.2f} pixels/km used in sample.py for 3km regions")

    return res_y, res_x


def generate_target_heatmap(dataset_scaled, res_y, res_x):
    """
    Generate a heatmap of the target dataset in normalized [0,1] space.

    Args:
        dataset_scaled: Dataset scaled to [0, 1] space
        res_y: Heatmap height in pixels
        res_x: Heatmap width in pixels

    Returns:
        heatmap: 2D array of shape (res_y, res_x) with density values
        Binary mask where density > 0
    """
    print(f"  Concatenating {len(dataset_scaled)} trajectories...")
    # Concatenate all trajectory points
    all_points = np.concatenate(dataset_scaled)
    print(f"  Total points: {len(all_points):,}")

    # Create 2D histogram (heatmap)
    print(f"  Creating {res_y}×{res_x} histogram...")
    # Note: histogram2d uses (x, y) ordering but we want (y, x) for lat/lon
    heatmap, xedges, yedges = np.histogram2d(
        all_points[:, 0],  # y-coordinates (latitude)
        all_points[:, 1],  # x-coordinates (longitude)
        bins=(res_y, res_x),
        range=((0, 1), (0, 1)),
        density=False  # Use counts, not density
    )

    # Create binary mask for rejection sampling
    print(f"  Creating binary mask...")
    mask = heatmap > 0

    print(f"  Heatmap statistics:")
    print(f"    Non-zero cells: {mask.sum():,} / {mask.size:,} ({100*mask.sum()/mask.size:.2f}%)")
    print(f"    Max density: {heatmap.max():.0f} points/cell")
    print(f"    Mean density (non-zero): {heatmap[mask].mean():.2f} points/cell")

    return heatmap, mask


def check_trajectory_validity(trajectory, heatmap_mask, reject_threshold=0.1):
    """
    Check if a trajectory is valid based on the heatmap mask.

    Hard rejection: reject if ANY point falls in zero-density cells
    Tolerance: allow up to `reject_threshold` fraction of points outside

    Args:
        trajectory: Single trajectory of shape (seq_len, 2) in [0, 1] space
        heatmap_mask: Binary mask of shape (res_y, res_x) where True = valid region
        reject_threshold: Maximum fraction of points allowed outside (default: 0.1 = 10%)

    Returns:
        is_valid: Boolean indicating if trajectory passes rejection criteria
        fraction_outside: Fraction of points falling outside valid regions
    """
    seq_len = len(trajectory)
    res_y, res_x = heatmap_mask.shape

    # Convert continuous coordinates to bin indices
    # Clip to [0, 1] range to handle minor numerical errors
    traj_clipped = np.clip(trajectory, 0, 1)

    # Map to bins (be careful with edge case at exactly 1.0)
    y_bins = (traj_clipped[:, 0] * res_y).astype(int)
    x_bins = (traj_clipped[:, 1] * res_x).astype(int)

    # Clip indices to valid range [0, res-1]
    y_bins = np.clip(y_bins, 0, res_y - 1)
    x_bins = np.clip(x_bins, 0, res_x - 1)

    # Check which points fall in valid cells
    points_valid = heatmap_mask[y_bins, x_bins]

    # Compute fraction outside
    fraction_outside = 1.0 - points_valid.sum() / seq_len

    # Trajectory is valid if fraction outside is within tolerance
    is_valid = fraction_outside <= reject_threshold

    return is_valid, fraction_outside


def rejection_sample_unconditional(diffusion, heatmap_mask, n_target_samples,
                                   batch_size=256, reject_threshold=0.1,
                                   snapshot_interval=1000, max_iterations=300,
                                   output_dir='.'):
    """
    Generate samples from unconditional model with rejection sampling.

    Args:
        diffusion: Trained unconditional diffusion model
        heatmap_mask: Binary mask indicating valid spatial regions
        n_target_samples: Target number of accepted samples
        batch_size: Number of samples to generate per iteration
        reject_threshold: Maximum fraction of points allowed outside
        snapshot_interval: Save snapshot every N accepted samples
        max_iterations: Maximum number of sampling iterations
        output_dir: Directory to save snapshots

    Returns:
        accepted_samples: Array of accepted samples
        stats: Dictionary with sampling statistics
    """
    accepted_samples = []
    total_generated = 0
    iteration = 0

    stats = {
        'total_generated': 0,
        'total_accepted': 0,
        'acceptance_rate_history': [],
        'iteration_history': [],
        'fraction_outside_history': []
    }

    print(f"\nStarting rejection sampling:")
    print(f"  Target: {n_target_samples} accepted samples")
    print(f"  Batch size: {batch_size}")
    print(f"  Reject threshold: {reject_threshold*100:.1f}% points outside")
    print(f"  Snapshot interval: {snapshot_interval}")
    print()

    pbar = tqdm(total=n_target_samples, desc="Accepted samples",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    while len(accepted_samples) < n_target_samples and iteration < max_iterations:
        iteration += 1

        # Generate a batch of samples
        print(f"[Iteration {iteration}] Generating {batch_size} samples...", flush=True)
        with torch.no_grad():
            samples_batch = diffusion.sample(batch_size)  # Shape: (batch, 2, seq_len)
            samples_batch = samples_batch.cpu().numpy()
        print(f"[Iteration {iteration}] Generation complete. Checking validity...", flush=True)

        # Transpose to (batch, seq_len, 2)
        samples_batch = samples_batch.transpose(0, 2, 1)

        total_generated += batch_size

        # Check each sample
        batch_accepted = []
        fractions_outside = []

        for sample in samples_batch:
            is_valid, frac_outside = check_trajectory_validity(
                sample, heatmap_mask, reject_threshold
            )
            fractions_outside.append(frac_outside)

            if is_valid:
                batch_accepted.append(sample)

        # Add accepted samples
        n_accepted_this_batch = len(batch_accepted)
        if batch_accepted:
            accepted_samples.extend(batch_accepted)
            pbar.update(n_accepted_this_batch)

        # Update statistics
        current_acceptance_rate = len(accepted_samples) / total_generated
        stats['acceptance_rate_history'].append(current_acceptance_rate)
        stats['iteration_history'].append(iteration)
        stats['fraction_outside_history'].extend(fractions_outside)

        # Print iteration stats
        avg_frac_outside = np.mean(fractions_outside)
        print(f"[Iteration {iteration}] Accepted: {n_accepted_this_batch}/{batch_size} | "
              f"Total accepted: {len(accepted_samples)}/{n_target_samples} | "
              f"Acceptance rate: {current_acceptance_rate:.3f} | "
              f"Avg frac outside: {avg_frac_outside:.3f}")
        print()

        # Save snapshot
        if snapshot_interval > 0 and len(accepted_samples) % snapshot_interval < batch_size:
            snapshot_path = os.path.join(
                output_dir,
                f'snapshot_{len(accepted_samples)}_samples.npy'
            )
            np.save(snapshot_path, np.array(accepted_samples))
            tqdm.write(f"Saved snapshot: {snapshot_path}")

    pbar.close()

    # Final statistics
    stats['total_generated'] = total_generated
    stats['total_accepted'] = len(accepted_samples)
    stats['final_acceptance_rate'] = len(accepted_samples) / total_generated if total_generated > 0 else 0
    stats['iterations'] = iteration

    print(f"\nRejection sampling completed:")
    print(f"  Total generated: {total_generated}")
    print(f"  Total accepted: {len(accepted_samples)}")
    print(f"  Final acceptance rate: {stats['final_acceptance_rate']:.3f}")
    print(f"  Iterations: {iteration}")

    if iteration >= max_iterations:
        print(f"  WARNING: Reached maximum iterations ({max_iterations})")

    return np.array(accepted_samples), stats


def visualize_results(target_dataset, target_heatmap, synthetic_samples,
                     heatmap_mask, stats, output_path):
    """
    Create visualization comparing target and synthetic distributions.

    Args:
        target_dataset: Original target dataset (scaled to [0,1])
        target_heatmap: Heatmap of target dataset
        synthetic_samples: Accepted synthetic samples
        heatmap_mask: Binary mask used for rejection
        stats: Sampling statistics dictionary
        output_path: Path to save visualization
    """
    # Generate synthetic heatmap
    all_synth_points = np.concatenate(synthetic_samples)
    res_y, res_x = target_heatmap.shape

    synth_heatmap, _, _ = np.histogram2d(
        all_synth_points[:, 0],
        all_synth_points[:, 1],
        bins=(res_y, res_x),
        range=((0, 1), (0, 1)),
        density=False
    )

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Heatmaps
    im1 = axes[0, 0].imshow(np.log1p(target_heatmap), origin='lower',
                            extent=[0, 1, 0, 1], cmap='viridis')
    axes[0, 0].set_title('Target Dataset Heatmap (log scale)')
    axes[0, 0].set_xlabel('Longitude (normalized)')
    axes[0, 0].set_ylabel('Latitude (normalized)')
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(np.log1p(synth_heatmap), origin='lower',
                            extent=[0, 1, 0, 1], cmap='viridis')
    axes[0, 1].set_title('Synthetic Samples Heatmap (log scale)')
    axes[0, 1].set_xlabel('Longitude (normalized)')
    axes[0, 1].set_ylabel('Latitude (normalized)')
    plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[0, 2].imshow(heatmap_mask, origin='lower',
                            extent=[0, 1, 0, 1], cmap='gray')
    axes[0, 2].set_title('Rejection Mask (white = valid)')
    axes[0, 2].set_xlabel('Longitude (normalized)')
    axes[0, 2].set_ylabel('Latitude (normalized)')
    plt.colorbar(im3, ax=axes[0, 2])

    # Row 2: Trajectories and statistics
    # Sample trajectories for visualization
    n_vis_trajs = min(50, len(target_dataset))
    for traj in target_dataset[:n_vis_trajs]:
        axes[1, 0].plot(traj[:, 1], traj[:, 0], alpha=0.3, linewidth=0.5)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title(f'Target Trajectories (n={n_vis_trajs})')
    axes[1, 0].set_xlabel('Longitude (normalized)')
    axes[1, 0].set_ylabel('Latitude (normalized)')
    axes[1, 0].set_aspect('equal')

    n_vis_synth = min(50, len(synthetic_samples))
    for traj in synthetic_samples[:n_vis_synth]:
        axes[1, 1].plot(traj[:, 1], traj[:, 0], alpha=0.3, linewidth=0.5)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title(f'Synthetic Trajectories (n={n_vis_synth})')
    axes[1, 1].set_xlabel('Longitude (normalized)')
    axes[1, 1].set_ylabel('Latitude (normalized)')
    axes[1, 1].set_aspect('equal')

    # Acceptance rate over time
    if len(stats['acceptance_rate_history']) > 0:
        axes[1, 2].plot(stats['iteration_history'], stats['acceptance_rate_history'])
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].set_ylabel('Cumulative Acceptance Rate')
        axes[1, 2].set_title('Acceptance Rate History')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].axhline(y=stats['final_acceptance_rate'],
                          color='r', linestyle='--',
                          label=f'Final: {stats["final_acceptance_rate"]:.3f}')
        axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        prog='sample_unconditional_rejection',
        description='Generate synthetic trajectories using rejection sampling on unconditional diffusion model'
    )

    parser.add_argument('dataset',
                       choices=['geolife-25', 'geolife-100', 'porto-25', 'porto-100',
                               'cabspotting-25', 'cabspotting-100', 'brest-100'],
                       help='Target dataset name')
    parser.add_argument('model_weights',
                       help='Path to unconditional model weights (.pth file)')
    parser.add_argument('--n-samples', type=int, default=10000,
                       help='Target number of accepted samples (default: 10000)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for sample generation (default: 256)')
    parser.add_argument('--snapshot-interval', type=int, default=1000,
                       help='Save snapshot every N samples (default: 1000, 0=disable)')
    parser.add_argument('--max-iterations', type=int, default=300,
                       help='Maximum sampling iterations (default: 300; 10,000 accepted samples took up to 128 batches of 256)')
    parser.add_argument('--reject-threshold', type=float, default=0.1,
                       help='Maximum fraction of points allowed outside valid regions (default: 0.1)')
    parser.add_argument('--pixels-per-km', type=float, default=21.33,
                       help='Target heatmap resolution in pixels per km (default: 21.33, matching 64x64 for 3x3km)')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for results (default: current directory)')
    parser.add_argument('--seq-len', type=int, default=128,
                       help='Sequence length (default: 128)')
    parser.add_argument('--noise-scale', type=float, default=1.0,
                       help='Noise scale used during model training (default: 1.0)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("Unconditional Rejection Sampling")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Model weights: {args.model_weights}")
    print(f"Target samples: {args.n_samples}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)

    # Load target dataset
    print("\nLoading target dataset...")
    start_time = time.time()
    target_dataset = load_dataset_from_str(args.dataset, prefix='../data/')

    # Scale to [0, 1]
    target_dataset_scaled = minmaxscale(target_dataset)
    print(f"Loaded {len(target_dataset_scaled)} trajectories")
    print(f"Time: {time.time() - start_time:.2f}s")

    # Compute heatmap resolution (use scaled dataset in [0,1] space)
    print("\nComputing heatmap resolution...")
    res_y, res_x = compute_heatmap_resolution(target_dataset_scaled, args.dataset, args.pixels_per_km)

    # Generate target heatmap
    print("\nGenerating target dataset heatmap...")
    start_time = time.time()
    target_heatmap, heatmap_mask = generate_target_heatmap(
        target_dataset_scaled, res_y, res_x
    )
    print(f"Time: {time.time() - start_time:.2f}s")

    # Load model
    print("\nLoading unconditional diffusion model...")
    print(f"  Model path: {args.model_weights}")
    start_time = time.time()

    # Model hyperparameters (from train_unconditional.py)
    n_dim = 2
    hidden_dim = 128
    seq_len = args.seq_len
    n_heads = 2
    n_layers = 8
    timesteps = 500
    beta_schedule = 'cosine'
    noise_scale = args.noise_scale

    print(f"  Creating model architecture...")
    model = TransformerDenoiser(
        dim_data=n_dim,
        dim_embedding=hidden_dim,
        seq_len=seq_len,
        n_heads=n_heads,
        n_layers=n_layers
    )

    print(f"  Creating diffusion wrapper (timesteps={timesteps}, noise_scale={noise_scale})...")
    diffusion = GaussianDiffusion1D(
        model,
        seq_length=seq_len,
        timesteps=timesteps,
        beta_schedule=beta_schedule,
        noise_scale=noise_scale
    )

    # Load checkpoint
    print(f"  Loading checkpoint weights...")
    checkpoint = torch.load(args.model_weights)
    diffusion.load_state_dict(checkpoint['diffusion_state_dict'])

    print(f"  Moving model to CUDA...")
    diffusion.cuda()
    diffusion.eval()

    print(f"✓ Model loaded successfully")
    print(f"  Time: {time.time() - start_time:.2f}s")

    # Perform rejection sampling
    print("\n" + "="*80)
    accepted_samples, stats = rejection_sample_unconditional(
        diffusion=diffusion,
        heatmap_mask=heatmap_mask,
        n_target_samples=args.n_samples,
        batch_size=args.batch_size,
        reject_threshold=args.reject_threshold,
        snapshot_interval=args.snapshot_interval,
        max_iterations=args.max_iterations,
        output_dir=args.output_dir
    )

    # Save final results
    output_prefix = os.path.join(args.output_dir,
                                f'unconditional_rejection_{args.dataset}')

    samples_path = f'{output_prefix}_samples.npy'
    np.save(samples_path, accepted_samples)
    print(f"\nSaved accepted samples to: {samples_path}")

    stats_path = f'{output_prefix}_stats.pkl'
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    print(f"Saved statistics to: {stats_path}")

    # Generate visualization
    print("\nGenerating visualization...")
    viz_path = f'{output_prefix}_visualization.png'
    visualize_results(
        target_dataset=target_dataset_scaled,
        target_heatmap=target_heatmap,
        synthetic_samples=accepted_samples,
        heatmap_mask=heatmap_mask,
        stats=stats,
        output_path=viz_path
    )

    print("\n" + "="*80)
    print("Rejection sampling completed successfully!")
    print("="*80)
    print(f"Final acceptance rate: {stats['final_acceptance_rate']:.3f}")
    print(f"Total samples generated: {stats['total_generated']}")
    print(f"Total samples accepted: {stats['total_accepted']}")
    print(f"Output files:")
    print(f"  - Samples: {samples_path}")
    print(f"  - Statistics: {stats_path}")
    print(f"  - Visualization: {viz_path}")
    print("="*80)


if __name__ == '__main__':
    main()
