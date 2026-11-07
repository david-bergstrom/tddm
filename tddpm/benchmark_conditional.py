import torch
import numpy as np
from denoising_diffusion_pytorch import GaussianDiffusion1D
import time
import pickle
import os
import argparse

import sys
sys.path.append('../')
from data import load_dataset_from_str

from dataset import ConditionalDataset
from model import TransformerDenoiser

# Parse arguments
parser = argparse.ArgumentParser(
    prog='benchmark_conditional',
    description='Benchmark TDDM conditional sampling for memory and time performance')
parser.add_argument('dataset_from', choices=['geolife-25', 'geolife-100', 'porto-25', 'porto-100', 'cabspotting-25', 'cabspotting-100'])
parser.add_argument('dataset_to', choices=['geolife-25', 'geolife-100', 'porto-25', 'porto-100', 'cabspotting-25', 'cabspotting-100'])
parser.add_argument('--region-size', default=3000, type=int, help='Region size in meters (default: 3000 for 3x3 km)')
parser.add_argument('--run-id', default=None)
args = parser.parse_args()

# Default run IDs of the released models for the 100% datasets
DEFAULT_RUN_IDS = {
    'geolife-100': 'ku7c6kiw',
    'porto-100': 'sc7ukq34',
    'cabspotting-100': 'vctqwp9s'
}

# Use default run-id if not specified and dataset has a default
if args.run_id is None and args.dataset_from in DEFAULT_RUN_IDS:
    args.run_id = DEFAULT_RUN_IDS[args.dataset_from]
    print(f"Using default run-id for {args.dataset_from}: {args.run_id}")

print(f"Benchmarking {args.dataset_from} -> {args.dataset_to}")

# Setup parameters (matching sample.py)
dataset_str = args.dataset_from
target_dataset_str = args.dataset_to
region_size = args.region_size

n_dim = 2
beta_schedule = 'cosine'
seq_len = 128
save_n = 100000
heatmap_res = 64  # Fixed to 64x64 for benchmarking

# Model hyperparameters (from hyperparameter search)
hidden_dim = 128
timesteps = 500
n_layers = 8
n_heads = 2

# Determine checkpoint path
if args.run_id is not None:
    checkpoint_path = f'model_weights/conditional-{dataset_str}-{args.run_id}.pth'
else:
    checkpoint_path = f'model_weights/conditional-{dataset_str}.pth'

print(f"Loading model from {checkpoint_path}")

# Load model
model = TransformerDenoiser(n_dim, hidden_dim, seq_len=seq_len, n_heads=n_heads, n_layers=n_layers)

diffusion = GaussianDiffusion1D(
    model,
    seq_length=seq_len,
    timesteps=timesteps,
    beta_schedule=beta_schedule,
)

checkpoint = torch.load(checkpoint_path, weights_only=False)
diffusion.load_state_dict(checkpoint['diffusion_state_dict'])
print(f"Loaded model weights from checkpoint")

diffusion.cuda()
diffusion.eval()

# Load dataset for conditioning
dataset_path = os.path.join('../data/preprocessed/', f'{dataset_str}-{save_n}-conditional-128-{region_size}.npz')
print(f"Loading dataset from {dataset_path}")
dataset = ConditionalDataset(dataset_path)

# Load target dataset for marginal distribution
print(f"Loading target dataset {target_dataset_str}")
unconditional_dataset = load_dataset_from_str(target_dataset_str, prefix='../data/')

print("\n" + "="*60)
print("BENCHMARK 1: Marginal Distribution Calculation")
print("="*60)

# Benchmark marginal distribution calculation time
torch.cuda.synchronize()
start_time = time.time()

# Calculate histogram (matching lines 189-193 in sample.py)
min_x, min_y = unconditional_dataset.min(axis=(0, 1))
max_x, max_y = unconditional_dataset.max(axis=(0, 1))

# Simple grid calculation for marginal distribution
latitudes = np.arange(min_x, max_x, region_size / 2)
longitudes = np.arange(min_y, max_y, region_size / 2)

max_lat_actual = latitudes[-1] + region_size / 2
max_lon_actual = longitudes[-1] + region_size / 2

all_concatenated = np.concatenate(unconditional_dataset)
heatmap_data, xedges, yedges = np.histogram2d(
    all_concatenated[:, 0],
    all_concatenated[:, 1],
    bins=(len(latitudes), len(longitudes)),
    range=((min_x, max_lat_actual), (min_y, max_lon_actual))
)

torch.cuda.synchronize()
marginal_dist_time = time.time() - start_time

print(f"Marginal distribution calculation time: {marginal_dist_time:.4f} seconds")
print(f"Grid size: {len(latitudes)} x {len(longitudes)}")

# Create a sample heatmap for conditioning (use first from dataset)
sample_heatmap = dataset.heatmaps[0]  # Already normalized

print("\n" + "="*60)
print("WARMUP: Running warmup sample")
print("="*60)

# Warmup run
with torch.no_grad():
    condition = sample_heatmap.unsqueeze(0).float().cuda()
    _ = diffusion.sample_conditional(condition, 1)
    torch.cuda.synchronize()

print("Warmup complete")

# Function to benchmark sampling
def benchmark_sampling(batch_size, n_samples=10):
    """
    Benchmark sampling with given batch_size.
    Returns peak VRAM in MB and time per sample in seconds.
    """
    times = []
    vram_peaks = []

    for i in range(n_samples):
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start = time.time()
        with torch.no_grad():
            condition = sample_heatmap.unsqueeze(0).expand(batch_size, 1, heatmap_res, heatmap_res).float().cuda()
            samples = diffusion.sample_conditional(condition, batch_size)

        torch.cuda.synchronize()
        elapsed = time.time() - start

        # Get peak memory in bytes, convert to MB
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)

        times.append(elapsed)
        vram_peaks.append(peak_memory)

    # Calculate statistics
    times = np.array(times)
    vram_peaks = np.array(vram_peaks)

    avg_time_per_sample = times.mean() / batch_size
    avg_peak_vram = vram_peaks.mean()

    return avg_peak_vram, avg_time_per_sample, times, vram_peaks

print("\n" + "="*60)
print("BENCHMARK 2: Single Sample (batch_size=1)")
print("="*60)

single_vram, single_time, single_times, single_vrams = benchmark_sampling(1, n_samples=10)
print(f"Peak VRAM: {single_vram:.2f} MB")
print(f"Time per sample: {single_time:.4f} seconds")
print(f"  (std: {(single_times.std() / 1):.4f}s)")

# Benchmark batched sampling
batch_sizes = [8, 64, 256]
batched_results = {}

for bs in batch_sizes:
    print("\n" + "="*60)
    print(f"BENCHMARK 3: Batched Sampling (batch_size={bs})")
    print("="*60)

    vram, time_per_sample, times, vrams = benchmark_sampling(bs, n_samples=10)
    batched_results[bs] = {
        'peak_vram_mb': vram,
        'time_per_sample': time_per_sample,
        'raw_times': times,
        'raw_vrams': vrams
    }

    print(f"Peak VRAM: {vram:.2f} MB")
    print(f"Time per sample: {time_per_sample:.4f} seconds")
    print(f"  (std: {(times.std() / bs):.4f}s)")

# Prepare results dictionary with numpy types
results = {
    'marginal_dist_time': np.float64(marginal_dist_time),
    'single_sample': {
        'peak_vram_mb': np.float64(single_vram),
        'time_per_sample': np.float64(single_time),
        'raw_times': single_times,
        'raw_vrams': single_vrams
    },
    'batched': {}
}

for bs, data in batched_results.items():
    results['batched'][bs] = {
        'peak_vram_mb': np.float64(data['peak_vram_mb']),
        'time_per_sample': np.float64(data['time_per_sample']),
        'raw_times': data['raw_times'],
        'raw_vrams': data['raw_vrams']
    }

# Save results
exp_str = f'{args.dataset_from}-to-{args.dataset_to}'
if args.region_size != 3000:
    exp_str += f'-region-size-{args.region_size}'
if args.run_id is not None:
    exp_str += f'-run_id-{args.run_id}'

filename = f'benchmark-{exp_str}.pickle'
with open(filename, 'wb') as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Results saved to: {filename}")
print(f"\nMarginal distribution time: {marginal_dist_time:.4f}s")
print(f"\nSingle sample:")
print(f"  VRAM: {single_vram:.2f} MB")
print(f"  Time: {single_time:.4f}s")
print(f"\nBatched samples:")
for bs in batch_sizes:
    data = results['batched'][bs]
    print(f"  Batch size {bs}:")
    print(f"    VRAM: {data['peak_vram_mb']:.2f} MB")
    print(f"    Time per sample: {data['time_per_sample']:.4f}s")
