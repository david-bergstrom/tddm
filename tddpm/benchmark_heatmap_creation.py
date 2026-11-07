import numpy as np
import time
import pickle
import os
import argparse
import sys
sys.path.append('../')

from preproc import Heatmapper
from data_generation import extract_signals, region_occupancy, random_region

# Parse arguments
parser = argparse.ArgumentParser(
    prog='benchmark_heatmap_creation',
    description='Benchmark heatmap creation methods for memory and time performance')
parser.add_argument('dataset', choices=['geolife-25', 'geolife-100', 'porto-25', 'porto-100', 'cabspotting-25', 'cabspotting-100'])
parser.add_argument('--region-size', default=3000, type=int, help='Region size in meters (default: 3000 for 3x3 km)')
args = parser.parse_args()

print(f"Benchmarking heatmap creation for {args.dataset}")
print(f"Region size: {args.region_size}m x {args.region_size}m")

# Load dataset
dataset_str = args.dataset
region_size = args.region_size
heatmap_res = 64

# Load preprocessed dataset
save_n = 100000
dataset_path = os.path.join('../data/preprocessed/', f'{dataset_str}-{save_n}.npy')
print(f"Loading dataset from {dataset_path}")
dataset = np.load(dataset_path).astype(np.float32)

# Calculate dataset bounds
min_x, min_y = dataset.min(axis=(0, 1))
max_x, max_y = dataset.max(axis=(0, 1))
bounds = [min_x, max_x, min_y, max_y]

print(f"Dataset bounds: lat [{min_x:.2f}, {max_x:.2f}], lon [{min_y:.2f}, {max_y:.2f}]")

# Generate random regions for benchmarking
# Use different counts to test scalability
region_counts = [10, 50, 100, 500]
max_regions = max(region_counts)

print(f"\nGenerating {max_regions} random regions...")
np.random.seed(42)  # For reproducibility
regions = []
for i in range(max_regions):
    region = random_region(size=(region_size, region_size), bounds=bounds, max_rotation=360)
    regions.append(region)

print("="*60)
print("BENCHMARK 1: Heatmapper Initialization")
print("="*60)

# Benchmark Heatmapper initialization
start_time = time.time()
heatmapper = Heatmapper(dataset, cells=100)
init_time = time.time() - start_time

print(f"Heatmapper initialization time: {init_time:.4f} seconds")
print(f"Number of cells in spatial tree: {len(heatmapper.tree)}")

# Function to benchmark Heatmapper.get_heatmap()
def benchmark_heatmapper(heatmapper, regions, n_regions):
    """
    Benchmark Heatmapper.get_heatmap() for n_regions.
    Returns average time per heatmap.
    """
    regions_subset = regions[:n_regions]

    start = time.time()
    heatmaps = []
    for region in regions_subset:
        heatmap = heatmapper.get_heatmap(region, res=heatmap_res)
        heatmaps.append(heatmap)
    elapsed = time.time() - start

    avg_time_per_heatmap = elapsed / n_regions

    # Count how many heatmaps were empty (NaN)
    empty_count = sum(1 for hm in heatmaps if np.isnan(hm).all())

    return avg_time_per_heatmap, elapsed, empty_count

# Function to benchmark extract_signals + region_occupancy
def benchmark_original(dataset, regions, n_regions):
    """
    Benchmark original extract_signals + region_occupancy approach.
    Returns average time per heatmap, separate times for extraction and occupancy.
    """
    regions_subset = regions[:n_regions]

    extraction_times = []
    occupancy_times = []
    empty_count = 0

    for region in regions_subset:
        # Time trajectory extraction
        start = time.time()
        trajectories_in_region = extract_signals(dataset, region, min_len=24)
        extraction_time = time.time() - start
        extraction_times.append(extraction_time)

        # Time heatmap creation
        if len(trajectories_in_region) > 0:
            start = time.time()
            heatmap, _, _ = region_occupancy(trajectories_in_region, res=heatmap_res)
            occupancy_time = time.time() - start
            occupancy_times.append(occupancy_time)
        else:
            empty_count += 1
            occupancy_times.append(0)

    total_extraction = sum(extraction_times)
    total_occupancy = sum(occupancy_times)
    total_time = total_extraction + total_occupancy

    avg_time_per_heatmap = total_time / n_regions
    avg_extraction = total_extraction / n_regions
    avg_occupancy = total_occupancy / n_regions

    return avg_time_per_heatmap, total_time, avg_extraction, avg_occupancy, empty_count

# Run benchmarks for different region counts
heatmapper_results = {}
original_results = {}

for n_regions in region_counts:
    print("\n" + "="*60)
    print(f"BENCHMARK 2: Heatmapper with {n_regions} regions")
    print("="*60)

    avg_time, total_time, empty_count = benchmark_heatmapper(heatmapper, regions, n_regions)
    heatmapper_results[n_regions] = {
        'avg_time_per_heatmap': avg_time,
        'total_time': total_time,
        'empty_count': empty_count
    }

    print(f"Total time: {total_time:.4f}s")
    print(f"Average time per heatmap: {avg_time:.6f}s ({avg_time*1000:.3f}ms)")
    print(f"Empty regions: {empty_count}/{n_regions} ({100*empty_count/n_regions:.1f}%)")

    print("\n" + "="*60)
    print(f"BENCHMARK 3: Original method with {n_regions} regions")
    print("="*60)

    avg_time, total_time, avg_extraction, avg_occupancy, empty_count = benchmark_original(dataset, regions, n_regions)
    original_results[n_regions] = {
        'avg_time_per_heatmap': avg_time,
        'total_time': total_time,
        'avg_extraction_time': avg_extraction,
        'avg_occupancy_time': avg_occupancy,
        'empty_count': empty_count
    }

    print(f"Total time: {total_time:.4f}s")
    print(f"Average time per heatmap: {avg_time:.6f}s ({avg_time*1000:.3f}ms)")
    print(f"  - Extraction: {avg_extraction:.6f}s ({avg_extraction*1000:.3f}ms)")
    print(f"  - Occupancy: {avg_occupancy:.6f}s ({avg_occupancy*1000:.3f}ms)")
    print(f"Empty regions: {empty_count}/{n_regions} ({100*empty_count/n_regions:.1f}%)")

    # Calculate speedup
    if avg_time > 0:
        speedup = original_results[n_regions]['avg_time_per_heatmap'] / heatmapper_results[n_regions]['avg_time_per_heatmap']
        print(f"\nSpeedup: {speedup:.2f}x faster with Heatmapper")

# Prepare results dictionary
results = {
    'dataset': dataset_str,
    'region_size': region_size,
    'heatmap_res': heatmap_res,
    'num_trajectories': len(dataset),
    'heatmapper_init_time': init_time,
    'heatmapper_num_cells': len(heatmapper.tree),
    'heatmapper': heatmapper_results,
    'original': original_results
}

# Save results
filename = f'benchmark-heatmap-{dataset_str}-{region_size}.pickle'
with open(filename, 'wb') as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Results saved to: {filename}")
print(f"\nHeatmapper initialization: {init_time:.4f}s")
print(f"Spatial tree cells: {len(heatmapper.tree)}")

print("\nPer-heatmap performance comparison:")
print(f"{'Regions':<10} {'Heatmapper (ms)':<20} {'Original (ms)':<20} {'Speedup':<10}")
print("-" * 60)
for n_regions in region_counts:
    hm_time = heatmapper_results[n_regions]['avg_time_per_heatmap'] * 1000
    orig_time = original_results[n_regions]['avg_time_per_heatmap'] * 1000
    speedup = orig_time / hm_time if hm_time > 0 else 0
    print(f"{n_regions:<10} {hm_time:<20.3f} {orig_time:<20.3f} {speedup:<10.2f}x")
