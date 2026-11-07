import torch
import numpy as np
from denoising_diffusion_pytorch import GaussianDiffusion1D
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys
sys.path.append('../')
from data import load_dataset_from_str
from preproc import InterpolatingHeatmapper

from dataset import ConditionalDataset
from data_generation import extract_signals, region_occupancy
from model import TransformerDenoiser
from evaluation import generate_heatmap, undo_rotation_and_scaling, unscaled_occupancy
from tqdm import tqdm
import pickle
import os

import argparse
import time

script_start_time = time.time()

parser = argparse.ArgumentParser(
            prog='sample',
            description='Uses a conditional model trained on the dataset specified in the `from` argument to synthesize a trajectory based on aggregate information from the dataset specified by the `to` argument')
_dataset_choices = [
    'geolife-25', 'geolife-100', 'porto-25', 'porto-100', 'cabspotting-25', 'cabspotting-100',
    'geolife-100-no-map-match', 'porto-100-no-map-match', 'cabspotting-100-no-map-match',
    # Corner variants for 4-corner generalization experiment
    'geolife-25-bl', 'geolife-25-br', 'geolife-25-tl', 'geolife-25-tr',
    'porto-25-bl', 'porto-25-br', 'porto-25-tl', 'porto-25-tr',
    'cabspotting-25-bl', 'cabspotting-25-br', 'cabspotting-25-tl', 'cabspotting-25-tr',
    # Maritime AIS
    'brest-100',
    'brest-25-bl', 'brest-25-br', 'brest-25-tl', 'brest-25-tr',
]
parser.add_argument('dataset_from', choices=_dataset_choices)
parser.add_argument('dataset_to', choices=_dataset_choices)
parser.add_argument('--region-size', default=3000, type=int,
                    help='Side of the conditioning region in metres; must match the model (the released models use 3000)')
parser.add_argument('--n-samples', default=10000, type=int )
parser.add_argument('--run-id', default=None)
parser.add_argument('--dropout', default=0.0, type=float, help='Dropout percentage for heatmap (0.0-1.0)')
parser.add_argument('--res', type=int, default=64, help='Heatmap resolution (default: 64)')
args = parser.parse_args()


dataset_str = args.dataset_from
target_dataset_str = args.dataset_to
target_dataset_size = args.n_samples
exp_str = f'{dataset_str}-to-{target_dataset_str}'
# Seed runs (--run-id seed<N>, from train_conditional.py --seed) always name the region size, also at 1000.
seed_run = args.run_id is not None and args.run_id.startswith('seed')
if args.region_size != 1000 or seed_run:
    exp_str += f'-region-size-{args.region_size}'
if target_dataset_size != 10000:
    exp_str += f'-n_samples-{target_dataset_size}'
if args.run_id is not None:
    exp_str += f'run_id-{args.run_id}'
if args.dropout > 0.0:
    exp_str += f'-dropout-{int(args.dropout * 100)}'
if args.res != 64:
    exp_str += f'-{args.res}res'

print(exp_str, 'started!')

unconditional_dataset = load_dataset_from_str(target_dataset_str, prefix='../data/')

min_x, min_y = unconditional_dataset.min(axis=(0, 1))
max_x, max_y = unconditional_dataset.max(axis=(0, 1))

original_region = (min_x, max_x, min_y, max_y)

batch_size = 8
num_workers = 10
n_dim = 2
beta_schedule = 'cosine'
adam_betas = (0.9, 0.99)
seq_len = 128
save_n = 100000
region_size = args.region_size
heatmap_res = args.res

# from hyperparameter search
hidden_dim = 128
timesteps = 500
n_layers = 8
n_heads = 2

# Build checkpoint path with resolution suffix
res_suffix = f'-{heatmap_res}res' if heatmap_res != 64 else ''

# For resolution experiment models, always include region_size and save_n when res != 64
if heatmap_res != 64:
    if args.run_id is not None:
        # conditional-geolife-100-100000-3000-16res-<run_id>.pth
        checkpoint_path = f'model_weights/conditional-{dataset_str}-100000-{region_size}{res_suffix}-{args.run_id}.pth'
    else:
        # conditional-geolife-100-100000-1000-16res.pth
        checkpoint_path = f'model_weights/conditional-{dataset_str}-100000-{region_size}{res_suffix}.pth'
# For legacy 64x64 models
elif args.run_id is not None and (seed_run or (region_size != 1000 and os.path.exists(f'model_weights/conditional-{dataset_str}-100000-{region_size}-{args.run_id}.pth'))):
    # conditional-geolife-100-100000-3000-seed42.pth (train_conditional.py --seed 42), conditional-geolife-100-100000-3000-<run_id>.pth
    checkpoint_path = f'model_weights/conditional-{dataset_str}-100000-{region_size}-{args.run_id}.pth'
elif args.run_id is None and region_size != 1000:
    # conditional-geolife-100-no-map-match-100000-3000.pth
    checkpoint_path = f'model_weights/conditional-{dataset_str}-100000-{region_size}.pth'
elif args.run_id is None:
    # conditional-geolife-100.pth
    checkpoint_path = f'model_weights/conditional-{dataset_str}.pth'
else:
    checkpoint_path = f'model_weights/conditional-{dataset_str}-{args.run_id}.pth'


model = TransformerDenoiser(n_dim, hidden_dim, seq_len=seq_len, n_heads=n_heads, n_layers=n_layers, heatmap_resolution=heatmap_res)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = seq_len,
    timesteps = timesteps,
    beta_schedule = beta_schedule,
)

checkpoint = torch.load(checkpoint_path, weights_only=False)

diffusion.load_state_dict(checkpoint['diffusion_state_dict'])
print(f"Loaded model weights from checkpoint {checkpoint_path}")
diffusion.cuda()
diffusion.eval()

windowed_subset_save_path = f'../data/preprocessed/'

# Build dataset path with resolution suffix
if heatmap_res != 64:
    dataset_path = os.path.join(windowed_subset_save_path, f'{dataset_str}-{save_n}-conditional-128-{region_size}-{heatmap_res}res.npz')
else:
    dataset_path = os.path.join(windowed_subset_save_path, f'{dataset_str}-{save_n}-conditional-128-{region_size}.npz')

print(f"Using dataset ({heatmap_res}x{heatmap_res}): {dataset_path}")
dataset = ConditionalDataset(dataset_path)
print(f"Heatmapper type: {dataset.heatmapper_type}")

# Use InterpolatingHeatmapper when training data was generated with it
if dataset.heatmapper_type == 'interpolating':
    interpolating_heatmapper = InterpolatingHeatmapper(
        unconditional_dataset,
        meters_per_cell=dataset.heatmapper_meters_per_cell,
        sigma=dataset.heatmapper_sigma,
        device='cpu')
else:
    interpolating_heatmapper = None

def sample_heatmap(diffusion, query_heatmap, n_samples, sample_batch_size=256, dropout=0.0):
    """
    Given a diffusion model and a heatmap, generate n_samples conditioned on the heatmap.
    """
    sample_batch_size = min(sample_batch_size, n_samples)

    query_heatmap = torch.tensor(query_heatmap).unsqueeze(0)
    all_samples = []

    n_batches = n_samples // sample_batch_size
    remainder = n_samples % sample_batch_size

    def sample(n_samples):
        condition = query_heatmap.unsqueeze(0).expand(n_samples, 1, heatmap_res, heatmap_res).float().cuda()
        # Apply dropout before normalization if specified
        if dropout > 0.0:
            dropout_mask = torch.bernoulli(torch.ones_like(condition) * (1 - dropout))
            condition = condition * dropout_mask
        condition = (condition - dataset.heatmaps_mean) / dataset.heatmaps_std
        # Sample `n_samples` number of samples from the model
        samples = diffusion.sample_conditional(condition, n_samples)
        # Convert to numpy
        samples = samples.cpu().numpy()

        return samples

    for i in range(n_batches):
        all_samples.append(sample(sample_batch_size))

    if remainder > 0:
        all_samples.append(sample(remainder))

    all_samples = np.concatenate(all_samples, axis=0)

    return all_samples

min_lat, max_lat, min_lon, max_lon = original_region
map_height = max_lat - min_lat
map_width = max_lon - min_lon

width = region_size
height = region_size
rotation = 0

regions = []

latitudes = np.arange(min_lat, max_lat, height / 2)
longitudes = np.arange(min_lon, max_lon, width / 2)
indices = []

for i, region_lon in enumerate(longitudes):
    for j, region_lat in enumerate(latitudes):
        regions.append((region_lat, region_lon, rotation, height, width))
        indices.append((i, j))

indices[:10]

max_lat_actual = latitudes[-1] + height / 2
max_lon_actual = longitudes[-1] + width / 2

heatmap_data, xedges, yedges = unscaled_occupancy(unconditional_dataset, 512, 512)

plt.imshow(np.log(heatmap_data), extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]], origin='lower', interpolation='nearest')

for region_lat, region_lon, rotation, height, width in regions:
    rect = patches.Rectangle((region_lon, region_lat), width, height, angle=rotation, linewidth=1, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)

plt.savefig(f'{exp_str}_regions_displayed_over_heatmap.png')

# heatmap_data, xedges, yedges = unscaled_occupancy(unconditional_dataset, len(longitudes), len(latitudes))
all_concatenated = np.concatenate(unconditional_dataset)
heatmap_data, xedges, yedges = np.histogram2d(all_concatenated[:, 0], all_concatenated[:, 1], 
                                              bins=(len(latitudes), len(longitudes)),
                                              range=((min_lat, max_lat_actual),
                                                     (min_lon, max_lon_actual)))


plt.imshow(np.log(heatmap_data), extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]], origin='lower', interpolation=None)

for region_lat, region_lon, rotation, height, width in regions:
    rect = patches.Rectangle((region_lon, region_lat), width, height, angle=rotation, linewidth=1, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)

plt.colorbar()
plt.xlim(0)

plt.savefig(f'{exp_str}_subregion_counts.png')

counts = []

for (i, j), (region_lat, region_lon, rotation, height, width) in zip(indices, regions):
    counts.append(int(heatmap_data[j:(j+2), i:(i+2)].sum()))

counts = np.array(counts)

scaled_counts_float = (target_dataset_size * counts / counts.sum()).round()
scaled_counts_float.sum()

scaled_counts = np.ceil(target_dataset_size * counts / counts.sum()).astype('int')


def sample_synthetic_region(diffusion, dataset, region, n_samples, sample_batch_size=64, dropout=0.0):
    """
    Extract subtrajectories from region, creates a heatmap of the real data, creates synthetic
    trajectories conditioned on the heatmap and finally calculate a heatmap of the synthetic samples.
    """
    assert n_samples >= sample_batch_size

    trajectories_in_region = extract_signals(dataset, region)

    if len(trajectories_in_region) == 0:
        return []

    if interpolating_heatmapper is not None:
        heatmap = interpolating_heatmapper.get_heatmap(region, res=heatmap_res)
    else:
        heatmap, _, _ = region_occupancy(trajectories_in_region, res=heatmap_res)

    samples = sample_heatmap(diffusion, heatmap, n_samples=n_samples, sample_batch_size=sample_batch_size, dropout=dropout)
    synthetic_heatmap = generate_heatmap(samples, res=heatmap_res)

    return trajectories_in_region, heatmap, samples, synthetic_heatmap

results = []

setup_end_time = time.time()
print(f"Setup took {setup_end_time - script_start_time:.2f} seconds")

start_time = time.time()
for scaled_count, region in tqdm(zip(scaled_counts, regions), total=len(regions)):
    n_sample = scaled_count

    if n_sample > 0:
        # print(n_sample)
        batch_size = min(256, n_sample)
        results.append(sample_synthetic_region(diffusion, unconditional_dataset, region, n_sample, batch_size, dropout=args.dropout))
    else:
        results.append([])

end_time = time.time()
print(f"Sampling took {end_time - start_time:.2f} seconds")

post_processing_start_time = time.time()

filename = f'{exp_str}.pickle'
with open(filename, 'wb') as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

# TODO: Read the file
# with open(filename, 'rb') as f:
#     results = pickle.load(f)

nonzero_results = []
nonzero_regions = []

for result, region in zip(results, regions):
    if len(result) != 0:
        nonzero_results.append(result)
        nonzero_regions.append(region)

restored_originals = []
restored_samples = []

for (trajectories_in_region, heatmap, samples, synthetic_heatmap), region in zip(nonzero_results, nonzero_regions):
    # for traj in trajectories_in_region:
        # restored_originals.append(undo_rotation_and_scaling(traj, region))
    
    for traj in samples:
        restored_samples.append(undo_rotation_and_scaling(traj.T, region))

restored_samples = np.stack(restored_samples)

target_region_actual = [unconditional_dataset[:, :, 0].min(), unconditional_dataset[:, :, 0].max(), unconditional_dataset[:, :, 1].min(), unconditional_dataset[:, :, 1].max()]
target_region_actual

original_region

restored_scaled_samples = restored_samples - [target_region_actual[0], target_region_actual[2]]

restored_scaled_samples = restored_scaled_samples / [target_region_actual[1] - target_region_actual[0],
                                                     target_region_actual[3] - target_region_actual[2]]

is_outside = np.any(restored_scaled_samples > 1, axis=(1, 2))
restored_scaled_samples_filtered = restored_scaled_samples[~is_outside]
print(f'{len(restored_scaled_samples_filtered)=}')
print(f'{restored_scaled_samples_filtered.max(axis=(0, 1))=}\n{restored_scaled_samples_filtered.min(axis=(0, 1))=}')

# heatmap_data, xedges, yedges = unscaled_occupancy(unconditional_dataset, len(longitudes), len(latitudes))
all_concatenated = np.concatenate(restored_samples)
heatmap_data, xedges, yedges = np.histogram2d(all_concatenated[:, 0], all_concatenated[:, 1], 
                                              bins=(len(latitudes), len(longitudes)),
                                              range=((min_lat, max_lat_actual),
                                                     (min_lon, max_lon_actual)))


plt.imshow(np.log(heatmap_data), extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]], origin='lower', interpolation=None)

for region_lat, region_lon, rotation, height, width in regions:
    rect = patches.Rectangle((region_lon, region_lat), width, height, angle=rotation, linewidth=1, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)

plt.colorbar()
plt.xlim(0)

plt.savefig(f'{exp_str}_synthetic_subregion_counts.png')

np.save(f'tddpm-conditional-{exp_str}.npy', restored_scaled_samples_filtered)

global_heatmap, xedges, yedges = unscaled_occupancy(unconditional_dataset, 1024, 1024)
# global_heatmap, xedges, yedges = unscaled_occupancy(restored_originals, 1024, 1024)
global_synth_heatmap, xedges, yedges = unscaled_occupancy(restored_samples, 1024, 1024)

f, axs = plt.subplots(ncols=2, nrows=2, figsize=(40, 40))
axs = [ax for row in axs for ax in row]

axs[1].imshow(np.log(global_heatmap), extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]], origin='lower', interpolation='nearest')
# axs[1].imshow(np.log(global_heatmap), extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]], origin='lower', interpolation=None)

for region_lat, region_lon, rotation, height, width in regions:
    rect = patches.Rectangle((region_lon, region_lat), width, height, angle=rotation, linewidth=1, edgecolor='r', facecolor='none')
    axs[1].add_patch(rect)

# for traj in restored_originals:
for traj in unconditional_dataset:
    axs[0].plot(traj[:, 1], traj[:, 0], alpha=1)

for traj in restored_samples:
    axs[2].plot(traj[:, 1], traj[:, 0], alpha=1)

axs[3].imshow(np.log(global_synth_heatmap), extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]], origin='lower', interpolation='nearest')

plt.savefig(f'{exp_str}.png')

script_end_time = time.time()
print(f"Post-processing took {script_end_time - post_processing_start_time:.2f} seconds")
print(f"Total script execution time: {script_end_time - script_start_time:.2f} seconds")



