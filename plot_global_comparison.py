"""Plot 2x2 global comparison: real vs synthetic trajectories and log-density heatmaps.

Usage:
    python plot_global_comparison.py brest-100
    python plot_global_comparison.py brest-100 --n-samples 10000
    python plot_global_comparison.py geolife-100 porto-100 cabspotting-100
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

from data import load_dataset_from_str
from synthetic_loader import find_synthetic_datasets, load_synthetic
from preprocess_data import meters_from_point
from preproc import get_limit
from config import nice_names


def get_limits_meters(dataset_str):
    limits = get_limit(dataset_str)
    origin = [limits[0], limits[2]]
    limits_meters = meters_from_point(np.array([
        [limits[0], limits[2]],
        [limits[1], limits[3]]
    ]), origin)
    return limits_meters.flatten()


def inverse_scale(scaled, data_range):
    d_min, d_max = data_range
    return scaled * (d_max - d_min) + d_min


def plot_comparison(dataset_str, model, n_samples, save_format):
    all_datasets = find_synthetic_datasets()

    train_data = load_dataset_from_str(dataset_str=dataset_str, prefix='data/')
    d_min = np.min(train_data, axis=(0, 1), keepdims=True)
    d_max = np.max(train_data, axis=(0, 1), keepdims=True)
    data_range = (d_min, d_max)

    select = list(filter(
        lambda x: x.dataset == dataset_str and x.seq_len == 128 and x.model_name == model,
        all_datasets))
    if len(select) == 0:
        print(f'No synthetic data found for {dataset_str} / {model}')
        return

    synthetic_data = load_synthetic(select[0])
    synthetic_data = inverse_scale(synthetic_data, data_range)
    limits = get_limits_meters(dataset_str)

    n_real = min(n_samples, len(train_data))
    n_synth = min(n_samples, len(synthetic_data))

    # Subsample real data randomly for fair visual comparison
    rng = np.random.default_rng(42)
    real_idx = rng.choice(len(train_data), n_real, replace=False)
    train_subset = train_data[real_idx]

    colors = ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6',
              '#ee9b00', '#ca6702', '#bb3e03', '#ae2012', '#9b2226']
    plt.rc('axes', prop_cycle=cycler(color=colors))

    text_size = (5.50107, 9.00177)
    figsize = (text_size[0], text_size[1] * 0.7)
    line_params = {'linewidth': 0.3}

    f, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize, sharex=True, sharey=True)

    # Real trajectories
    for traj in train_subset:
        axs[0, 0].plot(traj[:, 1], traj[:, 0], **line_params)

    # Real heatmap
    train_flat = train_subset.reshape(-1, 2)
    aspect_ratio = (limits[3] - limits[1]) / (limits[2] - limits[0])
    bins_width = 200
    bins = (int(aspect_ratio * bins_width), bins_width)
    hm, _ = np.histogramdd(train_flat, bins=bins,
                           range=([limits[0], limits[2]], [limits[1], limits[3]]))
    axs[0, 1].imshow(np.log(hm + 1e-1), origin='lower',
                      extent=[limits[1], limits[3], limits[0], limits[2]])

    # Synthetic trajectories
    for traj in synthetic_data[:n_synth]:
        axs[1, 0].plot(traj[:, 1], traj[:, 0], **line_params)

    # Synthetic heatmap
    synth_flat = synthetic_data[:n_synth].reshape(-1, 2)
    hm, _ = np.histogramdd(synth_flat, bins=bins,
                           range=([limits[0], limits[2]], [limits[1], limits[3]]))
    axs[1, 1].imshow(np.log(hm + 1e-1), origin='lower',
                      extent=[limits[1], limits[3], limits[0], limits[2]])

    for ax in axs.flatten():
        ax.set_aspect('equal')
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

    ds_nice = nice_names.get(dataset_str, dataset_str)
    axs[0, 0].set_title(f'Original samples ({n_real:,})')
    axs[1, 0].set_title(f'Synthetic samples ({n_synth:,})')
    axs[0, 1].set_title('Original log-density')
    axs[1, 1].set_title('Synthetic log-density')

    out = f'figures/{dataset_str}_{model}_global_comparison.{save_format}'
    plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f'Saved {out}')


def main():
    parser = argparse.ArgumentParser(description='Plot real vs synthetic global comparison')
    parser.add_argument('datasets', nargs='+', help='Dataset names (e.g. brest-100)')
    parser.add_argument('--model', default='tddpm', help='Model name (default: tddpm)')
    parser.add_argument('--n-samples', type=int, default=10000, help='Number of samples to plot (default: 10000)')
    parser.add_argument('--format', default='png', choices=['png', 'pdf'], help='Output format')
    args = parser.parse_args()

    for dataset_str in args.datasets:
        plot_comparison(dataset_str, args.model, args.n_samples, args.format)


if __name__ == '__main__':
    main()
