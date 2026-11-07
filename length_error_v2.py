"""
Length error: Jensen-Shannon divergence between per-trajectory travel distances.

The length of a trajectory is its total travel distance in metres, the sum of the Euclidean distances
between consecutive points. Every trajectory contributes one value, the values of the real and the
synthetic set are histogrammed on shared bins, and the two histograms are compared with the JSD. This is
the Length error of LDPTrace (Du et al. 2023, calculate_length_error: 20 buckets over the real range),
which DiffTraj (Zhu et al. 2023, appendix) takes over as a trajectory-level metric of "the distribution of
travel distances". Here the bins are
num_bins equal bins on [0, longest real trajectory] plus one overflow bin, so synthetic trajectories that
are too long are penalised instead of dropped, and every model is scored on the same bins. The real
reference is the same seeded 3,000-window subset that run_evaluation.py uses for Density, Start/End and
Pattern.
"""
import argparse
import pickle
import numpy as np

from synthetic_loader import find_synthetic_datasets, load_synthetic
from run_evaluation import load_train_data, get_random_subset, filter_data, jsd
from config import datasets, models


def trajectory_lengths(trajectories, extent_meters_yx):
    """
    Compute the total travel distance of every trajectory.

    Args:
        trajectories: Array of shape (N, seq_len, 2), in city-normalised [0, 1] (y, x) coordinates
        extent_meters_yx: Extent of the city in metres per axis, in the same (y, x) order as the coordinates

    Returns:
        1D array of N lengths in metres
    """
    # float64, since the samples are float32 and the lengths are sums of 127 steps
    steps = np.diff(trajectories.astype(np.float64), axis=1) * np.asarray(extent_meters_yx)  # (N, seq_len-1, 2)
    return np.linalg.norm(steps, axis=-1).sum(axis=1)


def length_histograms(real_lengths, synthetic_lengths, num_bins=20):
    """
    Histogram the real and synthetic trajectory lengths on shared bins.

    Args:
        real_lengths: 1D array of real trajectory lengths
        synthetic_lengths: 1D array of synthetic trajectory lengths
        num_bins: Number of bins between 0 and the longest real trajectory

    Returns:
        Tuple of (real fractions, synthetic fractions, bin edges), with num_bins + 1 bins; the last bin is the
        overflow bin, from the longest real trajectory to infinity
    """
    # Bins are anchored to the real data only: num_bins equal bins on [0, longest real trajectory], plus one
    # overflow bin for longer synthetic trajectories. np.histogram bins are half-open, except that the very
    # last one (here the overflow bin) is closed, so the last edge is nudged up to keep the longest real
    # trajectory in the last regular bin.
    upper = np.nextafter(real_lengths.max(), np.inf)
    edges = np.append(np.linspace(0, upper, num_bins + 1), np.inf)

    P, _ = np.histogram(real_lengths, bins=edges)
    G, _ = np.histogram(synthetic_lengths, bins=edges)
    P = P / P.sum()
    G = G / G.sum()
    assert P[-1] == 0, "A real trajectory landed in the overflow bin"
    return P, G, edges


def compute_length_error(real_trajectories, synthetic_trajectories, extent_meters_yx, num_bins=20):
    """
    Compute the JSD between the trajectory length distributions of real and synthetic trajectories.

    Args:
        real_trajectories: Array of shape (N, seq_len, 2), city-normalised
        synthetic_trajectories: Array of shape (M, seq_len, 2), in the same city-normalised frame
        extent_meters_yx: Extent of the city in metres per axis
        num_bins: Number of bins between 0 and the longest real trajectory

    Returns:
        Tuple of (JSD in bits, fraction of the synthetic trajectories longer than every real trajectory)
    """
    if len(synthetic_trajectories) == 0:
        raise ValueError("No synthetic trajectories left after filtering")

    real_lengths = trajectory_lengths(real_trajectories, extent_meters_yx)
    synthetic_lengths = trajectory_lengths(synthetic_trajectories, extent_meters_yx)

    # No smoothing, run_evaluation.jsd takes care of the empty bins
    P, G, _ = length_histograms(real_lengths, synthetic_lengths, num_bins)

    overflow = G[-1]  # read before jsd, which modifies its arguments in place
    return jsd(P, G), overflow


def main():
    parser = argparse.ArgumentParser(description='Calculate the JSD between trajectory length distributions')
    parser.add_argument('--num-bins', type=int, default=20, help='Number of bins for histogram (default: 20)')
    parser.add_argument('--output', type=str, default='length_measures_v2.pickle',
                        help='Output pickle file (default: length_measures_v2.pickle)')
    args = parser.parse_args()

    # Get all available synthetic datasets
    all_datasets = find_synthetic_datasets()

    # Load existing results if available
    try:
        results = pickle.load(open(args.output, 'rb'))
        print(f"Loaded existing results from {args.output}")
    except FileNotFoundError:
        results = {}

    # Use seq_len=128 as in run_evaluation.py
    seq_len = 128

    # Cache for the real reference sets
    real_data_cache = {}

    print(f"Datasets: {datasets}")
    print(f"Models: {models}")
    print(f"Using {args.num_bins} bins for histograms\n")

    # Iterate through all configured combinations
    for dataset_str in datasets:
        for model, filter_out in models:
            # Determine the actual model to use for data loading
            # For city-to-city generalization, use baseline tddpm for diagonal entries
            model_for_loading = model

            prefix = 'tddpm-generalize-from-'
            if model.startswith(prefix):
                source_dataset_in_model_name = model[len(prefix):]
                if source_dataset_in_model_name == dataset_str:
                    # This is a diagonal entry, use the standard TDDPM model for data loading
                    model_for_loading = 'tddpm'

            # Store results under the original key, same format as the other measure pickles
            key = (dataset_str, seq_len, model, filter_out)

            # Skip if already computed
            if key in results:
                print(f"Skipping {key} - already computed")
                continue

            print(f"Processing: {key}")

            # Find matching synthetic dataset
            matching_synthetic = list(filter(
                lambda s: s.model_name == model_for_loading and s.dataset == dataset_str and s.seq_len == seq_len,
                all_datasets
            ))

            if len(matching_synthetic) == 0:
                print(f"  No synthetic data for {model_for_loading} on {dataset_str}")
                continue
            elif len(matching_synthetic) > 1:
                print(f"  Warning: Multiple synthetic datasets found for {model_for_loading} on {dataset_str}, using first")

            synth_desc = matching_synthetic[0]

            # Load synthetic dataset
            try:
                synthetic_data = load_synthetic(synth_desc)
            except FileNotFoundError:
                print(f"  Synthetic data file not found for {model_for_loading} on {dataset_str}")
                continue

            # Samples are already in the city-normalised frame of the real data; do not rescale them
            synthetic_data = filter_data(synthetic_data, filter_out)

            # Load and cache the real reference set (filter applied), the same one run_evaluation.py uses
            cache_key = (dataset_str, filter_out)
            if cache_key not in real_data_cache:
                train_data, extent_meters_yx = load_train_data(dataset_str, seq_len, filter_out)
                real_data_cache[cache_key] = (get_random_subset(train_data), extent_meters_yx)
            real_data, extent_meters_yx = real_data_cache[cache_key]

            # Compute length error
            try:
                js, overflow = compute_length_error(real_data, synthetic_data, extent_meters_yx, num_bins=args.num_bins)
                results[key] = float(js)
                print(f"  ✓ Length error (JSD): {js:.4f}, longer than all real: {overflow:.2%}")
            except Exception as e:
                print(f"  Error computing length error: {e}")
                continue

    # Save results to pickle
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n✓ Results saved to {args.output}")
    print(f"✓ Computed length error for {len(results)} dataset-model-filter combinations")

    # Print summary statistics
    if results:
        all_js = list(results.values())
        print(f"\nSummary statistics:")
        print(f"  Mean JSD: {np.mean(all_js):.4f}")
        print(f"  Std JSD:  {np.std(all_js):.4f}")
        print(f"  Min JSD:  {np.min(all_js):.4f}")
        print(f"  Max JSD:  {np.max(all_js):.4f}")


if __name__ == '__main__':
    main()
