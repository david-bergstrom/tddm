import argparse
import pickle
import numpy as np

from synthetic_loader import find_synthetic_datasets, load_synthetic
from data import load_dataset_from_str, minmaxscale
from config import datasets, models


def bayesian_smoothed_histogram(speeds, bins, speed_range):
    """
    Create a Bayesian-smoothed histogram for speed distribution.

    Args:
        speeds: 1D array of speeds
        bins: Number of bins, or an array of bin edges
        speed_range: Tuple (min_speed, max_speed) for histogram range, or None when bins are edges

    Returns:
        Normalized probability distribution with Bayesian smoothing
    """
    H, edges = np.histogram(speeds, bins=bins, range=speed_range)
    # Bayesian smoothing: (count + 1) / (total_bins + total_count)
    H = (H + 1) / (len(H) + H.sum())
    return H, edges


def KL(p, q):
    """Compute KL divergence KL(p||q)."""
    return (p * np.log(p / q)).sum()


def _filter_out_quarter(dataset, lat_pred, lon_pred):
    overlaps = np.logical_and(lat_pred(dataset[:, :, 0]), lon_pred(dataset[:, :, 1])).any(axis=1)
    return dataset[~overlaps]


# Axis 0 is latitude and axis 1 longitude, as in run_evaluation.filter_out_training_region and preproc.LIMITS
_QUARTER_PREDICATES = {
    'lower-left-quarter':  (lambda lat: lat < 0.5, lambda lon: lon < 0.5),
    'lower-right-quarter': (lambda lat: lat < 0.5, lambda lon: lon >= 0.5),
    'upper-left-quarter':  (lambda lat: lat >= 0.5, lambda lon: lon < 0.5),
    'upper-right-quarter': (lambda lat: lat >= 0.5, lambda lon: lon >= 0.5),
}


def filter_data(data, filter_out):
    """Apply data filtering based on filter_out option."""
    if filter_out == 'none':
        return data
    if filter_out in _QUARTER_PREDICATES:
        lat_pred, lon_pred = _QUARTER_PREDICATES[filter_out]
        return _filter_out_quarter(data, lat_pred, lon_pred)
    raise ValueError(f"Unknown value for filter_out: {filter_out}")


def compute_speeds(trajectories, extent_meters_yx):
    """
    Compute speeds from trajectories using Euclidean distance.

    Args:
        trajectories: Array of shape (N, seq_len, 2), in city-normalised [0, 1] coordinates
        extent_meters_yx: Extent of the city in metres per axis, used to convert the steps to metres

    Returns:
        1D array of all speeds in metres per step (m/s at 1 Hz), flattened from all trajectories
    """
    # Compute differences between consecutive points, in metres
    diffs = np.diff(trajectories, axis=1) * np.asarray(extent_meters_yx)  # Shape: (N, seq_len-1, 2)

    # Compute Euclidean distance (speed proxy)
    speeds = np.linalg.norm(diffs, axis=-1)  # Shape: (N, seq_len-1)

    # Flatten to 1D array
    return speeds.flatten()


def compute_speed_kl_divergence(real_trajectories, synthetic_trajectories, extent_meters_yx, num_bins=100):
    """
    Compute KL divergence between speed distributions of real and synthetic trajectories.

    Args:
        real_trajectories: Array of shape (N, seq_len, 2), city-normalised
        synthetic_trajectories: Array of shape (N, seq_len, 2), in the same city-normalised frame
        extent_meters_yx: Extent of the city in metres per axis
        num_bins: Number of bins between 0 and the fastest real step

    Returns:
        Tuple of (kl_synthetic_real, kl_real_synthetic, kl_symmetric, js_divergence)
    """
    # Compute speeds in metres per step
    real_speeds = compute_speeds(real_trajectories, extent_meters_yx)
    synthetic_speeds = compute_speeds(synthetic_trajectories, extent_meters_yx)

    # Bins are anchored to the real data only, so every model is scored on the same bins:
    # num_bins equal bins on [0, fastest real step], plus one overflow bin for faster synthetic steps.
    # np.histogram bins are half-open, so the last edge is nudged up to keep the fastest real step in range.
    upper = np.nextafter(real_speeds.max(), np.inf)
    edges = np.append(np.linspace(0, upper, num_bins + 1), np.inf)

    # Create smoothed histograms
    real_hist, _ = bayesian_smoothed_histogram(real_speeds, edges, None)
    synthetic_hist, _ = bayesian_smoothed_histogram(synthetic_speeds, edges, None)

    # Compute KL divergences
    kl_synthetic_real = KL(synthetic_hist, real_hist)
    kl_real_synthetic = KL(real_hist, synthetic_hist)
    kl_symmetric = 0.5 * (kl_synthetic_real + kl_real_synthetic)

    # Compute JS divergence
    avg_hist = 0.5 * (real_hist + synthetic_hist)
    js_divergence = 0.5 * (KL(real_hist, avg_hist) + KL(synthetic_hist, avg_hist))

    return kl_synthetic_real, kl_real_synthetic, kl_symmetric, js_divergence


def main():
    parser = argparse.ArgumentParser(description='Calculate KL divergence between speed distributions')
    parser.add_argument('--num-bins', type=int, default=100, help='Number of bins for histogram (default: 100)')
    # Synthetic data used as loaded, speeds in metres, bins anchored to the real data
    parser.add_argument('--output', type=str, default='speed_kl_measures_v2.pickle',
                        help='Output pickle file (default: speed_kl_measures_v2.pickle)')
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

    # Cache for loaded real datasets
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

            source_dataset_in_model_name = None
            prefix = 'tddpm-generalize-from-'
            if model.startswith(prefix):
                source_dataset_in_model_name = model[len(prefix):]
                if source_dataset_in_model_name == dataset_str:
                    # This is a diagonal entry, use the standard TDDPM model for data loading
                    model_for_loading = 'tddpm'

            # Store results under the original key
            key = (dataset_str, model, filter_out)

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

            # Load and cache real dataset (with filter applied)
            cache_key = (dataset_str, filter_out)
            if cache_key not in real_data_cache:
                real_data = load_dataset_from_str(dataset_str=dataset_str, prefix='data/')
                real_data, d_min, d_max = minmaxscale(real_data, return_bounds=True)
                extent_meters_yx = tuple(np.squeeze(d_max - d_min).tolist())
                real_data = filter_data(real_data, filter_out)
                real_data_cache[cache_key] = (real_data, extent_meters_yx)
            real_data, extent_meters_yx = real_data_cache[cache_key]

            # Compute KL divergence
            try:
                kl_sr, kl_rs, kl_sym, js = compute_speed_kl_divergence(
                    real_data, synthetic_data, extent_meters_yx, num_bins=args.num_bins
                )
                results[key] = (kl_sr, kl_rs, kl_sym, js)
                print(f"  ✓ Symmetric KL: {kl_sym:.4f}, JS: {js:.4f}")
            except Exception as e:
                print(f"  Error computing KL divergence: {e}")
                continue

    # Save results to pickle
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n✓ Results saved to {args.output}")
    print(f"✓ Computed KL divergence for {len(results)} dataset-model-filter combinations")

    # Print summary statistics
    if results:
        all_kl_sym = [v[2] for v in results.values()]
        print(f"\nSummary statistics:")
        print(f"  Mean symmetric KL: {np.mean(all_kl_sym):.4f}")
        print(f"  Std symmetric KL:  {np.std(all_kl_sym):.4f}")
        print(f"  Min symmetric KL:  {np.min(all_kl_sym):.4f}")
        print(f"  Max symmetric KL:  {np.max(all_kl_sym):.4f}")


if __name__ == '__main__':
    main()
