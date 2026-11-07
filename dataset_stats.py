#!/usr/bin/env python3
import argparse
import os
import sys
from glob import glob

import numpy as np


REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
PREPROCESSED_DIR = os.path.join(REPO_ROOT, 'data', 'preprocessed')


def compute_stats(arr: np.ndarray) -> dict:
    # arr shape: (num_trajectories, seq_len, 2), coordinates in meters
    num_traj = arr.shape[0]
    seq_len = arr.shape[1]

    # Bounding box across all points (meters)
    mins = arr.min(axis=(0, 1))  # (2,)
    maxs = arr.max(axis=(0, 1))  # (2,)
    width_m = float(maxs[1] - mins[1])
    height_m = float(maxs[0] - mins[0])
    width_km = width_m / 1000.0
    height_km = height_m / 1000.0
    area_km2 = (width_km * height_km)

    # Total length per trajectory (sum of step norms)
    diffs = np.diff(arr, axis=1)  # (N, seq_len-1, 2)
    step_lengths = np.linalg.norm(diffs, axis=-1)  # (N, seq_len-1)
    total_lengths = step_lengths.sum(axis=1)  # (N,)

    return {
        'num_traj': int(num_traj),
        'seq_len': int(seq_len),
        'width_km': width_km,
        'height_km': height_km,
        'area_km2': area_km2,
        'mean_total_len_m': float(total_lengths.mean()),
        'median_total_len_m': float(np.median(total_lengths)),
        'mean_step_len_m': float(step_lengths.mean()),
    }


def find_default_files() -> list:
    # Expect files like: porto-100-100000.npy, geolife-100-100000.npy, cabspotting-100-100000.npy
    patterns = [
        os.path.join(PREPROCESSED_DIR, 'porto-100-100000.npy'),
        os.path.join(PREPROCESSED_DIR, 'geolife-100-100000.npy'),
        os.path.join(PREPROCESSED_DIR, 'cabspotting-100-100000.npy'),
    ]
    files = [p for p in patterns if os.path.isfile(p)]
    # Fallback: pick any *-100000.npy three datasets if named slightly differently
    if len(files) == 0:
        files = sorted(glob(os.path.join(PREPROCESSED_DIR, '*-100000.npy')))
    return files


def main():
    parser = argparse.ArgumentParser(description='Print stats for preprocessed trajectory datasets (.npy)')
    parser.add_argument('files', nargs='*', help='Paths to .npy datasets (default: detect common 100k files in data/preprocessed)')
    args = parser.parse_args()

    files = args.files if args.files else find_default_files()
    if not files:
        print(f'No datasets found. Looked in {PREPROCESSED_DIR}', file=sys.stderr)
        sys.exit(1)

    for path in files:
        try:
            arr = np.load(path)
        except Exception as e:
            print(f'Failed to load {path}: {e}', file=sys.stderr)
            continue

        stats = compute_stats(arr.astype(np.float32))
        name = os.path.basename(path)
        print(f'== {name} ==')
        print(f'- trajectories: {stats["num_traj"]}')
        print(f'- seq_len: {stats["seq_len"]}')
        print(f'- bbox width x height (km): {stats["width_km"]:.2f} x {stats["height_km"]:.2f}')
        print(f'- area (km^2): {stats["area_km2"]:.2f}')
        print(f'- mean total length (m): {stats["mean_total_len_m"]:.1f} (median: {stats["median_total_len_m"]:.1f})')
        print(f'- mean step length (m): {stats["mean_step_len_m"]:.2f}')
        print()


if __name__ == '__main__':
    main() 