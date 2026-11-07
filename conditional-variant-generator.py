from preproc import get_limit, find_partial_region, InterpolatingHeatmapper
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import os
import argparse


# Module-level globals shared with Pool workers via fork (copy-on-write).
# Set in main() before Pool creation — never pickled.
_worker_train = None
_worker_heatmapper = None
_worker_params = None


def _process_chunk(chunk):
    """Process a chunk of trajectories using the fork-inherited shared state."""
    p = _worker_params
    regions = []
    heatmaps = []
    trajectories = []

    for traj in chunk:
        for j in range(p['n_regions_per_window']):
            found, region, traj_scaled, heatmap = find_partial_region(
                traj, dataset=_worker_train, size=p['region_size'],
                bounds=p['limits'], fraction_inside=p['fraction_inside'],
                heatmapper=_worker_heatmapper, random_tries=p['random_tries'],
                max_rotation=p['max_rotation'], res=p['res'])
            if found:
                regions.append(region)
                heatmaps.append(heatmap)
                trajectories.append(traj_scaled)
            else:
                break

    return len(chunk), regions, heatmaps, trajectories


def main():
    global _worker_train, _worker_heatmapper, _worker_params

    parser = argparse.ArgumentParser(description='Load dataset')
    parser.add_argument('dataset_str')
    parser.add_argument('region_side', type=int)
    parser.add_argument('--fraction_inside', type=float, default=1, required=False)
    parser.add_argument('--max_rotation', type=int, default=360)
    parser.add_argument('--res', type=int, default=64, help='Heatmap resolution (default: 64)')
    parser.add_argument('--regions-per-window', type=int, default=10, help='Number of training examples generated from each trajectory (default: 10).')
    parser.add_argument('--n-processes', type=int, default=20, help='Number of parallel processes (default: 20)')
    args = parser.parse_args()

    dataset_str = args.dataset_str
    seq_len = 128
    limits = get_limit(dataset_str)
    region_size = args.region_side, args.region_side
    fraction_inside = args.fraction_inside
    max_rotation = args.max_rotation
    res = args.res

    windowed_subset_save_path = f'data/preprocessed/'
    save_n = 100000  # number of random windows to use
    save_path = os.path.join(windowed_subset_save_path, f'{dataset_str}-{save_n}.npy')

    train = np.load(save_path)
    train = train.astype(np.float32)

    min_x, min_y = train.min(axis=(0, 1))
    max_x, max_y = train.max(axis=(0, 1))

    limits_m = [min_y, max_y, min_x, max_x]
    print(limits, limits_m)

    n_regions_per_window = args.regions_per_window
    random_tries = 10000

    pb = tqdm(total=n_regions_per_window*len(train))

    # Set globals before Pool creation so forked workers inherit them via COW
    _worker_train = train
    _worker_heatmapper = InterpolatingHeatmapper(train, meters_per_cell=1, sigma=0, device='cpu')

    # Prevent OMP thread pool deadlock when forking with torch tensors
    import torch
    torch.set_num_threads(1)
    _worker_params = dict(
        n_regions_per_window=n_regions_per_window,
        region_size=region_size,
        limits=limits_m,
        fraction_inside=fraction_inside,
        random_tries=random_tries,
        max_rotation=max_rotation,
        res=res,
    )

    chunk_size = 16
    chunks = [train[i:i + chunk_size] for i in range(0, len(train), chunk_size)]
    processed = 0
    found = 0

    regions = []
    heatmaps = []
    trajectories = []

    n_processes = args.n_processes
    with Pool(processes=n_processes) as pool:
        for n_processed, regions_chunk, heatmaps_chunk, trajectories_chunk in pool.imap_unordered(_process_chunk, chunks):
            regions.extend(regions_chunk)
            heatmaps.extend(heatmaps_chunk)
            trajectories.extend(trajectories_chunk)
            processed += n_processed*n_regions_per_window
            found += len(regions_chunk)
            pb.update(n_processed*n_regions_per_window)
            pb.set_description_str(f'Processed {processed} trajectories, found {found} regions')

    if res != 64:
        filename = f'data/preprocessed/{dataset_str}-{save_n}-conditional-{seq_len}-{region_size[0]}-{res}res.npz'
    else:
        filename = f'data/preprocessed/{dataset_str}-{save_n}-conditional-{seq_len}-{region_size[0]}.npz'
    np.savez_compressed(filename, regions=regions, heatmaps=heatmaps, trajectories=trajectories,
                        heatmapper_type=np.array('interpolating'),
                        heatmapper_meters_per_cell=np.array(_worker_heatmapper.meters_per_cell),
                        heatmapper_sigma=np.array(_worker_heatmapper.sigma))


if __name__ == '__main__':
    main()
