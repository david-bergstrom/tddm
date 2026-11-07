from preproc import read_porto_dfs, get_limit, find_cuts, to_numpy_with_time_elapsed, interpolate_subtraj, find_partial_region, read_cabspotting_dfs, read_geolife_dfs, read_brest_dfs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
import os
from tddpm.common.data import SlidingDataset
from random import randint
import requests
import json
from functools import partial
import argparse


os.makedirs('data/preprocessed', exist_ok=True)
os.makedirs('figures', exist_ok=True)


def match_trajectory(df_in, debug=False):
    """
    Assumes there is a Valhalla server running on port 8002
    """
    meili_coordinates = df_in.rename(columns={'latitude': 'lat', 'longitude': 'lon'}) # .drop(columns=['date_time'])
    meili_coordinates = meili_coordinates.to_json(orient='records')

    # Providing needed data for the body of Meili's request
    meili_head = '{"shape":'
    # Those are parameters that you can change according to the Meili's documentation
    meili_tail = ""","search_radius": 150, "shape_match":"map_snap", "costing":"auto", "format":"osrm"}"""

    # Combining all the string into a single request
    meili_request_body = meili_head + meili_coordinates + meili_tail

    # print(meili_request_body)

    url = "http://localhost:8002/trace_route"

    # Providing headers to the request
    headers = {'Content-type': 'application/json'}

    # We need to send our JSON as a string
    data = str(meili_request_body)

    # Sending a request
    r = requests.post(url, data=data, headers=headers)

    if r.status_code == 200:
        # Parsing the JSON response
        response = json.loads(r.text)
        
        res = []
        for match in response['tracepoints']:
            if match is not None and 'location' in match:
                res.append(match['location'] + [match['distance']])
            else:
                res.append([np.nan, np.nan])

        df_matched = pd.DataFrame(res, columns=['lon', 'lat', 'distance'])
        # df_matched['valid'] = df_matched.isna().any(axis=1)
        return df_matched
    else:
        return None


def unpack_trajectories(all_concatenated, idx):
    trajectories = []
    for start, stop in zip(idx, idx[1:]):
        trajectories.append(all_concatenated[start:stop])
    return trajectories


def meters_from_point(trajectory, origin):
    lon = np.radians(trajectory[:, 1])
    lat = np.radians(trajectory[:, 0])

    origin = np.radians(origin)
    origin_lat, origin_lon = origin

    dlon = lon - origin_lon
    dlat = lat - origin_lat
    
    # a = np.sin(dlat/2.0)**2 + np.cos(lat) * np.cos(lat + dlat) * np.sin(dlon/2.0)**2
    # return 6367 * 2 * np.arcsin(np.sqrt(a))

    x = np.sin(dlat/2.0)**2 + np.cos(lat) * np.cos(lat + dlat) * np.sin(0/2.0)**2
    y = np.sin(0/2.0)**2 + np.cos(lat) * np.cos(lat + 0) * np.sin(dlon/2.0)**2

    x = 6367 * 2 * np.arcsin(np.sqrt(x))
    y = 6367 * 2 * np.arcsin(np.sqrt(y))

    return 1000 * np.stack([x, y], axis=-1)


def process_df(df, target_dt, use_map_match=True):
    subtraj_np = to_numpy_with_time_elapsed(df)
    subtraj_np = interpolate_subtraj(subtraj_np, target_dt)

    subtraj_df = pd.DataFrame(subtraj_np, columns=['lat', 'lon'])

    if use_map_match:
        matched = match_trajectory(subtraj_df)
        if matched is not None:
            # Convert to float32 immediately to reduce memory
            matched = matched.astype(np.float32)
        return matched
    else:
        # Return raw coordinates without map matching, with dummy distance column of zeros
        # Convert to float32 immediately to reduce memory
        subtraj_df['distance'] = np.float32(0.0)
        return subtraj_df.astype(np.float32)


def plot(data, data_desc):
    plt.figure(figsize=(10,10))
    for traj in data[:1000]:
        plt.plot(traj[:, 1], traj[:, 0], linewidth=.1)
    plt.axis('equal')
    plt.title(f'First 1000 trajectories of {data_desc}')
    filename = f'figures/{data_desc}.png'
    plt.savefig(filename)
    plt.close()
    print(f'Saved figure to {filename}')


def main():
    parser = argparse.ArgumentParser(
                prog='Data Preprocessor',
                description='Preprocesses trajectory datasets (Geolife, Porto, Cabspotting, Brest) by map matching (not for Brest) and filtering')
    parser.add_argument('dataset', choices=[
        'geolife-25', 'geolife-100', 'porto-25', 'porto-100', 'cabspotting-25', 'cabspotting-100',
        # Corner variants for 4-corner generalization experiment
        'geolife-25-bl', 'geolife-25-br', 'geolife-25-tl', 'geolife-25-tr',
        'porto-25-bl', 'porto-25-br', 'porto-25-tl', 'porto-25-tr',
        'cabspotting-25-bl', 'cabspotting-25-br', 'cabspotting-25-tl', 'cabspotting-25-tr',
        # Maritime AIS
        'brest-100',
        'brest-25', 'brest-25-bl', 'brest-25-br', 'brest-25-tl', 'brest-25-tr',
    ])
    parser.add_argument('--read-map-matched', default=False, action='store_true')
    parser.add_argument('--source-dataset', default=None, type=str,
                        help='Extract a geographic subset from another dataset\'s map-matched data. '
                             'E.g. --source-dataset porto-100 when preprocessing porto-25-bl. '
                             'Loads the source .npz and re-filters to the target bounds.')
    parser.add_argument('--no-map-match', default=False, action='store_true', help='Skip map matching and use raw interpolated GPS coordinates')
    args = parser.parse_args()

    dataset_str = args.dataset
    seq_len = 128

    # Maritime datasets have no road network — always skip map matching
    if 'brest' in dataset_str:
        args.no_map_match = True

    # unconditional

    limits = get_limit(dataset_str)
    target_dt = 1
    if 'brest' in dataset_str:
        # Maritime: 1-50 knots = 1.852-92.6 km/h, 30-minute gap threshold
        max_dt = 1800
        valid_velocity = (1.852, 92.6)
    else:
        max_dt = 70
        valid_velocity = (10, 120)
    windowing = 'sliding'
    progressbars = True
    save_n = 10000

    region_size = 0.05, 0.05
    fraction_inside = 0.80

    n_processes = 20
    chunksize = 100

    # paths
    suffix = '-no-map-match' if args.no_map_match else ''
    save_processed = f'data/preprocessed/{dataset_str}-filtered-map-matched{suffix}.npz'
    windowed_subset_save_path = f'data/preprocessed/'

    if progressbars:
        pb = tqdm
    else:
        pb = lambda iter: iter

    if args.source_dataset is not None:
        # Load map-matched data from a different (larger) dataset and re-filter geographically.
        # This avoids re-running map matching for subsets like corners of a city.
        source_path = f'data/preprocessed/{args.source_dataset}-filtered-map-matched{suffix}.npz'
        npz = np.load(source_path)
        print(f"Loaded source data from {source_path}")

        src_concatenated = npz['all_concatenated']  # (N, 3) with [lat, lon, distance]
        src_idx = npz['idx']

        min_lat, max_lat, min_lon, max_lon = limits

        # Re-filter each trajectory to the target geographic bounds,
        # cutting at boundary crossings (same logic as find_cuts but without velocity/dt checks,
        # since those were already applied when creating the source data).
        min_seq_len = 10
        all_subtrajs = []
        for start, stop in zip(src_idx[:-1], src_idx[1:]):
            traj = src_concatenated[start:stop]
            # Find contiguous runs of points inside the target bounds
            inside = ((min_lat < traj[:, 0]) & (traj[:, 0] < max_lat) &
                      (min_lon < traj[:, 1]) & (traj[:, 1] < max_lon))
            # Split at boundary crossings
            last_cut = 0
            for j in range(1, len(inside)):
                if not inside[j]:
                    if j - last_cut >= min_seq_len and inside[last_cut]:
                        all_subtrajs.append(traj[last_cut:j])
                    last_cut = j + 1
            # Handle final segment
            if len(inside) - last_cut >= min_seq_len and inside[last_cut]:
                all_subtrajs.append(traj[last_cut:])

        print(f"Extracted {len(all_subtrajs)} sub-trajectories within {dataset_str} bounds from {args.source_dataset}")

        all_concatenated = np.concatenate(all_subtrajs, axis=0)
        idx = np.array([0] + [len(s) for s in all_subtrajs], dtype=int).cumsum()

        # Save the filtered data so it can be reloaded with --read-map-matched later
        np.savez_compressed(save_processed, all_concatenated=all_concatenated, idx=idx)
        print(f"Saved filtered data to {save_processed}")

    elif args.read_map_matched:
        npz = np.load(save_processed)
        print(f"Read data from {save_processed}")

        all_concatenated = npz['all_concatenated']
        idx = npz['idx']
    else:
        if 'porto' in dataset_str:
            dfs = read_porto_dfs()
        elif 'cabspotting' in dataset_str:
            dfs = read_cabspotting_dfs()
        elif 'geolife' in dataset_str:
            dfs = read_geolife_dfs()
        elif 'brest' in dataset_str:
            dfs = read_brest_dfs()

        print(f"Extracted {len(dfs)} trajectories from {dataset_str}")

        plt.figure(figsize=(10,10))
        for df in dfs:
            # matched_df = matched_df[matched_df['valid']]
            plt.plot(df['longitude'], df['latitude'], linewidth=.1)
        plt.xlim(*limits[2:])
        plt.ylim(*limits[:2])
        plt.savefig(f'figures/{dataset_str}-raw-trajectories.png')
        print(f"Saved figures of raw data to: figures/{dataset_str}-raw-trajectories.png")

        process_df_with_target_dt = partial(process_df, target_dt=target_dt, use_map_match=not args.no_map_match)

        with Pool(n_processes) as pool:
            all_subtrajs_np = []  # Store numpy arrays directly

            for df in tqdm(dfs):
                valid_subtrajs = find_cuts(df, limits, max_dt, valid_velocity, 10)

                # Process each subtrajectory and convert to numpy immediately
                for matched_df in pool.imap(process_df_with_target_dt, valid_subtrajs):
                    if matched_df is not None:
                        # Split on NaN boundaries and convert to numpy immediately
                        for _, df_i in matched_df.dropna().groupby(matched_df['lat'].isna().cumsum()):
                            # Already float32 from process_df, extract numpy array
                            subtraj_np = df_i[['lat', 'lon', 'distance']].to_numpy()
                            all_subtrajs_np.append(subtraj_np)

        # Plot trajectories (sample for memory efficiency)
        plt.figure(figsize=(10,10))
        plot_sample = min(len(all_subtrajs_np), 10000)
        for i in range(0, plot_sample, max(1, plot_sample // 1000)):
            traj = all_subtrajs_np[i]
            plt.plot(traj[:, 1], traj[:, 0], linewidth=.1)
        plt.axis('equal')
        plt.savefig(f'figures/{dataset_str}-matched-trajectories.png')

        all_concatenated = np.concatenate(all_subtrajs_np, axis=0)
        idx = np.array([0] + [len(subtraj) for subtraj in all_subtrajs_np], dtype=int).cumsum()

        res = 512

        plt.figure(figsize=(10, 10))
        H, xedges, yedges = np.histogram2d(all_concatenated[:, 0], all_concatenated[:, 1], bins=(res, res), range=np.array(limits).reshape(2, 2), density=False)
        plt.imshow(np.log(H), extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]], origin='lower', interpolation='nearest')
        plt.axis('equal')
        plt.colorbar()
        plt.savefig(f'figures/{dataset_str}-matched-trajectories-histogram.png')
        plt.close()

        len(all_subtrajs_np)

        plt.figure(figsize=(10, 10))
        plt.hist(all_concatenated[:, 2], bins=100, density=True, cumulative=False)
        plt.title("Distance between original and map matched trajectories")
        plt.savefig(f'figures/{dataset_str}-original-matched-distance-histogram.png')
        plt.close()

        # Validation removed to save memory - unpacking/repacking creates full copy of data
        np.savez_compressed(save_processed, all_concatenated=all_concatenated, idx=idx)

    transformed = meters_from_point(all_concatenated, [limits[0], limits[2]])
    print("Transformed trajectories to meters")

    trajectories = unpack_trajectories(transformed, idx)
    print(f"Unpacked {len(trajectories)} trajectories")

    if args.no_map_match:
        # Skip noise addition when not using map matching
        trajectories_with_noise = trajectories
        print("Skipping noise addition (no map matching used)")
    else:
        # Add noise based on map matching distance distribution
        ps, bins = np.histogram(all_concatenated[:, 2], bins=1000)
        ps = ps / ps.sum()

        gen = np.random.default_rng(seed=1234)
        noise_idx = gen.choice(np.arange(len(bins)-1), size=(len(trajectories)), p=ps)
        noise_amp = gen.uniform(bins[noise_idx], bins[noise_idx+1])
        direction = gen.uniform(0, 2*np.pi, size=(len(trajectories)))

        noise_x = np.cos(direction)*noise_amp
        noise_y = np.sin(direction)*noise_amp
        noise = np.stack([noise_x, noise_y], axis=-1)

        trajectories_with_noise = []

        for traj, traj_noise in zip(trajectories, noise):
            trajectories_with_noise.append(traj + traj_noise.reshape(1, 2))
        print("Noisy trajectories calculated")
        del trajectories

    print("Plotting all trajectories with noise added...")
    plt.figure(figsize=(10,10))
    for traj in trajectories_with_noise[:1000]:
        plt.plot(traj[:, 1], traj[:, 0], linewidth=.1)
    plt.axis('equal')
    plt.savefig(f'figures/{dataset_str}-trajectories-with-noise.png')
    plt.close()

    sliding_subtrajs = SlidingDataset(trajectories_with_noise, seq_len)

    idx = np.arange(len(sliding_subtrajs))

    np.random.seed(12345)
    np.random.shuffle(idx)

    save_n_training = [10000, 100000]
    save_n_embedder = 100000

    for save_n in save_n_training:
        actual_n = min(save_n, len(sliding_subtrajs))
        if actual_n < save_n:
            print(f"WARNING: Only {actual_n} windows available, requested {save_n}")
        train = np.zeros((actual_n, seq_len, 2))

        for i, j in enumerate(idx[:actual_n]):
            train[i, :, :] = sliding_subtrajs[j][:, :2]

        data_desc = f'{dataset_str}{suffix}-{save_n}'
        filename = f'{data_desc}.npy'
        save_path = os.path.join(windowed_subset_save_path, filename)
        np.save(save_path, train)
        print(f"Saved {save_n} windows to {save_path}")

        plot(train, data_desc)

    start_idx = save_n_training[-1]
    actual_n_embedder = min(save_n_embedder, max(0, len(sliding_subtrajs) - start_idx))
    if actual_n_embedder < save_n_embedder:
        print(f"WARNING: Only {actual_n_embedder} windows available for embedder (after skipping {start_idx}), requested {save_n_embedder}")
    embedder_train = np.zeros((actual_n_embedder, seq_len, 2))

    for i, j in enumerate(idx[start_idx:start_idx+actual_n_embedder]):
        embedder_train[i, :, :] = sliding_subtrajs[j][:, :2]

    data_desc = f'{dataset_str}{suffix}-embedder-train-{save_n_embedder}'
    filename = f'{data_desc}.npy'
    save_path = os.path.join(windowed_subset_save_path, filename)
    np.save(save_path, embedder_train)
    print(f"Saved {save_n_embedder} windows to {save_path}")
    plot(embedder_train, data_desc)

    print("Testing conditional data extraction!")

    min_x, min_y = train.min(axis=(0, 1))
    max_x, max_y = train.max(axis=(0, 1))
    limits_m = [min_y, max_y, min_x, max_x]
    region_size_debug = 5000, 5000

    fraction_inside_debug = 1

    for i in range(10):
        test_traj_idx = randint(0, len(train)-1)
        success, region, traj, heatmap = find_partial_region(train[test_traj_idx], 
            train,
            size=region_size_debug,
            bounds=limits_m,
            fraction_inside=fraction_inside_debug
            )

        if success:
            plt.figure()
            plt.imshow(heatmap, origin='lower', extent=[0, region_size_debug[0], 0, region_size_debug[1]])
            scaled_traj = np.array(region_size_debug) * traj
            plt.plot(scaled_traj[:, 1], scaled_traj[:, 0], '.', color='red')
            plt.savefig(f'figures/{dataset_str}-conditional-data-extraction-test-{i}.png')
            plt.close()
        else:
            print("Found no region!")


if __name__ == '__main__':
    main()