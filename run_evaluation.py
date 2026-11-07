import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

from synthetic_loader import find_synthetic_datasets, load_synthetic
from common import SlidingDataset
from data import load_dataset_from_str, minmaxscale
from config import models

def load_train_data(dataset_str, seq_len, filter_out):
    data = load_dataset_from_str(dataset_str=dataset_str, prefix='data/')
    data, d_min, d_max = minmaxscale(data, return_bounds=True)
    extent_meters_yx = tuple(np.squeeze(d_max - d_min).tolist())
    data = filter_data(data, filter_out)
    return SlidingDataset(data, seq_len), extent_meters_yx

def bayesian_smoothed_heatmap(trajectories, res, range):
    """
    Unscaled variant of data_generation.region_occupancy, mainly useful for visualization.
    """
    all_concatenated = np.concatenate(trajectories)
    H, xedges, yedges = np.histogram2d(all_concatenated[:, 0], all_concatenated[:, 1], bins=(res, res), range=range)
    n_elements = res**2
    H = (H + 1) / (n_elements + H.sum())
    return H, xedges, yedges

def KL(p, q):
    return (p * np.log(p / q)).sum()


def get_random_subset(dataset, n=3000, seed=12345, seq_len=128):
    idx = np.arange(len(dataset))

    np.random.seed(seed)
    np.random.shuffle(idx)

    train = np.zeros((n, seq_len, 2))

    for i, j in enumerate(idx[:n]):
        train[i, :, :] = dataset[j]

    return train

def density(dataset, bins=16):
    all_concat = np.concatenate(dataset, axis=0)
    h, _, _ = np.histogram2d(all_concat[:, 0], 
                             all_concat[:, 1], 
                             range=((0, 1), (0, 1)), 
                             bins=bins)
    h = h / h.sum()
    return h

def jsd(P, G):
    M = 0.5 * (P+G)

    P += 1e-16
    G += 1e-16
    M += 1e-16

    jsd = .5 * np.sum(P * np.log2(P/M)) + .5 * np.sum(G * np.log2(G/M))
    return jsd

def density_error(train, synth, bins=16):  
    P = density(train, bins)
    G = density(synth, bins)

    return jsd(P, G)

def trip_error(train, synth):
    P_start = density(train[:, 0, None])
    P_stop = density(train[:, -1, None])

    G_start = density(synth[:, 0, None])
    G_stop = density(synth[:, -1, None])

    return 0.5 * jsd(P_start, G_start) + 0.5 * jsd(P_stop, G_stop)

def od_sparse(dataset, bins_yx):
    by, bx = bins_yx
    starts = dataset[:, 0, :]
    stops = dataset[:, -1, :]
    s_y = np.clip((starts[:, 0] * by).astype(int), 0, by - 1)
    s_x = np.clip((starts[:, 1] * bx).astype(int), 0, bx - 1)
    e_y = np.clip((stops[:, 0] * by).astype(int), 0, by - 1)
    e_x = np.clip((stops[:, 1] * bx).astype(int), 0, bx - 1)
    o = s_y * bx + s_x
    d = e_y * bx + e_x
    keys, counts = np.unique(np.stack([o, d], axis=1), axis=0, return_counts=True)
    total = counts.sum()
    return {tuple(k): c / total for k, c in zip(keys, counts)}

def od_jsd(P, G):
    js = 0.0
    for k in set(P) | set(G):
        p = P.get(k, 0.0)
        g = G.get(k, 0.0)
        m = 0.5 * (p + g)
        if p > 0:
            js += 0.5 * p * np.log2(p / m)
        if g > 0:
            js += 0.5 * g * np.log2(g / m)
    return js

def od_error(train, synth, extent_meters_yx, cell_size_meters=1500):
    by = int(np.ceil(extent_meters_yx[0] / cell_size_meters))
    bx = int(np.ceil(extent_meters_yx[1] / cell_size_meters))
    P = od_sparse(train, (by, bx))
    G = od_sparse(synth, (by, bx))
    return od_jsd(P, G)

def top_n(train, synth, n=10, bins=16):
    P = density(train, bins)
    P = P.reshape(-1)
    G = density(synth, bins)
    G = G.reshape(-1)
    
    top_n_train = np.argpartition(P, -n)[-n:]
    top_n_synth = np.argpartition(G, -n)[-n:]

    top_n_train = set(top_n_train)
    top_n_synth = set(top_n_synth)
    relevant_retrieved_instances = top_n_synth.intersection(top_n_train)
    precision = len(relevant_retrieved_instances) / len(top_n_synth)
    recall = len(relevant_retrieved_instances) / len(top_n_train)
    top_n_score = 2 * (precision * recall) / (precision + recall)

    return top_n_score

def filter_out_training_region(dataset, quarter='lower-left'):
    """Filter out trajectories that pass through the training region (one quadrant).

    After min-max normalization, each quadrant occupies [0,0.5) or [0.5,1] on each axis.
    A trajectory is considered 'in' the training region if *any* of its points fall in
    that quadrant.
    """
    lat = dataset[:, :, 0]
    lon = dataset[:, :, 1]

    if quarter == 'lower-left':
        in_training = np.logical_and(lat < 0.5, lon < 0.5).any(axis=1)
    elif quarter == 'lower-right':
        in_training = np.logical_and(lat < 0.5, lon >= 0.5).any(axis=1)
    elif quarter == 'upper-left':
        in_training = np.logical_and(lat >= 0.5, lon < 0.5).any(axis=1)
    elif quarter == 'upper-right':
        in_training = np.logical_and(lat >= 0.5, lon >= 0.5).any(axis=1)
    else:
        raise ValueError(f"Unknown quarter: {quarter}")

    return dataset[~in_training]


FILTER_QUARTERS = {
    'lower-left-quarter':  'lower-left',
    'lower-right-quarter': 'lower-right',
    'upper-left-quarter':  'upper-left',
    'upper-right-quarter': 'upper-right',
}


def filter_data(data, filter_out):
    if filter_out == 'none':
        return data
    elif filter_out in FILTER_QUARTERS:
        return filter_out_training_region(data, FILTER_QUARTERS[filter_out])
    else:
        raise ValueError(f"Unknown value for filter_out: {filter_out}")

dataset_strs = ['geolife-100', 'porto-100', 'cabspotting-100', 'brest-100']
# These are equivalent to (dataset, seq_len, 'tddpm'), since we train and generalize to the same dataset
ignore_combinations = [('geolife-100', 128, 'tddpm-generalize-from-geolife-100'),
                       ('porto-100', 128, 'tddpm-generalize-from-porto-100'),
                       ('cabspotting-100', 128, 'tddpm-generalize-from-cabspotting-100'),
                       ('brest-100', 128, 'tddpm-generalize-from-brest-100')]
seq_lens = [128]

def main():
    parser = argparse.ArgumentParser(description='Run evaluation of trajectory models')
    parser.add_argument('mode', type=str, nargs='?', default='eval', choices=['eval', 'summary'],
                       help='Mode to run in: eval (default) or summary')

    args = parser.parse_args()

    if args.mode == 'eval':
        eval()
    elif args.mode == 'summary':
        summary()

def summary():
    # Load stored results
    try:
        kls = pickle.load(open("kl_measures.pickle", "rb"))
    except FileNotFoundError:
        kls = {}

    try:
        difftraj_measures = pickle.load(open("difftraj_measures.pickle", "rb"))
    except FileNotFoundError:
        difftraj_measures = {}

    try:
        od_measures = pickle.load(open("od_measures.pickle", "rb"))
    except FileNotFoundError:
        od_measures = {}

    # Get all available synthetic datasets
    all_datasets = find_synthetic_datasets()

    print("\nResults Summary:")
    print("---------------")

    # Check what experiments have been run
    all_keys = set()
    for d in [kls, difftraj_measures, od_measures]:
        all_keys.update(d.keys())
        
    completed = []
    missing = []
    no_data = []
    
    for dataset_str in dataset_strs:
        for seq_len in seq_lens:
            for model, filter_out in models:
                if (dataset_str, seq_len, model) in ignore_combinations:
                    continue
                
                key = (dataset_str, seq_len, model, filter_out)
                
                # Check if synthetic data exists
                descs = list(filter(
                    lambda s: s.model_name == model and s.dataset == dataset_str and s.seq_len == seq_len, 
                    all_datasets))
                
                synthetic_exists = any(descs)
                if len(descs) > 1:
                    raise ValueError(f'Several registry entries for {key}')
                
                if not synthetic_exists:
                    no_data.append(key)
                    continue

                try:
                    load_synthetic(descs[0])
                except FileNotFoundError:
                    no_data.append(key)
                    continue
                
                if key in kls and key in difftraj_measures and key in od_measures:
                    completed.append(key)
                else:
                    missing_metrics = []
                    if key not in kls:
                        missing_metrics.append("KL")
                    if key not in difftraj_measures:
                        missing_metrics.append("DiffTraj")
                    if key not in od_measures:
                        missing_metrics.append("OD")
                    missing.append((key, missing_metrics))

    print("\nCompleted Experiments:")
    for c in completed:
        print(f"- Dataset: {c[0]}, Seq Length: {c[1]}, Model: {c[2]}, Filter Out: {c[3]}")
        
    print("\nMissing Results:")
    for m, metrics in missing:
        print(f"- Dataset: {m[0]}, Seq Length: {m[1]}, Model: {m[2]}, Filter Out: {m[3]}")
        print(f"  Missing metrics: {', '.join(metrics)}")
        
    print("\nNo Synthetic Data Available:")
    for n in no_data:
        print(f"- Dataset: {n[0]}, Seq Length: {n[1]}, Model: {n[2]}, Filter Out: {n[3]}")

def eval():
    os.makedirs('figures', exist_ok=True)

    all_datasets = find_synthetic_datasets()
    pprint(all_datasets)

    try:
        kls = pickle.load(open("kl_measures.pickle", "rb"))
    except FileNotFoundError:
        kls = {}

    try:
        difftraj_measures = pickle.load(open("difftraj_measures.pickle", "rb"))
    except FileNotFoundError:
        difftraj_measures = {}

    try:
        od_measures = pickle.load(open("od_measures.pickle", "rb"))
    except FileNotFoundError:
        od_measures = {}

    train_datas = {}

    from noise_floors import compute_noise_floors, NOISE_FLOOR_KEY
    filter_outs_seen = set()
    for _, filter_out in models:
        filter_outs_seen.add(filter_out)
    for dataset_str in dataset_strs:
        for seq_len in seq_lens:
            for filter_out in filter_outs_seen:
                floor_key = (dataset_str, seq_len, NOISE_FLOOR_KEY, filter_out)
                if floor_key in kls and floor_key in difftraj_measures and floor_key in od_measures:
                    continue
                if (dataset_str, seq_len, filter_out) not in train_datas:
                    train_datas[(dataset_str, seq_len, filter_out)] = load_train_data(dataset_str, seq_len, filter_out)
                train_data, extent_meters_yx = train_datas[(dataset_str, seq_len, filter_out)]
                print("Computing noise floors for:", (dataset_str, seq_len, filter_out))
                kls_f, difftraj_f, od_f = compute_noise_floors(train_data, extent_meters_yx, seq_len)
                kls[floor_key] = kls_f
                difftraj_measures[floor_key] = difftraj_f
                od_measures[floor_key] = od_f

    for dataset_str in dataset_strs:
        for seq_len in seq_lens:
            for model, filter_out in models:
                key = (dataset_str, seq_len, model, filter_out)
                done = key in kls and key in difftraj_measures and key in od_measures
                if done:
                    print("Skipping", *key)
                    continue     

                print(*key)

                plot = list(filter(
                    lambda s: s.model_name == model and s.dataset == dataset_str and s.seq_len == seq_len, 
                    all_datasets))

                if len(plot) == 0:
                    print("No data for", model, seq_len, dataset_str)
                    continue
                elif len(plot) > 1:
                    raise ValueError(f'Several registry entries for {key}')

                if (dataset_str, seq_len, filter_out) not in train_datas:
                    train_datas[(dataset_str, seq_len, filter_out)] = load_train_data(dataset_str, seq_len, filter_out)

                train_data, extent_meters_yx = train_datas[(dataset_str, seq_len, filter_out)]

                synth_desc = plot[0]
                try:
                    synth = load_synthetic(synth_desc)
                except FileNotFoundError:
                    print("No synthetic data for", model, seq_len, dataset_str)
                    continue

                synth = filter_data(synth, filter_out)

                if filter_out == 'lower-left-quarter':
                    # Plot the train and synth data to validate that filter is applied correctly
                    f, ax = plt.subplots(figsize=(20, 10), ncols=2)
                    for traj in train_data:
                        ax[0].plot(traj[:, 1], traj[:, 0])
                    for traj in synth:
                        ax[1].plot(traj[:, 1], traj[:, 0])
                    plt.savefig(f'figures/{model}_{dataset_str}_{seq_len}_{filter_out}.png')
                    plt.close()

                if key not in kls:
                    print("Calculating various divergence metrics for key:", key)
                    res = 256
                    original_hm, _, _ = bayesian_smoothed_heatmap(train_data, res=res, range=[[0, 1], [0, 1]])
                    synthetic_hm, _, _ = bayesian_smoothed_heatmap(synth, res=res, range=[[0, 1], [0, 1]])
                    kl_synth_real = KL(synthetic_hm, original_hm)
                    kl_real_synth = KL(original_hm, synthetic_hm)
                    kl_symmetric = (kl_synth_real + kl_real_synth) / 2
                    M = (original_hm + synthetic_hm) / 2
                    js = 0.5 * KL(original_hm, M) + 0.5 * KL(synthetic_hm, M)
                    # print(model, seq_len, kl_synth_real, kl_real_synth, kl_symmetric, js)
                    kls[key] = (kl_synth_real, kl_real_synth, kl_symmetric, js)

                if key not in difftraj_measures:
                    print("Calculating difftraj measures for:", key)
                    train_subset = get_random_subset(train_data)
                    de = density_error(train_subset, synth)
                    te = trip_error(train_subset, synth)
                    top_n_score = top_n(train_subset, synth, bins=16, n=100)

                    # Index 2 is Length, which length_error_v2.py computes; the table generators fill it in
                    difftraj_measures[key] = de, te, float('nan'), top_n_score

                if key not in od_measures:
                    print("Calculating OD measure for:", key)
                    train_od = get_random_subset(train_data, n=min(30000, len(train_data)))
                    od_measures[key] = od_error(train_od, synth, extent_meters_yx)


    pickle.dump(kls, open("kl_measures.pickle", 'wb'))
    pickle.dump(difftraj_measures, open("difftraj_measures.pickle", 'wb'))
    pickle.dump(od_measures, open("od_measures.pickle", 'wb'))

if __name__ == '__main__':
    main()
