"""Real-vs-real noise floors for each evaluation metric.

For every JSD/divergence-style metric we report, comparing two finite samples
drawn from the *same* (real) distribution gives a positive value due to
sample noise. That value is the floor below which a real-vs-synthetic score
is indistinguishable from "perfect synthetic".

The protocol follows each metric's eval setup (same subset sizes and cells),
with one half of the real windows as the reference and the other half in place
of the synthetic set.

The split is over the preprocessed 128-step windows (data/preprocessed/<dataset>-100000.npy).
Windows cut from the same trajectory can land in both halves, so the halves are
disjoint but not fully independent.
"""
import numpy as np

from common import SlidingDataset


def compute_noise_floors(train_data, extent_meters_yx, seq_len=128, seed=20260503):
    from run_evaluation import (
        bayesian_smoothed_heatmap, KL, density_error, trip_error,
        top_n, od_error, get_random_subset,
    )

    raw = train_data.data
    n_total = len(raw)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_total)
    half = n_total // 2
    a_ds = SlidingDataset(raw[idx[:half]], seq_len)
    b_ds = SlidingDataset(raw[idx[half:]], seq_len)

    res = 256
    a_hm, _, _ = bayesian_smoothed_heatmap(a_ds, res=res, range=[[0, 1], [0, 1]])
    b_sub_kl = get_random_subset(b_ds, n=min(10000, len(b_ds)), seed=seed + 1)
    b_hm, _, _ = bayesian_smoothed_heatmap(b_sub_kl, res=res, range=[[0, 1], [0, 1]])
    kl_sr = KL(b_hm, a_hm)
    kl_rs = KL(a_hm, b_hm)
    kl_sym = (kl_sr + kl_rs) / 2
    M = (a_hm + b_hm) / 2
    js_val = 0.5 * KL(a_hm, M) + 0.5 * KL(b_hm, M)
    kls_floor = (kl_sr, kl_rs, kl_sym, js_val)

    a_sub = get_random_subset(a_ds, n=min(3000, len(a_ds)), seed=seed + 2)
    b_sub = get_random_subset(b_ds, n=min(10000, len(b_ds)), seed=seed + 3)
    de = density_error(a_sub, b_sub)
    te = trip_error(a_sub, b_sub)
    tn = top_n(a_sub, b_sub, bins=16, n=100)
    # Index 2 is Length, whose floor compute_v2_noise_floors.py computes; the table generators fill it in
    difftraj_floor = (de, te, float('nan'), tn)

    a_sub_od = get_random_subset(a_ds, n=min(30000, len(a_ds)), seed=seed + 4)
    b_sub_od = get_random_subset(b_ds, n=min(10000, len(b_ds)), seed=seed + 5)
    od_floor = od_error(a_sub_od, b_sub_od, extent_meters_yx)

    return kls_floor, difftraj_floor, od_floor


NOISE_FLOOR_KEY = '__noise_floor__'


def compute_v2_floors(train_data, extent_meters_yx, seq_len=128, seed=20260503):
    """Floors for Length (length_error_v2.py) and Speed KL, with the split and
    subsets of `compute_noise_floors`: Length compares 3,000 windows of half A
    (the evaluation's real reference size) with 10,000 of half B; Speed KL
    compares all of half A (the evaluation uses all real windows) with the
    same 10,000 of half B, as the KL floor.

    Returns (length_v2, speed_kl_v2), where speed_kl_v2 is the
    (kl_sr, kl_rs, kl_sym, js) tuple stored by speed_kl_divergence.py.
    """
    from run_evaluation import get_random_subset
    from length_error_v2 import compute_length_error
    from speed_kl_divergence import compute_speed_kl_divergence

    raw = train_data.data
    idx = np.random.default_rng(seed).permutation(len(raw))
    half = len(raw) // 2
    a_ds = SlidingDataset(raw[idx[:half]], seq_len)
    b_ds = SlidingDataset(raw[idx[half:]], seq_len)
    a_sub = get_random_subset(a_ds, n=min(3000, len(a_ds)), seed=seed + 2)
    b_sub = get_random_subset(b_ds, n=min(10000, len(b_ds)), seed=seed + 3)
    length_v2, _ = compute_length_error(a_sub, b_sub, extent_meters_yx)
    speed_kl_v2 = compute_speed_kl_divergence(np.asarray(raw[idx[:half]], dtype=np.float64), b_sub,
                                              extent_meters_yx)
    return float(length_v2), tuple(float(x) for x in speed_kl_v2)


def compute_tstr_scaled_diff_floor(train_data, seq_len=128, seed=20260503, n_train=10000,
                                   predictor_seeds=(0, 1, 2)):
    """Floor for the scaled-difference TSTR: the same predictor trained on
    min(n_train, |A|) windows of real half A (the size of a generated set)
    and tested on every window of real half B, with the split of
    `compute_noise_floors`. Returns the per-window MAEs on half B, averaged
    over the predictor seeds, so its mean is the mean over seeds.
    """
    import torch
    from run_evaluation import get_random_subset
    from eval.transformer_tstr import transformer_tstr_scaled_diff

    raw = train_data.data
    idx = np.random.default_rng(seed).permutation(len(raw))
    half = len(raw) // 2
    a_ds = SlidingDataset(raw[idx[:half]], seq_len)
    b = np.asarray(raw[idx[half:]], dtype=np.float32)
    a_sub = get_random_subset(a_ds, n=min(n_train, len(a_ds)), seed=seed + 6).astype(np.float32)
    losses = [transformer_tstr_scaled_diff(b, a_sub, seed=s) for s in predictor_seeds]
    return torch.stack(losses).mean(dim=0)
