try:
    from .data_generation import extract_signals, region_occupancy
except ImportError:
    from data_generation import extract_signals, region_occupancy
import matplotlib.pyplot as plt
import torch
import numpy as np

def undo_rotation_and_scaling(traj, region):
    """
    Undos the 0 to 1 scaling done by data_generation.crop_signal_with_rotation. The 
    region object is a tuple consisting of (lat, long, rotation, width (difference in lat), height (difference )). NB rotation is in degrees.
    """
    origin = region[0:2]
    rotation = region[2]
    size = region[3:5]
    
    rotation = rotation * np.pi / 180
    
    rotated_signal = np.zeros_like(traj)
    rotated_signal[:, 0] = np.cos(-rotation) * traj[:, 0] + np.sin(rotation) * traj[:, 1]
    rotated_signal[:, 1] = np.sin(-rotation) * traj[:, 0] + np.cos(rotation) * traj[:, 1]

    signal = rotated_signal * size
    signal = signal + origin
    return signal

def unscaled_occupancy(trajectories, res_x, res_y):
    """
    Unscaled variant of data_generation.region_occupancy, mainly useful for visualization.
    """
    all_concatenated = np.concatenate(trajectories)
    H, xedges, yedges = np.histogram2d(all_concatenated[:, 0], all_concatenated[:, 1], bins=(res_x, res_y), density=True)

    return H, xedges, yedges


def generate_heatmap(samples, res=64):
    samples_concat = np.concatenate(samples.transpose(0, 2, 1))
    fake_heatmap, _, _ = np.histogram2d(samples_concat[:, 0], samples_concat[:, 1], bins=(res, res), range=((0, 1), (0, 1)), density=True)
    return fake_heatmap


def sample_heatmap(diffusion, query_heatmap, n_samples, sample_batch_size=8):
    """
    Given a diffusion model and a heatmap, generate n_samples conditioned on the heatmap.
    """
    if isinstance(query_heatmap, np.ndarray):
        query_heatmap = torch.Tensor(query_heatmap)
    query_heatmap = query_heatmap.unsqueeze(0)
    all_samples = []

    n_batches = n_samples // sample_batch_size
    remainder = n_samples % sample_batch_size

    def sample(n_samples):
        condition = query_heatmap.unsqueeze(0).expand(n_samples, 1, 64, 64).float().cuda()
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


def sample_synthetic_region(diffusion, dataset, region, n_samples, sample_batch_size=64):
    """
    Extract subtrajectories from region, creates a heatmap of the real data, creates synthetic
    trajectories conditioned on the heatmap and finally calculate a heatmap of the synthetic samples.
    """
    assert n_samples > sample_batch_size

    trajectories_in_region = extract_signals(dataset, region)  # TODO: Set minimum length to same as the model was trained on
    heatmap, _, _ = region_occupancy(trajectories_in_region, res=64)

    samples = sample_heatmap(diffusion, heatmap, n_samples=n_samples, sample_batch_size=sample_batch_size)
    synthetic_heatmap = generate_heatmap(samples)

    return trajectories_in_region, heatmap, samples, synthetic_heatmap



def sample_validation_regions(diffusion, val_heatmaps, conditional_dataset):
    diffusion.eval()

    results = []

    for heatmap in val_heatmaps:
        heatmap = (torch.Tensor(heatmap) - conditional_dataset.heatmaps_mean) / conditional_dataset.heatmaps_std
        samples = sample_heatmap(diffusion, heatmap, n_samples=256, sample_batch_size=128)
        synthetic_heatmap = generate_heatmap(samples)

        results.append((samples, synthetic_heatmap))

    diffusion.train()

    f, axs = plt.subplots(figsize=(17, 30), ncols=6, nrows=len(val_heatmaps), sharey=True, sharex=True, constrained_layout=True)
    js_dists = []

    for (ax1, ax2, ax3, ax4, ax5, ax6), heatmap, (samples, synthetic_heatmap) in zip(axs, val_heatmaps, results):

        kl_synth_real = synthetic_heatmap * (np.log(synthetic_heatmap + 1e-9) - np.log(heatmap + 1e-9))
        actual_kl_synth_real = np.nansum(kl_synth_real)

        kl_real_synth = heatmap * (np.log(heatmap + 1e-9) - np.log(synthetic_heatmap + 1e-9))
        actual_kl_real_synth = np.nansum(kl_real_synth)

        js_dist = (actual_kl_synth_real + actual_kl_real_synth) / 2
        js_dists.append(js_dist)

        ax1.imshow(heatmap, extent=[0, 1, 0, 1], origin='lower', interpolation=None)
        ax1.set_title("Input heatmap")
        ax2.plot(samples[:, 1, :].T, samples[:, 0, :].T, alpha=.20)
        ax2.set_title("Synthetic Trajectories")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_aspect('equal')
        ax3.imshow(synthetic_heatmap, extent=[0, 1, 0, 1], origin='lower', interpolation=None)
        ax3.set_title("Heatmap of synthetic data")
        ax4.set_title('synthetic * log(synthetic / input)')
        ax4.imshow(kl_synth_real, extent=[0, 1, 0, 1], origin='lower', interpolation=None)
        ax5.set_title('input * log(input / synthetic)')
        ax5.imshow(kl_real_synth, extent=[0, 1, 0, 1], origin='lower', interpolation=None)
        ax6.set_title(f"JS = {js_dist:.0f}")
        ax6.imshow((kl_synth_real + kl_real_synth) / 2, extent=[0, 1, 0, 1], origin='lower', interpolation=None)

    return np.mean(js_dists), np.std(js_dists), f, results


def frac_outside(samples, heatmap):
    _, _, seq_len = samples.shape
    mask = heatmap > 0
    indices = np.zeros_like(samples, dtype=int)
    hm_size, _ = heatmap.shape

    indices[:, 0, :] = np.digitize(samples[:, 0, :], np.linspace(0, 1, hm_size))
    indices[:, 1, :] = np.digitize(samples[:, 1, :], np.linspace(0, 1, hm_size))
    indices[indices == hm_size] = hm_size - 1

    mask_values = mask[indices[:, 0, :], indices[:, 1, :]]

    n_inside = np.sum(mask_values, axis=1)
    n_outside = seq_len - n_inside
    return np.mean(n_outside / seq_len)