import numpy as np
import torch
from torch.utils.data import Dataset
import os


class ConditionalDataset(Dataset):
    def __init__(self, path, use_fp16=False):
        """
        Args:
            path: Path to the .npz file
            use_fp16: If True, load from pre-converted -fp16.npz file to reduce RAM by 4x.
                     Raises FileNotFoundError if fp16 file doesn't exist.
                     Precision loss is negligible (<0.001% relative error).
        """
        if use_fp16:
            # Load from pre-converted fp16 file
            fp16_path = path.replace('.npz', '-fp16.npz')
            if not os.path.exists(fp16_path):
                raise FileNotFoundError(
                    f"FP16 dataset not found: {fp16_path}\n"
                    f"Create it first: a copy of {os.path.basename(path)} with 'regions' and 'trajectories' "
                    f"unchanged and 'heatmaps' cast to float16"
                )

            print(f"Loading fp16 dataset: {os.path.basename(fp16_path)}")
            npz = np.load(fp16_path)

            self.regions = npz['regions']
            self.heatmaps = npz['heatmaps']  # Already fp16
            self.trajectories = npz['trajectories']

            self.n_samples = self.trajectories.shape[0]
            self.seq_len = self.trajectories.shape[1]
            self.n_dim = self.trajectories.shape[2]

            self.regions = torch.Tensor(self.regions)
            self.heatmaps = torch.from_numpy(self.heatmaps)
            self.trajectories = torch.Tensor(self.trajectories)
        else:
            # Standard loading (full precision)
            npz = np.load(path)

            self.regions = npz['regions']
            self.heatmaps = npz['heatmaps']
            self.trajectories = npz['trajectories']

            self.n_samples = self.trajectories.shape[0]
            self.seq_len = self.trajectories.shape[1]
            self.n_dim = self.trajectories.shape[2]

            self.regions = torch.Tensor(self.regions)
            self.heatmaps = torch.Tensor(self.heatmaps)
            self.trajectories = torch.Tensor(self.trajectories)

        # Record which heatmapper was used to generate this dataset
        self.heatmapper_type = str(npz['heatmapper_type']) if 'heatmapper_type' in npz.files else 'legacy'
        self.heatmapper_meters_per_cell = int(npz['heatmapper_meters_per_cell']) if 'heatmapper_meters_per_cell' in npz.files else 1
        self.heatmapper_sigma = int(npz['heatmapper_sigma']) if 'heatmapper_sigma' in npz.files else 0

        # Keep float16 for fp16 datasets to save memory
        if use_fp16:
            self.heatmaps_mean = self.heatmaps.mean()
            self.heatmaps_std = self.heatmaps.std()
            self.heatmaps = (self.heatmaps - self.heatmaps_mean) / self.heatmaps_std
        else:
            self.heatmaps_mean = self.heatmaps.float().mean()
            self.heatmaps_std = self.heatmaps.float().std()
            self.heatmaps = (self.heatmaps.float() - self.heatmaps_mean) / self.heatmaps_std
        self.heatmaps = self.heatmaps.unsqueeze(1)
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Cast heatmap to float32 for model compatibility (model weights are float32)
        # This is cheap since we only convert one sample at a time
        heatmap = self.heatmaps[idx].float()
        trajectory = self.trajectories[idx]
        return heatmap, trajectory