
import numpy as np
import torch
from torch.utils.data import Dataset

rng = np.random.default_rng()


class SlidingDataset(Dataset):
    def __init__(self, data, window_size):
        """
        data is a NxLxD where:
        - N is the number of observed sequences
        - L is the sequence length
        - D is the number of features

        If L varies for different sequences, data can be a list of NumPy
        arrays, each array with different dimensions. 

        window size is the length of the sliding window, tau in the PSA-GAN paper
        """
        self.data = data

        self.n_seq = len(self.data)
        self.window_size = window_size

        self.index_to_instance_window = []

        for i in range(self.n_seq):
            sequence = self.data[i]
            n_observations = sequence.shape[0]

            for j in range(0, n_observations - window_size + 1):
                self.index_to_instance_window.append((i, j))

        self.windows = len(self.index_to_instance_window)
                
    def __len__(self):
        return self.windows

    def __getitem__(self, idx):
        i, j = self.index_to_instance_window[idx]
        return self.data[i][j:j+self.window_size]


class NumpyDataset(Dataset):
    def __init__(self, data):
        self.n_samples = data.shape[0]
        self.seq_len = data.shape[1]
        self.n_dim = data.shape[2]
        # self.n_dim = 2**math.ceil(math.log2(data_n_dim))
        # self.data = torch.zeros([self.n_samples, 1, self.seq_len, self.n_dim])
        # self.data[:, 0, :, :data_n_dim] = torch.Tensor(data)
        self.data = torch.Tensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
