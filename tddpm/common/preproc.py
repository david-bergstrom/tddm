import numpy as np
import math


def minmaxscale(data, debug=False):
    if isinstance(data, list):
        """
        (N, T, D), where T might constant or not
        """
        d_min = np.min([np.min(a, axis=0) for a in data], axis=0)
        d_max = np.max([np.max(a, axis=0) for a in data], axis=0)

        scaled = []

        for arr in data:
            scaled.append((arr - d_min) / (d_max - d_min))

        return scaled
    elif isinstance(data, np.ndarray) and len(data.shape) == 3:
        """
        (N, T, D), where T might be constant
        """
        d_min = np.min(data, axis=(0, 1), keepdims=True)
        d_max = np.max(data, axis=(0, 1), keepdims=True)
        return (data - d_min) / (d_max - d_min)
    else:
        raise NotImplementedError("Could not figure out how to min-max scale the data")


def split(data, proportions):
    proportions = np.array(proportions)
    proportions = proportions / proportions.sum()

    if isinstance(data, np.ndarray) and data.shape[0] == 1:
        real_indices = proportions.cumsum() * data.shape[1]
        int_indices = real_indices.astype(int)
        int_indices = list(int_indices)
        int_indices.insert(0, 0)

        parts = []

        for start, stop in zip(int_indices, int_indices[1:]):
            parts.append(data[:, start:stop])
    else:
        real_indices = proportions.cumsum() * len(data) 
        int_indices = real_indices.astype(int)
        int_indices = list(int_indices)
        int_indices.insert(0, 0)

        parts = []

        for start, stop in zip(int_indices, int_indices[1:]):
            parts.append(data[start:stop])

    return parts


def partition_indices_for_trajectory(trajectory_len, partition_description):
    if partition_description['type'] == 'test_fraction':
        test_fraction = partition_description['fraction']
        test_partition_length = partition_description['test_partition_size']
        n_test_partitions = int(math.ceil((test_fraction * trajectory_len) / test_partition_length))
    
    n_train_partitions = n_test_partitions + 1
    n_partitions = n_train_partitions + n_test_partitions

    train_partition_length = (trajectory_len - (test_partition_length * n_test_partitions)) // n_train_partitions

    train_partitions = []
    test_partitions = []

    start_idx = 0

    for i in range(n_partitions-1):
        if i % 2 == 0:
            end_idx = start_idx + train_partition_length
            train_partitions.append((start_idx, end_idx))
        else:
            end_idx = start_idx + test_partition_length
            test_partitions.append((start_idx, end_idx))
        start_idx = end_idx

    train_partitions.append((start_idx, trajectory_len))

    return train_partitions, test_partitions

