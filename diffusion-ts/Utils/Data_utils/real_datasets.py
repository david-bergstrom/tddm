import os
import torch
import numpy as np
import pandas as pd

from scipy import io
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.masking_utils import noise_mask

import sys
sys.path.append("../")
from data import load_dataset_from_str, minmaxscale, rolling_window


class BenchmarkDataset(Dataset):
    def __init__(
        self, 
        name='',
        window=128, 
        num=30000, 
        dim=12, 
        save2npy=True, 
        neg_one_to_one=True, 
        seed=123,
        period='train',
        output_dir='./OUTPUT',
        predict_length=None,
        missing_ratio=None,
        mean_mask_length=3
    ):
        super(BenchmarkDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'

        if period == 'train':
            assert ~(predict_length is not None or missing_ratio is not None), ''

        assert predict_length is None
        assert missing_ratio is None
        assert name != ''

        self.pred_len, self.missing_ratio = predict_length, missing_ratio

        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        self.samples = load_dataset_from_str(name, prefix='../data/')

        self.auto_norm = neg_one_to_one

        if neg_one_to_one:
            self.samples = minmaxscale(self.samples, -1, 1)
        else:
            self.samples = minmaxscale(self.samples, 0, 1)

        self.samples = rolling_window(self.samples, window)

        self.var_num = self.samples.shape[2]
        self.sample_num = self.samples.shape[0]
        self.window = window

        self.period, self.save2npy = period, save2npy

        if period == 'test':
            raise NotImplementedError()

    def normalize(self, rawdata):
        raise NotImplementedError()

    def unnormalize(self, data):
        raise NotImplementedError()
    
    def mask_data(self, seed=2023):
        raise NotImplementedError()

    def __getitem__(self, ind):
        if self.period == 'test':
            raise NotImplementedError()
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num
