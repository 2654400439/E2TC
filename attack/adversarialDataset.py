"""
Date: 2022-03-08
Author: sunhanwu@iie.ac.cn
Desc: adversarial dataset
"""

import torch
from torch.utils.data import Dataset
import numpy as np

class AdversarialC2Data(Dataset):
    """
    adversarial sample for attack
    """
    def __init__(self, filename, target_class=5, keep_target=True, norm=False):
        """

        :param filename:
        :param target_class:
        :param keep_target: Whether to keep target data
        """
        data = np.load(filename)
        if norm:
            data[:,:-1] = (data[:,:-1] - np.min(data[:,:-1], axis=0)) / (np.max(data[:,:-1], axis=0) - np.min(data[:,:-1], axis=0)) + 1e-9
        print("Adversarial Dataset Load: {}".format(filename))
        if keep_target:
            self.data = data
        else:
            self.data = np.array([x for x in data if x[-1] != target_class])
        print(self.data.shape)

    def __getitem__(self, index):
        data = self.data[index]
        sample = [x if x < 1600 else 1599 for x in data[:-1]]
        sample = np.array(sample)
        sample = torch.from_numpy(sample)
        label = torch.from_numpy(data[-1:])
        return sample, label

    def __len__(self):
        return self.data.shape[0]
