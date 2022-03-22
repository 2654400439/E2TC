"""
Date: 2022-03-09
Author: sunhanwu@iie.ac.cn
Desc: collection dataset for trainning the subtitute model
"""
import torch
from torch.utils.data import Dataset
import numpy as np

class CollectionDataset(Dataset):
    """

    """
    def __init__(self, filename, sequence_len=50):
        self.data = np.load(filename, allow_pickle=True)
        self.sequence_len = sequence_len
        print(self.data.shape)

    def __getitem__(self, index):
        data = self.data[index]
        sample = data[:self.sequence_len]
        label = data[self.sequence_len:]
        sample = torch.tensor(sample, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.float)
        return sample, label

    def __len__(self):
        return self.data.shape[0]
