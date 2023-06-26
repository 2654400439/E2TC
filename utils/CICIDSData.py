import torch
from sklearn.utils import shuffle
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

dataconfig = {
    # "Benign": {
    #     "path": '/home/sunhanwu/datasets/cicids2017/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv',
    #     "label": ['BENIGN'],
    #     "tag": 0
    # },
    "Botnet": {
        "path": '/home/sunhanwu/datasets/cicids2017/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv',
        'label': ['Bot'],
        'tag': 1,
        'num': 1966
    },
    "Fuzzing": {
        "path": '/home/sunhanwu/datasets/cicids2017/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv',
        'label': ['DoS Hulk'],
        'tag': 2,
        'num': 231073
    },
    "PortScan": {
        "path": '/home/sunhanwu/datasets/cicids2017/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        'label': ['PortScan'],
        'tag': 3,
        'num': 127537
    },
    "BruteForce": {
        "path": '/home/sunhanwu/datasets/cicids2017/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv',
        'label': ['FTP-Patator', 'SSH-Patator'],
        'tag': 4,
        'num': 13835
    },
    "DDoS": {
        "path": '/home/sunhanwu/datasets/cicids2017/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        'label': ['DDoS'],
        'tag': 5,
        'num': 97718
    }
}

class CICIDS(Dataset):
    """
    CICIDS 2017 dataset
    """

    def __init__(self, name, norm=True):
        """

        :param name:
        :param num: 0 diy; 1 dataconfig
        :param norm:
        """
        print("load data form {}".format(name))
        assert name in dataconfig.keys()
        data = pd.read_csv(dataconfig[name]['path'])
        X1 = shuffle(data[data[' Label'].isin(['BENIGN'])]).iloc[:dataconfig[name]['num'], :]
        X2 = shuffle(data[data[' Label'].isin(dataconfig[name]['label'])]).iloc[:dataconfig[name]['num'], :]
        X = X1.append(X2)
        X = shuffle(X)
        X.loc[X[' Label'].isin(dataconfig[name]['label']), ' Label'] = 1
        X.loc[X[' Label'].isin(['BENIGN']), ' Label'] = 0
        X = X.iloc[:, 1:]
        if norm:
            max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-9)
            X.iloc[:, :-1] = X.iloc[:, :-1].apply(max_min_scaler)
        X = X.dropna()
        self.data = X.to_numpy()
        if len(self.data) >= 10000:
            self.data = self.data[:10000, :]

    def __getitem__(self, index):
        item = self.data[index]
        X = torch.tensor(item[:-1].tolist())
        y = torch.tensor([item[-1]])
        return X, y


    def __len__(self):
        return self.data.shape[0]
