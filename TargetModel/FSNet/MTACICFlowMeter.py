import torch
from sklearn.utils import shuffle
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


datapaths = {
    'Tofsee'    :   '/home/sunhanwu/datasets/MTA/cicflowcsv/Tofsee.csv',
    'Quakbot'   :   '/home/sunhanwu/datasets/MTA/cicflowcsv/Quakbot.csv',
    'Dridex'    :   '/home/sunhanwu/datasets/MTA/cicflowcsv/Dridex.csv',
    'Gozi'      :   '/home/sunhanwu/datasets/MTA/cicflowcsv/Gozi.csv',
    'TrickBot'  :   '/home/sunhanwu/datasets/MTA/cicflowcsv/TrickBot.csv'
}

class MTACICFlowMeter(Dataset):
    """
    MTA dataset and cicflowmeter features
    """
    def __init__(self, name, number, norm=True):
        """

        :param name:
        :param num: 0 diy; 1 dataconfig
        :param norm:
        """
        print("load data form {}".format(name))
        assert name in datapaths.keys()
        data = pd.read_csv(datapaths[name])
        benign = pd.read_csv("/home/sunhanwu/datasets/MTA/cicflowcsv/Benign.csv")
        X1 = shuffle(data).iloc[:number, :]
        X2 = shuffle(benign).iloc[:number, :]
        X = X1.append(X2)
        X = shuffle(X)
        if norm:
            max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-9)
            X.iloc[:, :-1] = X.iloc[:, :-1].apply(max_min_scaler)
        X = X.dropna()
        self.data = X.to_numpy()

    def __getitem__(self, index):
        item = self.data[index]
        X = torch.tensor(item[:-1].tolist())
        y = torch.tensor([item[-1]]).long()
        return X, y

    def __len__(self):
        return self.data.shape[0]
