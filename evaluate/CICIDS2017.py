import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

"""
CICIDS2017s数据集，提取统计特征
"""
class CICIDS2017Statistic():
    def __init__(self):
        print("CICIDS2017 Statistic Init")
        self.config =  {
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
        'num': 10000
    },
    "PortScan": {
        "path": '/home/sunhanwu/datasets/cicids2017/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        'label': ['PortScan'],
        'tag': 3,
        'num': 10000
    },
    "BruteForce": {
        "path": '/home/sunhanwu/datasets/cicids2017/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv',
        'label': ['FTP-Patator', 'SSH-Patator'],
        'tag': 4,
        'num': 10000
    },
    "DDoS": {
        "path": '/home/sunhanwu/datasets/cicids2017/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        'label': ['DDoS'],
        'tag': 5,
        'num': 10000
    }
}
        self.load_data()

    def load_data(self):
        self.data = {}
        for cls in self.config:
            print("load data from {}".format(self.config[cls]['path']))
            data = pd.read_csv(self.config[cls]['path'])
            benign = data[data[' Label'] == 'BENIGN']
            benign[' Label'] = 0
            benign = benign.iloc[:self.config[cls]['num'], 1:]
            malware = data[data[' Label'].isin(self.config[cls]['label'])]
            malware.loc[malware[' Label'].isin(self.config[cls]['label']), ' Label'] = self.config[cls]['tag']
            malware = malware.iloc[:self.config[cls]['num'], 1:]
            self.data[cls] = np.vstack([benign, malware])
            np.save("/home/sunhanwu/datasets/cicids2017/npy/" + cls + ".npy", self.data[cls])
            print("{} done: total num {}".format(cls, len(self.data[cls])))
    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


"""
CICIDS2017s数据集，提取序列特征
"""
class CICIDS2017_Sequence(Dataset):
    def __init__(self):
        print("test 2")
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

if __name__ == '__main__':
    D = CICIDS2017Statistic()