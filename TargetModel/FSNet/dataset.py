"""
Date: 2022-03-03
Author: sunhanwu@iie.ac.cn
Desc: dataset and dataloader for fsnet
      5 malicious botnet + 1 normal
"""
import os.path

import torch
from torch.utils.data import Dataset
import numpy as np

"""
MTA Dataset
"""
DATA_PATH = [
    ("1","/home/sunhanwu/datasets/MTA/text/Tofsee"),
    ("1", "/home/sunhanwu/datasets/MTA/text/Dridex"),
    ("1", "/home/sunhanwu/datasets/MTA/text/Quakbot"),
    ("1", "/home/sunhanwu/datasets/MTA/text/TrickBot"),
    ("1", "/home/sunhanwu/datasets/MTA/text/Gozi"),
    ("0", "/home/sunhanwu/datasets/MTA/text/CTUNone"),
    # ("0", "/home/sunhanwu/datasets/MTA/text/ISCXNone"),
    # ("0", "/home/sunhanwu/datasets/MTA/text/MTANone"),
]

"""
CICMAlAnal2017 Dataset
DATA_PATH = [
    ("0","/home/sunhanwu/datasets/CICMalAnal2017/Adware-Dowgin/text"),
    ("0", "/home/sunhanwu/datasets/CICMalAnal2017/Ransomware-Charger/text"),
    ("0", "/home/sunhanwu/datasets/CICMalAnal2017/Scareware-AVforandroid/text"),
    ("0", "/home/sunhanwu/datasets/CICMalAnal2017/SMSMalware-Nandrobox/text"),
    ("1", "/home/sunhanwu/datasets/CICMalAnal2017/Benign/text2"),
]
"""

class C2Data(Dataset):
    def __init__(self, botname, number=200, sequenceLen=40):
        self.data = []
        for label, datapath in DATA_PATH:
            if label == "1" and (botname not in datapath):
                continue
            print("Load Data: {}".format(datapath.split('/')[-1]))
            label = int(label)
            self.data += self.getSequenceData(datapath, label, number, sequenceLen)
        print("Total Number: {}".format(len(self.data)))

    def getSequenceData(self, datapath, label, number, sequenceLen):
        """
        Parse the "number" of sequence data from the datapath
        :param datapath: data path
        :param number: number of class samples
        :return: a list of tuple(consist of sequence and the label)
        """
        filenames = [os.path.join(datapath, x) for x in os.listdir(datapath)]
        fileNum = len(filenames)
        random_index = np.random.choice(np.arange(fileNum), size=number, replace=True)
        choiced_files = [filenames[x] for x in random_index]
        data = []
        index = 0
        while index < number:
            random_index = np.random.choice(np.arange(fileNum), size=1, replace=True)
            file = filenames[random_index[0]]
            with open(file, 'r') as f:
                sequence = [int(x.strip().split('\t')[1]) for x in f.readlines()]
            sequence = [x if x < 1600 else 1599 for x in sequence]
            # padding
            if len(sequence) <= int(sequenceLen * 0.3):
                continue
            else:
                index += 1
            if len(sequence) > sequenceLen:
                sequence = sequence[:sequenceLen]
            else:
                sequence = sequence + [0] * (sequenceLen - len(sequence))
            sample = (sequence, [label])
            data.append(sample)
        return data

    def __getitem__(self, index):
        sequence, label = self.data[index]
        sequence = torch.tensor(sequence)
        label = torch.tensor(label)
        return sequence, label

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    c2data = C2Data()
