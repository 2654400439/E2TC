import torch
from torch import nn
from TargetModel.FSNet.dataset import C2Data
from torch.utils.data import DataLoader
from TargetModel.FSNet.train import computeFPR
from TargetModel.FSNet.utils import save_model
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

class DNN(nn.Module):
    """

    """
    def __init__(self, param):
        self.input_size = param['input_size']
        self.class_num = param['class_num']
        self.linear1 = nn.Linear(self.input_size, 2048)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear3 = nn.Linear(1024, 512)
        self.linear4 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        x = F.relu(self.linear3(x))
        x = self.dropout(x)
        x = F.relu(self.linear4(x))
        return x


if __name__ == '__main__':
    epoch_size = 100
    batch_size = 128
    lr = 1e-4

    param = {
        "input_size": 30,
        "num_class": 2
    }
