"""
Date: 2022-04-12
Author: sunhanwu@iie.ac.cn
Desc: target model: DNN
"""

import torch
from torch import nn
from TargetModel.FSNet.dataset import C2Data
from torch.utils.data import DataLoader
from TargetModel.FSNet.train import computeFPR
from TargetModel.FSNet.utils import save_model
from sklearn.metrics import confusion_matrix
from utils.MTACICFlowMeter import MTACICFlowMeter
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


class TargetDNN(nn.Module):
    """
    target DNN model
    """
    def __init__(self, param):
        """

        :param param:
        """
        super(TargetDNN, self).__init__()
        self.input_size = param['input_size']
        self.class_num = param['num_class']

        self.linear1 = nn.Linear(self.input_size, int(self.input_size / 2))
        self.linear2 = nn.Linear(int(self.input_size / 2), int(self.input_size / 4))
        self.linear3 = nn.Linear(int(self.input_size / 4), self.class_num)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return x


if __name__ == '__main__':
    # hyper param
    epoch_size=100
    batch_size = 128
    lr = 1e-4

    # model param
    param = {
        "input_size": 77,
        "num_class": 2
    }
    Botnets = [
        "Tofsee",
        "Dridex",
        "Quakbot",
        "TrickBot",
        "Gozi"
    ]
    numbers = [2580, 2580, 1600, 690, 1250]

    index = 4

    sample_szie = numbers[index]
    botname = Botnets[index]
    normal = "CTUNone"
    arch = "dnn"

    # c2data = C2Data(botname, number=sample_szie, sequenceLen=30)
    c2data = MTACICFlowMeter(botname, number=sample_szie)
    total_size = len(c2data)

    # total_size = sample_szie * 2
    test_size = int(total_size * 0.2)
    train_size = int((total_size - test_size) * 0.8)
    valid_size = total_size - test_size - train_size
    print("train data: {}".format(train_size))
    print("valid data: {}".format(valid_size))
    print("test data: {}".format(test_size))

    # use GPU if it is available, oterwise use cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pre the dataloader
    train_valid_data, test_data = torch.utils.data.random_split(c2data, [train_size + valid_size, test_size])
    train_data, valid_data = torch.utils.data.random_split(train_valid_data, [train_size, valid_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=False)

    # model

    dnn = TargetDNN(param)
    dnn.to(device)

    # loss func
    crossEntropy = torch.nn.CrossEntropyLoss()
    adam = torch.optim.Adam(dnn.parameters(), lr=lr)
    # lossFunc = torch.nn.KLDivLoss()

    # trainning
    for i in range(epoch_size):
        dnn.train()
        loss_list = []
        acc_list = []
        recall_list = []
        f1_list = []
        for batch_x, batch_y in tqdm(train_loader):
            batch_x = batch_x.to(device, dtype=torch.float)
            batch_y = batch_y.to(device)
            output = dnn(batch_x)
            # output.shape = (batch_size, sequence, num_class)
            acc, recall, f1 = computeFPR(y_pred=output, y_target=batch_y)
            # ipdb.set_trace()
            batch_y = batch_y.squeeze()
            # batch_y = F.softmax(batch_y)
            # output = F.softmax(output)
            loss = crossEntropy(output, batch_y)

            acc_list.append(acc)
            recall_list.append(recall)
            f1_list.append(f1)
            loss_list.append(loss.item())

            adam.zero_grad()
            loss.backward()
            adam.step()
        print("[Training {:03d}] acc: {:.2%}, recall: {:.2%}, f1: {:.2%}, loss: {:.2f}".format(i + 1,
                                                                                               np.mean(acc_list),
                                                                                               np.mean(recall_list),
                                                                                               np.mean(f1_list),
                                                                                               np.mean(loss_list)))

        # validing
        dnn.eval()
        loss_list = []
        acc_list = []
        recall_list = []
        f1_list = []
        for batch_x, batch_y in valid_loader:
            batch_x = batch_x.to(device, dtype=torch.float)
            batch_y = batch_y.to(device)
            output = dnn(batch_x)
            # output.shape = (batch_size, sequence, num_class)
            acc, recall, f1 = computeFPR(y_pred=output, y_target=batch_y)
            batch_y = batch_y.squeeze()
            # batch_y = F.softmax(batch_y)
            # output = F.softmax(output)
            loss = crossEntropy(output, batch_y)

            acc_list.append(acc)
            recall_list.append(recall)
            f1_list.append(f1)
            loss_list.append(loss.item())
        print("[Validing {:03d}] acc: {:.2%}, recall: {:.2%}, f1: {:.2%}, loss: {:.2f}".format(i + 1,
                                                                                               np.mean(acc_list),
                                                                                               np.mean(recall_list),
                                                                                               np.mean(f1_list),
                                                                                               np.mean(loss_list)))

    # testing
    dnn.eval()
    loss_list = []
    acc_list = []
    recall_list = []
    f1_list = []
    y_true = []
    y_pred = []
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device, dtype=torch.float)
        batch_y = batch_y.to(device)
        output = dnn(batch_x)
        # output.shape = (batch_size, sequence, num_class)
        acc, recall, f1 = computeFPR(y_pred=output, y_target=batch_y)
        batch_y = batch_y.squeeze()
        # batch_y = F.softmax(batch_y)
        # output = F.softmax(output)
        loss = crossEntropy(output, batch_y)

        acc_list.append(acc)
        recall_list.append(recall)
        f1_list.append(f1)
        loss_list.append(loss.item())
        y_true += batch_y.detach().cpu().numpy().tolist()
        y_pred += torch.argmax(output, dim=1).detach().cpu().numpy().tolist()
    print("[Testing {:03d}] acc: {:.2%}, recall: {:.2%}, f1: {:.2%}, loss: {:.2f}".format(i + 1,
                                                                                          np.mean(acc_list),
                                                                                          np.mean(recall_list),
                                                                                          np.mean(f1_list),
                                                                                          np.mean(loss_list)))
    print(confusion_matrix(y_true, y_pred))
    FPR = {
        'acc': np.mean(acc_list),
        'recall': np.mean(recall_list),
        'f1': np.mean(f1_list),
        'metrix': confusion_matrix(y_true,y_pred)
    }
    hyper = {
        'epoch_size': epoch_size,
        'lr': lr,
        'batch_size': batch_size
    }
    filename = "../modelFile/proxy_mta_cicflowmeter_{}_{}.pkt".format(arch, botname)
    save_model(dnn, adam, param, hyper, FPR, filename)
