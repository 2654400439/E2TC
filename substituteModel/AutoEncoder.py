"""
Date: 2022-03-14
Author: sunhanwu@iie.ac.cn
Desc: autoencoder subtitute model for fsnet
"""
import torch
from torch import nn
import torch.nn.functional as F
from TargetModel.FSNet.dataset import C2Data
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from TargetModel.FSNet.train import computeFPR
from sklearn.metrics import confusion_matrix
from TargetModel.FSNet.train import save_model

class AutoEncoder(nn.Module):
    """

    """
    def __init__(self, param:dict):
        super(AutoEncoder, self).__init__()
        self.inputsSize = param['inputSize']
        self.middleSize = param['middleSize']
        self.classNum = param['classNum']
        self.encode_linear1 = nn.Linear(self.inputsSize, int(self.inputsSize / 2))
        self.encode_linear2 = nn.Linear(int(self.inputsSize / 2), self.middleSize)
        self.decode_linear1 = nn.Linear(self.middleSize, int(self.inputsSize / 2))
        self.decode_linear2 = nn.Linear(int(self.inputsSize / 2), self.inputsSize)
        self.classify = nn.Linear(self.middleSize, self.classNum)

    def encode(self, inputs):
        """

        :param inputs: inputs.shape=(batch_size, inputSize)
        :return: (batch_size, middleSize)
        """
        inputs = inputs.float()
        x = F.relu(self.encode_linear1(inputs))
        x = F.relu(self.encode_linear2(x))
        return x

    def decode(self, inputs):
        """

        :param inputs: inputs.shape=(batch_size, middleSize)
        :return: (batch_size, inputSize)
        """
        x = F.relu(self.decode_linear1(inputs))
        x = self.decode_linear2(x)
        return x

    def trainning(self, inputs):
        """

        :param inputs:
        :return:
        """
        middle = self.encode(inputs)
        input_recover = self.decode(middle)
        output = self.classify(middle)
        return output, input_recover

    def forward(self, inputs):
        """

        :param inputs:
        :return:
        """
        middle = self.encode(inputs)
        output = self.classify(middle)
        return output

if __name__ == '__main__':
    # hyper param
    batch_size = 128
    lr = 1e-3
    epoch_size = 20

    # model param
    param = {
        'inputSize': 30,
        'middleSize':10,
        'classNum':2
    }

    botname = "Dridex"
    normal = "CTUNone"
    arch = "autoencoder"

    # use GPU if it is available, oterwise use cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pre the dataloader
    # c2data = CollectionDataset('../adversarialData/collectionData.npy', sequence_len=40)
    c2data = C2Data(botname, number=8000, sequenceLen=30)
    train_valid_data, test_data = torch.utils.data.random_split(c2data, [12800, 3200])
    train_data, valid_data = torch.utils.data.random_split(train_valid_data, [10000, 2800])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=False)

    # model
    autoencoder = AutoEncoder(param)
    autoencoder.to(device)

    # loss func
    loss_func_classify = torch.nn.CrossEntropyLoss()
    loss_func_recover = torch.nn.MSELoss()

    adam =  torch.optim.Adam(autoencoder.parameters(), lr)

    # training
    for i in range(epoch_size):
        autoencoder.train()
        loss_list = []
        acc_list = []
        recall_list = []
        f1_list = []
        for batch_x, batch_y in tqdm(train_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.long().to(device)
            output, recover = autoencoder.trainning(batch_x)
            # output.shape = (batch_size, sequence, num_class)
            acc, recall, f1 = computeFPR(y_pred=output, y_target=batch_y)
            # acc, recall, f1 = computeFPR(y_pred=output, y_target=batch_y)
            batch_y = batch_y.squeeze()
            # batch_y = F.softmax(batch_y)
            # output = F.softmax(output)
            loss1 = loss_func_classify(output, batch_y)
            loss2 = loss_func_recover(recover, batch_x)
            loss = loss1 + loss2

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
        autoencoder.eval()
        loss_list = []
        acc_list = []
        recall_list = []
        f1_list = []
        for batch_x, batch_y in valid_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.long().to(device)
            output, recover = autoencoder.trainning(batch_x)
            acc, recall, f1 = computeFPR(y_pred=output, y_target=batch_y)
            # acc, recall, f1 = computeFPR(y_pred=output, y_target=batch_y)
            batch_y = batch_y.squeeze()
            # batch_y = F.softmax(batch_y)
            # output = F.softmax(output)
            loss1 = loss_func_classify(output, batch_y)
            loss2 = loss_func_recover(recover, batch_x)
            loss = loss1 + loss2


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
    autoencoder.train()
    loss_list = []
    acc_list = []
    recall_list = []
    f1_list = []
    y_true = []
    y_pred = []
    for batch_x, batch_y in tqdm(test_loader):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.long().to(device)
        output, recover = autoencoder.trainning(batch_x)
        # output.shape = (batch_size, sequence, num_class)
        acc, recall, f1 = computeFPR(y_pred=output, y_target=batch_y)
        # acc, recall, f1 = computeFPR(y_pred=output, y_target=batch_y)
        batch_y = batch_y.squeeze()
        # batch_y = F.softmax(batch_y)
        # output = F.softmax(output)
        loss1 = loss_func_classify(output, batch_y)
        loss2 = loss_func_recover(recover, batch_x)
        loss = loss1 + loss2

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
    filename = "../modelFile/subtitute_{}_{}_{}.pkt".format(arch, botname, normal)
    save_model(autoencoder, adam, param, hyper, FPR, filename)
