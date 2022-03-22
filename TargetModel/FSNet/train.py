"""
Date: 2022-03-04
Author: sunhanwu@iie.ac.cn
Desc: train the fsnet model, got the baseline
"""


import torch
from TargetModel.FSNet.dataset import C2Data
from torch.utils.data import DataLoader
from TargetModel.FSNet.FSNet import FSNet
from TargetModel.FSNet.utils import save_model
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
from warnings import filterwarnings
filterwarnings("ignore")



def trainModel(epoch_index, model, dataloader, criterrion, optimizer, device):
    """
    train model for an epoch
    :param model:
    :param dataloader:
    :param criterrion:
    :param optimizer:
    :return:
    """
    model.train()
    loss_list = []
    acc_list = []
    recall_list = []
    f1_list = []
    for batch_x, batch_y in tqdm(dataloader):
        # batch_x.shape: (batch_size, sequenceLen)
        batch_x = batch_x.to(device)
        # batch_y.shape: (batch_size, 1)
        batch_y = batch_y.to(device)

        # reconstruction
        z_e = model.encode(batch_x)
        # z_e.shape: (batch_size, num_layers * num_direction, hidden_size)
        z_d, D = model.decode(z_e)
        # z_d.shape: (batch_size, num_layers * num_direction, hidden_size)
        # D.shape: (batch_size, sequence_len, hidden_size * num_direction)
        z_reconstruction = model.reconstruction(D)
        # z_reconstruction.shape=(batch_size, sequence_len, vocab_size)

        # classification layer
        z_dense = model.dense(z_e, z_d)
        # z_dense.shape=(batch_size, num_class)

        # compute reconstruction loss
        z_reconstruction = torch.reshape(z_reconstruction, [-1, z_reconstruction.shape[-1]])
        # z_reconstruction.shape = (-1, vocab_size)
        batch_x = torch.reshape(batch_x, [-1])
        # batch_x.shape=(-1)
        reconstruction_loss = criterrion(z_reconstruction, batch_x)

        # compute classification loss
        batch_y = batch_y.squeeze()
        # batch_y.shape=(batch, )
        classification_loss = criterrion(z_dense, batch_y)
        loss = reconstruction_loss + classification_loss

        # compute FPR
        acc, recall, f1 = computeFPR(z_dense, batch_y)
        acc_list.append(acc)
        recall_list.append(recall)
        f1_list.append(f1)
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("[Training {:03d}] acc: {:.2%}, recall: {:.2%}, f1: {:.2%}, loss: {:.2f}".format(epoch_index + 1,
                                                                                     np.mean(acc_list),
                                                                                     np.mean(recall_list),
                                                                                     np.mean(f1_list),
                                                                                     np.mean(loss_list)))



def validModel(epoch_index, model, dataloader, criterrion, device, metrix_flag=False):
    """
    test model for an epoch
    :param model:
    :param dataloader:
    :param criterrion:
    :param optimizer:
    :return:
    """
    model.eval()
    loss_list = []
    acc_list = []
    recall_list = []
    f1_list = []
    y_true = []
    y_pred = []
    for batch_x, batch_y in dataloader:
        # batch_x.shape: (batch_size, sequenceLen)
        batch_x = batch_x.to(device)
        # batch_y.shape: (batch_size, 1)
        batch_y = batch_y.to(device)

        # reconstruction
        z_e = model.encode(batch_x)
        # z_e.shape: (batch_size, num_layers * num_direction, hidden_size)
        z_d, D = model.decode(z_e)
        # z_d.shape: (batch_size, num_layers * num_direction, hidden_size)
        # D.shape: (batch_size, sequence_len, hidden_size * num_direction)
        z_reconstruction = model.reconstruction(D)
        # z_reconstruction.shape=(batch_size, sequence_len, vocab_size)

        # classification layer
        z_dense = model.dense(z_e, z_d)
        # z_dense.shape=(batch_size, num_class)

        # compute reconstruction loss
        z_reconstruction = torch.reshape(z_reconstruction, [-1, z_reconstruction.shape[-1]])
        # z_reconstruction.shape = (-1, vocab_size)
        batch_x = torch.reshape(batch_x, [-1])
        # batch_x.shape=(-1)
        reconstruction_loss = criterrion(z_reconstruction, batch_x)

        # compute classification loss
        batch_y = batch_y.squeeze()
        # batch_y.shape=(batch, )
        classification_loss = criterrion(z_dense, batch_y)
        loss = reconstruction_loss + classification_loss

        # compute FPR
        acc, recall, f1 = computeFPR(z_dense, batch_y)
        acc_list.append(acc)
        recall_list.append(recall)
        f1_list.append(f1)
        loss_list.append(loss.item())
        y_true += batch_y.detach().cpu().numpy().tolist()
        y_pred += torch.argmax(z_dense, dim=1).detach().cpu().numpy().tolist()

    print("[{:03d}] acc: {:.2%}, recall: {:.2%}, f1: {:.2%}, loss: {:.2f}".format(epoch_index + 1,
                                                                                           np.mean(acc_list),
                                                                                           np.mean(recall_list),
                                                                                           np.mean(f1_list),
                                                                                           np.mean(loss_list)))
    if metrix_flag:
        print(confusion_matrix(y_true, y_pred))
    return np.mean(acc_list), np.mean(recall_list), np.mean(f1_list), confusion_matrix(y_true, y_pred)


def computeFPR(y_pred, y_target):
    """
    compute acc, recall, f1
    :param y_pred: [sequence_len, class_num]
    :param y_target: [sequence_len]
    :return:
    """
    y_target = y_target.cpu().numpy()
    max_value, pred_class = torch.max(y_pred, dim=1)
    pred_class = pred_class.cpu().numpy()
    acc = accuracy_score(y_target, pred_class)
    recall = recall_score(y_target, pred_class, average='macro')
    f1 = f1_score(y_target, pred_class, average='macro')
    return acc, recall, f1

def computeFPR_(y_pred, y_target):
    """
    compute acc, recall, f1
    :param y_pred: [sequence_len, class_num]
    :param y_target: [sequence_len]
    :return:
    """
    y_target = np.argmax(y_target.cpu().numpy(), 1)
    max_value, pred_class = torch.max(y_pred, dim=1)
    pred_class = pred_class.cpu().numpy()
    acc = accuracy_score(y_target, pred_class)
    recall = recall_score(y_target, pred_class, average='macro')
    f1 = f1_score(y_target, pred_class, average='macro')
    return acc, recall, f1






if __name__ == '__main__':
    # hyper param
    epoch_size = 10
    lr = 1e-3
    batch_size= 128
    botname = "Dridex"
    normal = "CTUNone"
    arch = "fsnet"

    # model param
    param = {
        "sequence_len": 30,
        "vocab_size": 1600,
        "emb_dim": 128,
        "hidden_size": 64,
        "dec_gru": 64,
        "num_layers": 2,
        "num_direction": 2,
        "num_class":  2,
    }

    # use GPU if it is available, oterwise use cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pre the dataloader
    c2data = C2Data(botname, number=8000, sequenceLen=30)
    train_valid_data, test_data = torch.utils.data.random_split(c2data, [12800, 3200])
    train_data, valid_data = torch.utils.data.random_split(train_valid_data, [10000, 2800])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=False)

    # model

    fsnet = FSNet(param)
    fsnet.to(device)

    # loss func
    crossEntropy = torch.nn.CrossEntropyLoss()
    adam = torch.optim.Adam(fsnet.parameters(), lr=lr)

    # train and valid
    for epoch_index in range(epoch_size):
        trainModel(epoch_index, fsnet, train_loader, crossEntropy, adam, device)
        validModel(epoch_index, fsnet, valid_loader, crossEntropy, device)
    print("Testing")
    acc, recall, f1, metrix = validModel(epoch_index, fsnet, test_loader, crossEntropy, device, metrix_flag=True)
    FPR = {
        'acc': acc,
        'recall': recall,
        'f1': f1,
        'metrix': metrix
    }
    hyper = {
        'epoch_size': epoch_size,
        'lr': lr,
        'batch_size': batch_size
    }
    filename = "../../modelFile/target_{}_{}_{}.pkt".format(arch, botname, normal)
    save_model(fsnet, adam, param, hyper, FPR,  filename)

