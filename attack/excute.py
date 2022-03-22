"""
Date: 2022-03-08
Author: sunhanwu@iie.ac.cn
Desc: execute the adversarial attack
"""

from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from TargetModel.FSNet.FSNet import FSNet
from attack.adversarialDataset import AdversarialC2Data
from torch.utils.data import DataLoader

def executeAttack(dataloader, model, device, file):
    y_true = []
    y_pred = []
    for batch_x, batch_y in tqdm(dataloader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        y_true += batch_y.detach().cpu().numpy().reshape(-1).tolist()
        output = model(batch_x)
        y_pred += torch.argmax(output, dim=1).detach().cpu().numpy().tolist()
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    con_matrix = confusion_matrix(y_true, y_pred)
    print("acc: {:.2%}, recall: {:.2%}, f1: {:.2%}".format(acc, recall, f1))
    print(con_matrix)
    print("acc: {:.2%}, recall: {:.2%}, f1: {:.2%}".format(acc, recall, f1), file=file)
    print(con_matrix, file=file)
    return acc, recall, f1, con_matrix




def attackFsnet(theta, gamma, file, advdataFile, modelfile):
    state = torch.load(modelfile)
    fsnet = FSNet(state['param'])
    fsnet.load_state_dict(state['model_dict'])

    # adversarial data
    batch_size = 128
    adversarialC2Data = AdversarialC2Data(advdataFile, keep_target=True)
    adversarialDataloader = DataLoader(adversarialC2Data, batch_size, shuffle=True, drop_last=False)
    print("Attack FSNet")
    print(fsnet)

    fsnet.eval()
    y_true = []
    y_pred = []
    for batch_x, batch_y in tqdm(adversarialDataloader):
        # batch_x.shape: (batch_size, sequenceLen)
        # batch_y.shape: (batch_size, 1)
        # reconstruction
        z_e = fsnet.encode(batch_x)
        # z_e.shape: (batch_size, num_layers * num_direction, hidden_size)
        z_d, D = fsnet.decode(z_e)
        # z_d.shape: (batch_size, num_layers * num_direction, hidden_size)
        # D.shape: (batch_size, sequence_len, hidden_size * num_direction)
        z_reconstruction = fsnet.reconstruction(D)
        # z_reconstruction.shape=(batch_size, sequence_len, vocab_size)

        # classification layer
        z_dense = fsnet.dense(z_e, z_d)
        # z_dense.shape=(batch_size, num_class)

        # compute reconstruction loss
        # z_reconstruction.shape = (-1, vocab_size)
        # batch_x.shape=(-1)

        # compute classification loss
        # batch_y.shape=(batch, )

        # compute FPR
        y_true += batch_y.data.numpy().reshape(-1).tolist()
        y_pred += torch.argmax(z_dense, dim=1).data.numpy().tolist()
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    con_matrix = confusion_matrix(y_true, y_pred)
    print("acc: {:.2%}, recall: {:.2%}, f1: {:.2%}".format(acc, recall, f1))
    print(con_matrix)
    print("acc: {:.2%}, recall: {:.2%}, f1: {:.2%}".format(acc, recall, f1), file=file)
    print(con_matrix, file=file)


