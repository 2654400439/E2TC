"""
Date: 2022-04-12
Author: sunhanwu@iie.ac.cn
Desc: attack the target model use adversarial examples
"""
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from attack.adversarialDataset import AdversarialC2Data
from TargetModel.TargetSVM import TargetSVM
from TargetModel.TargetLR import TargetLR
from TargetModel.TargetDT import TargetDT
from TargetModel.TargetRF import TargetRF
from TargetModel.TargetKNN import TargetKNN
from torch.utils.data import DataLoader

def attackDLModel(model, dataloader, device, batch_size=128):
    """

    :param model:
    :param data:
    :return:
    """
    model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    for batch_x, batch_y  in tqdm(dataloader):
        batch_x = batch_x.to(device).float()
        batch_y = batch_y.to(device)
        output = model(batch_x)
        y_true += batch_y.detach().cpu().data.numpy().reshape(-1).tolist()
        y_pred += torch.argmax(output, dim=1).detach().cpu().data.numpy().tolist()
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    con_matrix = confusion_matrix(y_true, y_pred)
    print("acc: {:.2%}, recall: {:.2%}, f1: {:.2%}".format(acc, recall, f1))
    print(con_matrix)

def attackMLModel(model, dataloader):
    """

    :param model:
    :param dataloader:
    :return:
    """
    y_pred, y_true = model.eval(dataloader)
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    con_matrix = confusion_matrix(y_true, y_pred)
    print("acc: {:.2%}, recall: {:.2%}, f1: {:.2%}".format(acc, recall, f1))
    print("con_metric:\n {}".format(con_matrix))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targetArch = "knn"
    surrogateArch = "rnn"
    botname = "Gozi"
    normal = "CTUNone"
    target_class=0
    targetModelFile = "../modelFile/target_{}_{}_{}.pkt".format(targetArch, botname, normal)
    adverDataFile = "../adversarialData/advdata_{}_{}_{}_5_160.0_20.npy".format(surrogateArch, botname, normal)
    advdata = AdversarialC2Data(adverDataFile, target_class)
    advdataloader = DataLoader(advdata, batch_size=128, shuffle=True, drop_last=False)
    param = {
        'n_neighbors': 6,
    }
    targetModel = TargetKNN(param)
    targetModel.load(targetModelFile)
    attackMLModel(targetModel, advdataloader)




