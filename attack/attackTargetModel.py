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
from TargetModel.TargetLSTM import TargetLSTM
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
    targetArch = "lstm"
    surrogateArch = "lstm"
    botname = "Gozi"
    theta = 0.01
    gamma = 0.1
    times = 20
    target_class=0
    targetModelFile = "../modelFile/target_mta_cicflowmeter_{}_{}.pkt".format(targetArch, botname)
    adverDataFile = "../adversarialData/advdata_mta_cicflowmeter_{}_{}_{}_{}_{}.npy".format(surrogateArch, botname, theta, gamma, times)
    advdata = AdversarialC2Data(adverDataFile, target_class)
    advdataloader = DataLoader(advdata, batch_size=128, shuffle=True, drop_last=False)

    print("Load adv model: {}".format(targetModelFile))
    adv_state = torch.load(targetModelFile)
    print("model param: {}".format(adv_state['param']))
    targetModel = TargetLSTM(adv_state['param'])
    targetModel.load_state_dict(adv_state['model_dict'])
    targetModel.to(device)
    attackDLModel(targetModel, advdataloader, device, batch_size=128)




