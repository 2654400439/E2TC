"""
Date: 2022-04-18
Author: sunhanwu@iie.ac.cn
Desc: Test the trade-off
"""

import torch
from attack.adversarialDataset import AdversarialC2Data
from TargetModel.TargetSVM import TargetSVM
from TargetModel.TargetLR import TargetLR
from TargetModel.TargetDT import TargetDT
from TargetModel.TargetRF import TargetRF
from TargetModel.TargetKNN import TargetKNN
from TargetModel.TargetDNN import TargetDNN
from TargetModel.TargetIF import TargetIF
from TargetModel.TargetLSTM import TargetLSTM
from TargetModel.FSNet.FSNet import FSNet
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def attackGammaML(sur_arch, target_arch, botname, normal, gamma: float):
    # sur_arch = "dnn"
    # target_arch = "svm"
    # botname = "Tofsee"
    # normal = "CTUNone"
    batch_size = 128
    target_class = 0
    advdata_file = "../adversarialData/advdata_{}_{}_{}_5_{}.0_20.npy".format(sur_arch, botname, normal, int(gamma))
    if target_arch == 'if':
        target_model_file = "../modelFile/target_mta_length_{}_{}.pkt".format(target_arch, botname)
    else:
        target_model_file = "../modelFile/target_{}_{}_{}.pkt".format(target_arch, botname, normal)
    target_model = None
    param = {
        'kernel': 'rbf',
        'C': 0.3,
        'criterion': 'gini',
        'n_estimators': 10,
        'max_depth': 10,
        'n_neighbors': 10,
        'outliers_fraction1': 0.2,
    }
    target_model = globals()["Target{}".format(target_arch.upper())](param)
    target_model.load(target_model_file)
    adv_data = AdversarialC2Data(advdata_file, target_class=target_class, keep_target=True)
    adv_loader = DataLoader(adv_data, batch_size=batch_size, shuffle=True)
    y_true, y_pred = target_model.eval(adv_loader)
    con_metrix = confusion_matrix(y_true, y_pred)
    print("confusion_metrix: \n{}".format(con_metrix))
    EDR = con_metrix[1][0] / (con_metrix[1][0] + con_metrix[1][1])
    return EDR

def attackGammaDL(sur_arch, target_arch, botname, normal, gamma: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    target_class = 0
    advdata_file = "../adversarialData/advdata_{}_{}_{}_5_{}.0_20.npy".format(sur_arch, botname, normal, int(gamma))
    target_model_file = "../modelFile/target_{}_{}_{}.pkt".format(target_arch, botname, normal)
    target_model = None
    state = torch.load(target_model_file)
    if target_arch == 'fsnet':
        target_model = FSNet(state['param'])
    else:
        target_model = globals()["Target{}".format(target_arch.upper())](state['param'])
    target_model.load_state_dict(state['model_dict'])
    target_model.to(device)
    adv_data = AdversarialC2Data(advdata_file, target_class=target_class, keep_target=True)
    adv_loader = DataLoader(adv_data, batch_size=batch_size, shuffle=True)
    target_model.eval()

    y_true = []
    y_pred = []
    for batch_x, batch_y in adv_loader:
        batch_x = batch_x.to(device, dtype=torch.float)
        batch_y = batch_y.to(device)
        output = target_model(batch_x)
        # output.shape = (batch_size, sequence, num_class)
        batch_y = batch_y.squeeze()
        # batch_y = F.softmax(batch_y)
        # output = F.softmax(output)
        y_true += batch_y.detach().cpu().numpy().tolist()
        y_pred += torch.argmax(output, dim=1).detach().cpu().numpy().tolist()
    con_metrix = confusion_matrix(y_true, y_pred)
    print("confusion_metrix: \n{}".format(con_metrix))
    EDR = con_metrix[1][0] / (con_metrix[1][0] + con_metrix[1][1])
    print("EDR: {}".format(EDR))
    return EDR

if __name__ == '__main__':
    sur_arch = "rnn"
    target_archs = [
        'if'
    ]
    # target_arch = "svm"
    botname = "Dridex"
    normal = "CTUNone"
    for target_arch in target_archs:
        EDR_list = []
        for gamma in range(0, 800, 5):
            edr = attackGammaML(sur_arch=sur_arch, target_arch=target_arch, botname=botname, normal=normal, gamma=float(gamma))
            EDR_list.append((gamma, edr))
        save_file = "../{}_{}_{}_edr_gamma_list.npy".format(sur_arch, target_arch, botname)
        np.save(save_file, EDR_list)
