"""
Date: 2022-07-20
Author: sunhanwu@iie.ac.cn
Desc: LR model for cicids 2017
"""
import os
import warnings
import torch
from torch.utils.data import DataLoader
from TargetModel.TargetLR import TargetLR
from TargetModel.TargetDT import TargetDT
from TargetModel.TargetSVM import TargetSVM
from TargetModel.TargetIF import TargetIF
from TargetModel.TargetMLP import TargetMLP
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from utils.CICIDSData import CICIDS
from utils.MTACICFlowMeter import MTACICFlowMeter
from TargetModel.FSNet.dataset import C2Data
warnings.filterwarnings("ignore")

def training(model, name, arch, batch_size = 128):
    cicids = CICIDS(name)
    sample_szie = len(cicids)
    total_size = sample_szie
    test_size = int(total_size * 0.2)
    train_size = int((total_size - test_size) * 0.8)
    valid_size = total_size - test_size - train_size
    print("train data: {}".format(train_size))
    print("valid data: {}".format(valid_size))
    print("test data: {}".format(test_size))
    train_data, test_data = torch.utils.data.random_split(cicids, [train_size + valid_size, test_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=False)
    model.train(train_loader)
    y_true, y_pred = model.eval(test_loader)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    print("{} pre: {:.2%}, recall: {:.2%}, f1: {:.2%}".format(name, precision, recall, f1))
    filename = "/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/modelFile/target_cicids_{}_{}.pkt".format(arch, name)
    model.save(filename)
    print("confusion_metrix: \n{}".format(confusion_matrix(y_true, y_pred)))


def trainingMTACICFlowMeter(model, name, number, arch, batch_size = 128):
    cicids = MTACICFlowMeter(name, number)
    sample_szie = len(cicids)
    total_size = sample_szie
    test_size = int(total_size * 0.2)
    train_size = int((total_size - test_size) * 0.8)
    valid_size = total_size - test_size - train_size
    print("train data: {}".format(train_size))
    print("valid data: {}".format(valid_size))
    print("test data: {}".format(test_size))
    train_data, test_data = torch.utils.data.random_split(cicids, [train_size + valid_size, test_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=False)
    model.train(train_loader)
    y_true, y_pred = model.eval(test_loader)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    print("{} pre: {:.2%}, recall: {:.2%}, f1: {:.2%}".format(name, precision, recall, f1))
    filename = "/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/modelFile/target_mta_cicflowmeter_{}_{}.pkt".format(arch, name)
    model.save(filename)

def trainingMta(model, name, arch, batch_size = 128, sample_size=200, feature_type='length'):
    c2data = C2Data(name, number=sample_size, sequenceLen=30, feature_type=feature_type)
    sample_szie = len(c2data)
    total_size = sample_szie
    test_size = int(total_size * 0.2)
    train_size = int((total_size - test_size) * 0.8)
    valid_size = total_size - test_size - train_size
    # print("train data: {}".format(train_size))
    # print("valid data: {}".format(valid_size))
    # print("test data: {}".format(test_size))
    train_data, test_data = torch.utils.data.random_split(c2data, [train_size + valid_size, test_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=False)
    model.train(train_loader)
    y_true, y_pred = model.eval(test_loader)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    print("{} pre: {:.2%}, recall: {:.2%}, f1: {:.2%}".format(name, precision, recall, f1))
    filename = "/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/modelFile/target_mta_{}_{}_{}.pkt".format(feature_type, arch, name)
    model.save(filename)
    # print("confusion_metrix: \n{}".format(confusion_matrix(y_true, y_pred)))

if __name__ == '__main__':
    param_lr = {
        'C': 0.3
    }
    param_svm = {
        'kernel': 'rbf',

    }
    param_if= {
        'outliers_fraction1': 0.2,
        "n_estimators": 200
    }
    param_mlp = {
        'activate': 'relu',
        'hidden_size': (50, 25, 13),
        'learning_rate_init': 0.001,
        'max_iter': 200,
        'momentum': 0.9,
        'solver': 'adam',
        'alpha': 0.01,
        'batch_size': 128
    }
    malwares = [
        # "Botnet",
        "Fuzzing",
        # "PortScan",
        # "BruteForce",
        # "DDoS"
    ]
    Botnets = [
        "Tofsee",
        "Dridex",
        "Quakbot",
        "TrickBot",
        "Gozi"
    ]
    lr = TargetLR(param_lr)
    dt = TargetDT()
    svm = TargetSVM(param_svm)
    IF = TargetIF(param_if)
    mlp = TargetMLP(param_mlp)
    models = [lr, dt, svm, IF, mlp]
    models_s = ['lr', 'dt', 'svm', 'if', 'mlp']
    numbers = [2580, 2580, 1600, 690, 1250]
    for i in range(3, 4):
        print("{}".format(models_s[i]))
        for name in malwares:
            training(models[i], name, arch=models_s[i])