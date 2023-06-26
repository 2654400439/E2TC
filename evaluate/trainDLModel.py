import numpy as np
import pandas as pd
from TargetModel.FSNet.dataset import C2Data as MTASequence
from utils.MTACICFlowMeter import MTACICFlowMeter as MTAStatistic
from utils.utils import modelIsExist
from TargetModel.TargetLR import TargetLR
from TargetModel.TargetDT import TargetDT
from TargetModel.TargetSVM import TargetSVM
from TargetModel.TargetRF import TargetRF
from TargetModel.TargetKNN import TargetKNN
from utils.utils import saveSklearnModel, loadSklearnModel, reportMetrics
from utils.CICIDSData import CICIDS
import time
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, DeepFool, CarliniL2Method, SaliencyMapMethod

def getDataloader(name, batch_size):
    cicids = CICIDS(name)
    sample_szie = len(cicids)
    total_size = sample_szie
    test_size = int(total_size * 0.2)
    train_size = int((total_size - test_size) * 0.8)
    valid_size = total_size - test_size - train_size
    train_data, test_data = torch.utils.data.random_split(cicids, [train_size + valid_size, test_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=False)
    return train_loader, test_loader

def trainSklearnModel(arch, name, train_dataloader, test_dataloader, f=None):
    # prepare model
    model = None
    if arch == "svm":
        param = {'kernel': 'rbf'}
        model = TargetSVM(param=param)
    elif arch == "lr":
        param = {
            'C': 0.3
        }
        model = TargetLR(param=param)
    elif arch == "dt":
        model = TargetDT
    elif arch == "rf":
        param = {
            'criterion': 'gini',
            'n_estimators': 10,
            'max_depth': 10,
        }
        model = TargetRF(param=param)
    elif arch == "knn":
        param = {
            'n_neighbors': 10
        }
        model = TargetKNN(param=param)
    else:
        raise Exception("架构错误，仅支持svm, lr, dt, rf, knn")

    # train model
    model.train(train_dataloader)
    y_true, y_pred = model.eval(test_dataloader)
    print("{} model metrics(training phase) for {}: ".format(arch, name))
    if f:
        print("{} model metrics(training phase) for {}: ".format(arch, name), file=f)
    reportMetrics(y_true, y_pred, f)
    path = "/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/modelFile/pickle/{}_{}.pickle".format(arch, name)
    saveSklearnModel(model, path)
    return model

def attackSklearnModel(model, name, min_value, max_value, eps, dataloader, method="fgsm", f=None):
    classifier = SklearnClassifier(model=model.clf, clip_values=(min_value, max_value))
    if method == "fgsm":
        attack = FastGradientMethod(estimator=classifier, eps=eps, batch_size=128)
    elif method == "bim":
        attack = BasicIterativeMethod(estimator=classifier, eps=eps, max_iter=1, verbose=False, batch_size=128)
    elif method == "cw":
        attack = CarliniL2Method(classifier=classifier, verbose=False, batch_size=128)
    elif method == "deepfool":
        attack = DeepFool(classifier=classifier, max_iter=1, verbose=False, batch_size=128)
    elif method == "jsma":
        attack = SaliencyMapMethod(classifier=classifier, verbose=False, batch_size=128)
    else:
        raise Exception("method 不支持，仅支持fgsm, bim, jsma, cw, deepfool")
    X = []
    y = []
    for batch_x, batch_y in dataloader:
        X += batch_x.data.numpy().tolist()
        y += batch_y.data.numpy().tolist()
    X = np.array(X)
    y = np.array(y)
    y_pred = classifier.predict(X)
    y_pred = np.argmax(y_pred, axis=1)
    print("{} model metrics(before {} attack) for {}:".format(arch, method, name))
    if f:
        print("{} model metrics(beofre {} attack) for {}:".format(arch, method, name), file=f)
        f.flush()
    reportMetrics(y, y_pred, f)
    X_adv = attack.generate(X)
    y_pred_adv = classifier.predict(X_adv)
    y_pred_adv = np.argmax(y_pred_adv, axis=1)
    print("{} model metrics(after {} attack) for {}:".format(arch, method, name))
    if f:
        print("{} model metrics(after {} attack):".format(arch, method), file=f)
        f.flush()
    reportMetrics(y, y_pred_adv, f)








if __name__ == '__main__':
    malwares = [
        "Botnet",
        "Fuzzing",
        "PortScan",
        "BruteForce",
        "DDoS"
    ]
    archs = [
        "svm",
        "lr",
        "dt",
        "rf",
        "knn"
    ]
    methods = [
        "fgsm",
        "bim",
        "jsma",
        # "cw",
        "deepfool"
    ]
    f = open("/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/evaluate4.txt", "w")
    for arch in archs:
        for name in malwares:
            for method in methods:
                train_dataloader, test_dataloader = getDataloader(name, batch_size=128)
                if modelIsExist(arch, name):
                    path = "/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/modelFile/pickle/{}_{}.pickle".format(arch, name)
                    model = loadSklearnModel(path)
                else:
                    model = trainSklearnModel(arch, name, train_dataloader, test_dataloader, f)
                try:
                    start = time.time()
                    attackSklearnModel(model, name, 0, 1, 0.1, test_dataloader, method, f)
                    sampleNum = len(test_dataloader)
                    end = time.time()
                    print("sample num: {}".format(sampleNum))
                    print("time consuming: {:.3f} s".format(end - start))
                    print("generate rate: {:.3f} k/s".format(sampleNum / (end - start)))
                    print("generate rate: {:.3f} k/s".format(sampleNum / (end - start)), file=f)
                except Exception as e:
                    print(e)
                    print(e, file=f)
                    pass
    f.close()