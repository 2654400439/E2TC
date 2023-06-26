import numpy as np
import pandas as pd
import json

from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestClassifier

def CalcGiniIndex(X, y):
    forest = RandomForestClassifier(n_estimators=10000, n_jobs=-1)
    forest.fit(X, y)
    return forest.feature_importances_.tolist()

def loadCICIDS(name):
    data = np.load("/home/sunhanwu/datasets/cicids2017/npy/{}.npy".format(name), allow_pickle=True)
    X = data[:, :-1].astype(np.float32)
    y = data[:, -1].astype(int)
    X[np.where(np.isnan(X))] = 0
    X[np.where(X >= np.finfo(np.float32).max)] = np.finfo(np.float32).max - 1
    return X, y

def loadMTA(name):
    data = np.load("/home/sunhanwu/datasets/MTA/cicflownpy/{}.npy".format(name), allow_pickle=True)
    X = data[:, :-1].astype(np.float32)
    y = data[:, -1].astype(int)
    X[np.where(np.isnan(X))] = 0
    X[np.where(X >= np.finfo(np.float32).max)] = np.finfo(np.float32).max - 1
    return X, y

def CalcCICIDS2017():
    malware = ['Botnet', 'BruteForce', 'DDoS', 'Fuzzing', 'PortScan']
    CICIDS2017_Gini = {}
    for cls in malware:
        print("calc gini index of {}".format(cls))
        X, y = loadCICIDS(cls)
        importance = CalcGiniIndex(X, y)
        CICIDS2017_Gini[cls] = importance
        print("calc gini index of {} done.".format(cls))
    with open('/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/result/CICIDS2017_GI.json', 'w') as f:
        json.dump(CICIDS2017_Gini, f)

def CalcMTA():
    malware = ['Dridex', 'Gozi', 'Quakbot', 'Tofsee', 'TrickBot']
    MTA_Gini = {}
    for cls in malware:
        print("calc gini index of {}".format(cls))
        X, y = loadMTA(cls)
        importance = CalcGiniIndex(X, y)
        MTA_Gini[cls] = importance
        print("calc gini index of {} done.".format(cls))
    with open('/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/result/MTA_GI.json', 'w') as f:
        json.dump(MTA_Gini, f)

if __name__ == '__main__':
    CalcMTA()