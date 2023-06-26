"""
Date: 2022-04-13
Author: sunhanwu@iie.ac.cn
Desc: target model: Isolate Forest
"""
from sklearn.ensemble import IsolationForest
import torch
from sklearn.metrics import confusion_matrix
import joblib
from torch.utils.data import DataLoader
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class TargetIF():
    """
    """
    def __init__(self, param):
        # 正则化
        self.clf = IsolationForest(
            n_estimators=param['n_estimators'],
            contamination=param['outliers_fraction1'],
            n_jobs=-1
        )

    def train(self, dataloader):
        X = []
        y = []
        for batch_x, batch_y in dataloader:
            X += batch_x.data.numpy().tolist()
            y += batch_y.data.numpy().tolist()
        X = np.array(X)
        y = np.array(y)
        # print("X.shape:{}".format(X.size))
        # print("y.shape:{}".format(y.size))
        self.clf.fit(X, y)
        # print("training score:{}".format(self.clf.score(X, y)))

    def eval(self, dataloader):
        X = []
        y = []
        for batch_x, batch_y in dataloader:
            X += batch_x.data.numpy().tolist()
            y += batch_y.data.numpy().tolist()
        X = np.array(X)
        y = np.array(y)
        # print("X.shape:{}".format(X.size))
        # print("y.shape:{}".format(y.size))
        y_pred = self.clf.predict(X)
        y_pred = y_pred.reshape(-1,1)
        y_pred[y_pred == -1] = 0
        y_pred[y_pred == 1] = 1
        return y_pred, y

    def save(self, filename):
        joblib.dump(self.clf, filename)

    def load(self, filename):
        self.clf = joblib.load(filename)