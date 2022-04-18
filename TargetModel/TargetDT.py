"""
Date: 2022-04-13
Author: sunhanwu@iie.ac.cn
Desc: target model: Decision Tree
"""
from sklearn import tree
from TargetModel.FSNet.dataset import C2Data
import torch
from sklearn.metrics import confusion_matrix
import joblib
from torch.utils.data import DataLoader
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class TargetDT():
    """

    """
    def __init__(self, param=None):
        # 正则化
        self.clf = tree.DecisionTreeClassifier()

    def train(self, dataloader):
        X = []
        y = []
        for batch_x, batch_y in dataloader:
            X += batch_x.data.numpy().tolist()
            y += batch_y.data.numpy().tolist()
        X = np.array(X)
        y = np.array(y)
        print("X.shape:{}".format(X.size))
        print("y.shape:{}".format(y.size))
        self.clf.fit(X, y)
        print("training score:{}".format(self.clf.score(X, y)))

    def eval(self, dataloader):
        X = []
        y = []
        for batch_x, batch_y in dataloader:
            X += batch_x.data.numpy().tolist()
            y += batch_y.data.numpy().tolist()
        X = np.array(X)
        y = np.array(y)
        print("X.shape:{}".format(X.size))
        print("y.shape:{}".format(y.size))
        y_pred = self.clf.predict(X)
        return y_pred, y

    def save(self, filename):
        joblib.dump(self.clf, filename)

    def load(self, filename):
        self.clf = joblib.load(filename)

if __name__ == '__main__':

    arch = "dt"
    sample_szie = 580
    botname = "Gozi"
    normal = "CTUNone"

    batch_size = 128

    total_size = sample_szie * 2
    test_size = int(total_size * 0.2)
    train_size = int((total_size - test_size) * 0.8)
    valid_size = total_size - test_size - train_size
    print("train data: {}".format(train_size))
    print("valid data: {}".format(valid_size))
    print("test data: {}".format(test_size))

    c2data = C2Data(botname, number=sample_szie, sequenceLen=30)
    train_data, test_data = torch.utils.data.random_split(c2data, [train_size + valid_size, test_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=False)

    svm = TargetDT()
    svm.train(train_loader)
    y_true, y_pred = svm.eval(test_loader)
    print("confusion_metrix: \n{}".format(confusion_matrix(y_true, y_pred)))
    filename = "../modelFile/target_{}_{}_{}.pkt".format(arch, botname, normal)
    svm.save(filename)


