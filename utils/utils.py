import pickle
import os
from sklearn.metrics import precision_score, recall_score, f1_score

def loadSklearnModel(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def saveSklearnModel(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def modelIsExist(arch, name):
    path = "/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/modelFile/pickle/{}_{}.pickle".format(arch, name)
    return os.path.exists(path)

def reportMetrics(y_true, y_pred, f=None):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("precision: {:.2%}\trecall: {:.2%}\tf1: {:.2%}".format(precision, recall, f1))
    if f:
        print("precision: {:.2%}\trecall: {:.2%}\tf1: {:.2%}".format(precision, recall, f1), file=f)
        f.flush()
