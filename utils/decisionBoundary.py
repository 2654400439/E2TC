from tqdm import tqdm
import torch
from substituteModel.DNN import DNN
from utils.CICIDSData import CICIDS
from torch.utils.data import DataLoader
from TargetModel.TargetLR import TargetLR
from TargetModel.TargetIF import TargetIF
from sklearn.metrics import confusion_matrix
from TargetModel.TargetDT import TargetDT
from TargetModel.TargetSVM import TargetSVM
import numpy as np

def getPredDataDL(dataloader, model, device, filename):
    y_pred = []
    origin_X = []
    for batch_x, batch_y in tqdm(dataloader):
        origin_X += batch_x.detach().cpu().numpy().tolist()
        batch_x = batch_x.to(device)
        output = model(batch_x)
        y_pred += torch.argmax(output, dim=1).detach().cpu().numpy().tolist()
    y_pred = np.array([[x] for x in y_pred])
    origin_X = np.array(origin_X)
    data = np.hstack((origin_X, y_pred))
    print(data.shape)
    np.save(filename, data)

def getPredDataML(dataloader, model, filename):
    X = []
    y = []
    for batch_x, batch_y in dataloader:
        X += batch_x.data.numpy().tolist()
        y += batch_y.data.numpy().tolist()
    X = np.array(X)
    y = np.array(y)
    y_pred = model.clf.predict(X)
    if 'if' in filename:
        y_pred[y_pred == -1] = 0
        y_pred[y_pred == 1] = 1
    print(confusion_matrix(y, y_pred))
    y_pred = np.array([[x] for x in y_pred])
    data = np.hstack((X, y_pred))
    print(data.shape)
    np.save(filename, data)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proxy_model_file = "/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/modelFile/proxy_cicids_dnn_Fuzzing.pkt"
    proxy_model_state = torch.load(proxy_model_file)
    # proxy_model = DNN(proxy_model_state['param'])
    proxy_model = DNN(proxy_model_state['param'])
    proxy_model.load_state_dict(proxy_model_state['model_dict'])
    proxy_model.to(device)
    dataset = CICIDS('Fuzzing')
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)
    output = "../data_Fuzzing_dnn.npy"
    # getPredDataDL(dataloader, proxy_model, device, output)

    target_model_file_svm = "/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/modelFile/target_cicids_svm_Fuzzing.pkt"
    target_model_file_lr = "/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/modelFile/target_cicids_lr_Fuzzing.pkt"
    target_model_file_if = "/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/modelFile/target_cicids_if_Fuzzing.pkt"
    target_model_file_dt = "/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/modelFile/target_cicids_dt_Fuzzing.pkt"
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
    # target_model_svm = TargetSVM(param_svm)
    # target_model_svm.load(target_model_file_svm)
    # svm_output = "/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/data_Fuzzing_svm.npy"
    # getPredDataML(dataloader, target_model_svm, svm_output)
    #
    # target_model_lr = TargetLR(param_lr)
    # target_model_lr.load(target_model_file_lr)
    # lr_output = "/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/data_Fuzzing_lr.npy"
    # getPredDataML(dataloader, target_model_lr, lr_output)

    target_model_if = TargetIF(param_if)
    target_model_if.load(target_model_file_if)
    if_output = "/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/data_Fuzzing_if.npy"
    getPredDataML(dataloader, target_model_if, if_output)

    # target_model_dt = TargetDT()
    # target_model_dt.load(target_model_file_dt)
    # dt_output = "/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/data_Fuzzing_dt.npy"
    # getPredDataML(dataloader, target_model_dt, dt_output)
