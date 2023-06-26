from attack.adversarialDataset import AdversarialC2Data
from torch.utils.data import DataLoader
from TargetModel.TargetLR import TargetLR
from TargetModel.TargetMLP import TargetMLP
from TargetModel.TargetSVM import TargetSVM
from TargetModel.TargetDT import TargetDT
from TargetModel.TargetIF import TargetIF
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

def attack(arch, name, theta, gamma, modify_times, target_class, model, filter_name):
    advmodel_filename = "/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/modelFile/target_cicids_{}_{}.pkt".format(arch, name)
    advdata_filename = "/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/adversarialData/advdata_cicids_{}_{}_{}_{}_{}_{}.npy".format("dnn",
                        name, theta, gamma, modify_times, filter_name)
    adv_data = AdversarialC2Data(advdata_filename, target_class=target_class, keep_target=True, norm=False)
    model.load(advmodel_filename)
    dataloader = DataLoader(adv_data, batch_size=128, shuffle=True)
    y_pred, y_true = model.eval(dataloader)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    con_matrix = confusion_matrix(y_true, y_pred)
    print("{} pre: {:.2%}, recall: {:.2%}, f1: {:.2%}".format(name, precision, recall, f1))
    print(con_matrix)

def attackMTA(arch, name, theta, gamma, modify_times, target_class, model, feature_type):
    advmodel_filename = "/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/modelFile/target_mta_{}_{}_{}.pkt".format(feature_type, arch, name)
    print("target model: {}".format(advmodel_filename))
    advdata_filename = "/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/adversarialData/advdata_mta_{}_{}_{}_{}_{}_{}.npy".format(
        feature_type, "dnn", name, theta, gamma, modify_times)
    adv_data = AdversarialC2Data(advdata_filename, target_class=target_class, keep_target=True, norm=False)
    model.load(advmodel_filename)
    dataloader = DataLoader(adv_data, batch_size=128, shuffle=True)
    y_pred, y_true = model.eval(dataloader)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    con_matrix = confusion_matrix(y_true, y_pred)
    print("{} pre: {:.2%}, recall: {:.2%}, f1: {:.2%}".format(name, precision, recall, f1))
    print(con_matrix)

def attackMTACICFLowMeter(arch, name, theta, gamma, modify_times, target_class, model, proxy_arch):
    advmodel_filename = "/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/modelFile/target_mta_cicflowmeter_{}_{}.pkt".format(arch, name)
    print("target model: {}".format(advmodel_filename))
    advdata_filename = "/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/adversarialData/advdata_mta_cicflowmeter_{}_{}_{}_{}_{}.npy".format(
        proxy_arch, name, theta, gamma, modify_times)
    adv_data = AdversarialC2Data(advdata_filename, target_class=target_class, keep_target=True, norm=False)
    model.load(advmodel_filename)
    dataloader = DataLoader(adv_data, batch_size=128, shuffle=True)
    y_pred, y_true = model.eval(dataloader)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    con_matrix = confusion_matrix(y_true, y_pred)
    print("{} pre: {:.2%}, recall: {:.2%}, f1: {:.2%}".format(name, precision, recall, f1))
    print(con_matrix)

def attackCICIDSModels():
    theta = 1
    gamma = 0.1
    modify_times = 20
    target_class = 0
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
        "Botnet",
        "Fuzzing",
        "PortScan",
        "BruteForce",
        "DDoS"
    ]
    name = "DDoS"
    arch = "if"
    filer_name = "both"
    lr = TargetLR(param_lr)
    dt = TargetDT()
    svm = TargetSVM(param_svm)
    IF = TargetIF(param_if)
    mlp = TargetMLP(param_mlp)
    attack(arch,name,theta,gamma,modify_times,target_class,IF, filter_name=filer_name)

def attackMTAModels():
    theta = 10.0
    gamma = 0.9
    modify_times = 20
    target_class = 0
    param_lr = {
        'c': 0.3
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
        "Botnet",
        "Fuzzing",
        "PortScan",
        "BruteForce",
        "DDoS"
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
    model = [lr, dt, svm, IF, mlp]
    model_s = ['lr', 'dt', 'svm', 'if', 'mlp']
    proxy_arch = "dnn"
    for i in range(5):
        for botname in Botnets:
            print("{}:{}".format(model_s[i], botname))
            attackMTACICFLowMeter(model_s[i],botname,theta,gamma,modify_times,target_class,model[i], proxy_arch)

if __name__ == '__main__':
    attackMTAModels()