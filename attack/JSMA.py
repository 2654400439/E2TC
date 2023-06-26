"""
Date: 2022-03-08
Author: sunhanwu@iie.ac.cn
Desc: JSMA adversarial attack
"""
from attack.collectionDataset import CollectionDataset
from attack.adversarialDataset import AdversarialC2Data
from attack.excute import executeAttack
from torch.utils.data import DataLoader
from substituteModel.DNN import DNN
from utils.MTACICFlowMeter import MTACICFlowMeter
from substituteModel.AutoEncoder import AutoEncoder
from TargetModel.FSNet.FSNet import FSNet
from TargetModel.TargetDNN import TargetDNN
from substituteModel.RNN import RNN
from TargetModel.TargetLSTM import TargetLSTM
from TargetModel.FSNet.dataset import C2Data
from utils.CICIDSData import CICIDS
from utils.CICIDSData import dataconfig
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from attack.excute import attackFsnet
from advertorch.attacks import JacobianSaliencyMapAttack, GradientSignAttack, CarliniWagnerL2Attack, LinfBasicIterativeAttack


def jsmaAttack(model: nn.Module, num_class: int, dataloader: DataLoader, device, target_class: int, filename,
               clip_min=0.0, clip_max=1.0, loss_fn=None, theta=1.0, gamma=1.0, modify_times=10,
               filter_array: list =None):
    """
    :return:
    """
    model = model.to(device)
    adv_dataset = []
    jsma = JacobianSaliencyMapAttack(
        model, num_class, clip_min=-1, clip_max=1, loss_fn=loss_fn, theta=theta, gamma=gamma, modify_times=modify_times)
    fgsm = GradientSignAttack(model, loss_fn=loss_fn, eps=gamma,
                              clip_min=-1, clip_max=1, targeted=True)
    cw = CarliniWagnerL2Attack(model, num_classes=num_class, confidence=0.9,
                               targeted=True, clip_min=clip_min, clip_max=clip_max)
    bim = LinfBasicIterativeAttack(model, eps=gamma, nb_iter=5, eps_iter=1., targeted=True, clip_min=-1, clip_max=1)
    batch_x_clone = None
    adv_x = None
    origin_dataset = []
    for batch_x, batch_y in tqdm(dataloader):
        batch_x = batch_x.float().to(device)
        batch_x_clone = batch_x.clone()
        batch_y = batch_y.to(device)
        target = torch.from_numpy(np.array([target_class] * batch_x.shape[0]))
        target = target.to(device)
        adv_x = jsma.perturb(batch_x, target)
        # adv_x = fgsm.perturb(batch_x, target)
        # adv_x = bim.perturb(batch_x, target)
        # adv_x = cw.perturb(batch_x, target)
        if filter_array:
            filter = torch.tensor(filter_array)
            filter = filter.repeat((batch_x.shape[0], 1))
            adv_x[filter == 0] = -0xffffffffffff
            batch_x_clone[filter == 1] = -0xffffffffffff
            adv_x = torch.max(adv_x, batch_x_clone)
        adv_dataset += torch.cat((adv_x, batch_y), dim=1).detach().cpu().numpy().tolist()
        origin_dataset += torch.cat((batch_x_clone, batch_y), dim=1).detach().cpu().numpy().tolist()
        # adv_dataset.append(adv_x[0].detach().cpu().numpy().tolist() + [torch.argmax(batch_y[0]).item()])
    print("origin sample:\n {}".format(batch_x_clone[-1]))
    print("adversarial sample:\n {}".format(adv_x[-1]))
    adv_dataset = np.array(adv_dataset, dtype=np.int64)
    print(adv_dataset.shape)
    print("save filename:{}".format(filename))
    np.save(filename, adv_dataset)
    np.save(filename.replace('advdata', 'oridata'), origin_dataset)
    return adv_dataset


def main(theta, gamma, name, filter_name, filter_array=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # generate(device, 'rnn')

    arch = "dnn"
    botname = name
    normal = "CTUNone"
    target_arch = "lstm"
    sample_size = 580
    adv_model_filename =  "../modelFile/proxy_cicids_{}_{}.pkt".format(arch, botname)
    # target_model_filename = "../modelFile/target_{}_{}_{}.pkt".format(target_arch, botname, normal)
    print("Load adv model: {}".format(adv_model_filename))
    adv_state = torch.load(adv_model_filename)
    print("model param: {}".format(adv_state['param']))
    if arch == "rnn":
        adv_model = RNN(adv_state['param'])
    elif arch == 'dnn':
        adv_model = DNN(adv_state['param'])
    elif arch == 'autoencoder':
        adv_model = AutoEncoder(adv_state['param'])
    else:
        adv_model = FSNet(adv_state['param'])
    adv_model.load_state_dict(adv_state['model_dict'])
    adv_model.to(device)
    for param in adv_model.parameters():
        param.requires_grad = False

    # gamma = 20
    # theta = 20
    target_class = 0
    modify_times = 2
    loss_fn = nn.CrossEntropyLoss()
    dataset = CICIDS(botname, norm=True)
    dataloader = DataLoader(dataset, batch_size=256,
                            shuffle=True, drop_last=False)
    f = open('../adversarialData/result.txt', 'w')
    print("theta: {}, gamma: {}".format(theta, gamma))
    print("theta: {}, gamma: {}".format(theta, gamma), file=f)
    advdata_filename = "../adversarialData/advdata_cicids_{}_{}_{}_{}_{}_{}.npy".format(arch, botname, theta, gamma, modify_times, filter_name)
    adv_dataset = jsmaAttack(adv_model, 2, dataloader, device, target_class=target_class, filename=advdata_filename,
                             loss_fn=loss_fn, clip_min=-1e9, clip_max=1e9, theta=theta, gamma=gamma, modify_times=modify_times, filter_array=filter_array)

    batch_size = 64
    adv_data = AdversarialC2Data(advdata_filename, target_class=target_class, keep_target=True, norm=False)
    adv_loader = DataLoader(adv_data, batch_size=batch_size, shuffle=True)
    print("attack {} model".format(arch))
    print("attack {} model".format(arch), file=f)
    executeAttack(adv_loader, adv_model, device, file=f)

    # print("attack fsnet model")
    # print("attack fsnet model", file=f)
    # attackFsnet(theta, gamma, f, advdata_filename, target_model_filename, device=device)
    f.close()

def mainMta(theta, gamma, name, sample_size, proxy_arch, feature_type, modify_times, filter_array=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # generate(device, 'rnn')

    arch = proxy_arch
    botname = name
    normal = "CTUNone"
    target_arch = "lstm"
    adv_model_filename =  "../modelFile/proxy_mta_{}_{}_{}.pkt".format(feature_type, arch, botname)
    # target_model_filename = "../modelFile/target_{}_{}_{}.pkt".format(target_arch, botname, normal)
    print("Load adv model: {}".format(adv_model_filename))
    adv_state = torch.load(adv_model_filename)
    print("model param: {}".format(adv_state['param']))
    if arch == "rnn":
        adv_model = RNN(adv_state['param'])
    elif arch == 'dnn':
        adv_model = DNN(adv_state['param'])
    elif arch == 'autoencoder':
        adv_model = AutoEncoder(adv_state['param'])
    elif arch == 'lstm':
        adv_model = TargetLSTM(adv_state['param'])
    else:
        adv_model = FSNet(adv_state['param'])
    adv_model.load_state_dict(adv_state['model_dict'])
    adv_model.to(device)
    for param in adv_model.parameters():
        param.requires_grad = False

    # gamma = 20
    # theta = 20
    target_class = 0
    # modify_times = 16
    loss_fn = nn.CrossEntropyLoss()
    dataset = C2Data(botname, number=sample_size, sequenceLen=30, feature_type=feature_type)
    # dataset = CICIDS(botname, norm=True)
    dataloader = DataLoader(dataset, batch_size=256,
                            shuffle=True, drop_last=False)
    print("theta: {}, gamma: {}".format(theta, gamma))
    advdata_filename = "../adversarialData/advdata_mta_{}_{}_{}_{}_{}_{}.npy".format(feature_type, arch, botname, theta, gamma, modify_times)
    adv_dataset = jsmaAttack(adv_model, 2, dataloader, device, target_class=target_class, filename=advdata_filename,
                             loss_fn=loss_fn, clip_min=0, clip_max=1600, theta=theta, gamma=gamma, modify_times=modify_times, filter_array=filter_array)

    batch_size = 64
    adv_data = AdversarialC2Data(advdata_filename, target_class=target_class, keep_target=True, norm=False)
    adv_loader = DataLoader(adv_data, batch_size=batch_size, shuffle=True)
    print("attack {} model".format(arch))
    executeAttack(adv_loader, adv_model, device, file=None)

    # print("attack fsnet model")
    # print("attack fsnet model", file=f)
    # attackFsnet(theta, gamma, f, advdata_filename, target_model_filename, device=device)
    # f.close()

def mainMtaCICFlowMeter(theta, gamma, botname, sample_size, proxy_arch, modify_times):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    botname = botname
    adv_model_filename =  "../modelFile/proxy_mta_cicflowmeter_{}_{}.pkt".format(proxy_arch, botname)
    print("Load adv model: {}".format(adv_model_filename))
    adv_state = torch.load(adv_model_filename)
    print("model param: {}".format(adv_state['param']))
    if proxy_arch == "rnn":
        adv_model = RNN(adv_state['param'])
    elif proxy_arch == 'dnn':
        adv_model = TargetDNN(adv_state['param'])
    elif proxy_arch == 'autoencoder':
        adv_model = AutoEncoder(adv_state['param'])
    elif proxy_arch == 'lstm':
        adv_model = TargetLSTM(adv_state['param'])
    else:
        adv_model = FSNet(adv_state['param'])
    adv_model.load_state_dict(adv_state['model_dict'])
    adv_model.to(device)
    for param in adv_model.parameters():
        param.requires_grad = False

    target_class = 0
    loss_fn = nn.CrossEntropyLoss()
    dataset = MTACICFlowMeter(botname, number=sample_size)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=False)
    print("theta: {}, gamma: {}".format(theta, gamma))
    advdata_filename = "../adversarialData/advdata_mta_cicflowmeter_{}_{}_{}_{}_{}.npy".format(proxy_arch, botname, theta, gamma, modify_times)
    jsmaAttack(adv_model, 2, dataloader, device, target_class=target_class, filename=advdata_filename,
                             loss_fn=loss_fn, clip_min=-1, clip_max=1, theta=theta, gamma=gamma, modify_times=modify_times, filter_array=None)
    batch_size = 64
    adv_data = AdversarialC2Data(advdata_filename, target_class=target_class, keep_target=True, norm=False)
    adv_loader = DataLoader(adv_data, batch_size=batch_size, shuffle=True)
    print("attack {} model".format(proxy_arch))
    # executeAttack(adv_loader, adv_model, device, file=None)

    # print("attack fsnet model")
    # print("attack fsnet model", file=f)
    # attackFsnet(theta, gamma, f, advdata_filename, target_model_filename, device=device)
    # f.close()

if __name__ == '__main__':
# for gamma in range(2, 11, 2):
#     for theta in range(11, 21, 1):
    filter_times = [0, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 69, 70, 71, 72, 73, 74, 75, 76]
    filter_sizes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14, 35, 36, 37, 38, 39,40,41,51,52,53,55,56,57,58,59,60,61,62,63,64]
    filter_both = filter_times + filter_sizes
    filter_times = [1 if x in filter_times else 0 for x in range(77)]
    filter_sizes = [1 if x in filter_sizes else 0 for x in range(77)]
    filter_both = [1 if x in filter_both else 0 for x in range(77)]
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
    numbers = [2580, 2580, 1600, 690, 1250]
    proxy_arch = "dnn"
    # for i in range(5):
    #     mainMtaCICFlowMeter(theta=10., gamma=0.9, botname=Botnets[i], sample_size=numbers[i], proxy_arch=proxy_arch, modify_times=20)
    main(theta=1., gamma=2.0, name="Botnet", filter_name="vis_times", filter_array=filter_times)