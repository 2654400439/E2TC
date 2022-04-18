"""
Date: 2022-04-18
Author: sunhanwu@iie.ac.cn
Desc: Test the trade-off
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from TargetModel.FSNet.dataset import C2Data
from substituteModel.DNN import DNN
from substituteModel.AutoEncoder import AutoEncoder
from TargetModel.FSNet.FSNet import FSNet
from substituteModel.RNN import RNN
from attack.adversarialDataset import AdversarialC2Data
from attack.JSMA import jsmaAttack
from attack.excute import attackFsnet
from attack.excute import executeAttack
import numpy as np
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def attackGamma(gamma: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # generate(device, 'rnn')

    arch = "autoencoder"
    botname = "TrickBot"
    normal = "CTUNone"
    target_arch = "fsnet"
    sample_size = 650
    adv_model_filename = "../modelFile/subtitute_{}_{}_{}.pkt".format(
        arch, botname, normal)
    target_model_filename = "../modelFile/target_{}_{}_{}.pkt".format(
        target_arch, botname, normal)
    # print("Load adv model: {}".format(adv_model_filename))
    adv_state = torch.load(adv_model_filename)
    # print("model param: {}".format(adv_state['param']))
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
    modify_times = 20
    loss_fn = nn.CrossEntropyLoss()
    dataset = C2Data(botname, number=sample_size, sequenceLen=30)
    dataloader = DataLoader(dataset, batch_size=256,
                            shuffle=True, drop_last=False)
    f = open('../adversarialData/result.txt', 'w')
    print("theta: {}, gamma: {}".format(5, gamma))
    print("theta: {}, gamma: {}".format(5, gamma), file=f)
    advdata_filename = "../adversarialData/advdata_{}_{}_{}_{}_{}_{}.npy".format(
        arch, botname, normal, 5, gamma, modify_times)
    adv_dataset = jsmaAttack(adv_model, 2, dataloader, device, target_class=target_class, filename=advdata_filename,
                             loss_fn=loss_fn, clip_min=60, clip_max=1600, theta=5, gamma=gamma, modify_times=modify_times)

    batch_size = 64
    adv_data = AdversarialC2Data(
        advdata_filename, target_class=target_class, keep_target=True)
    adv_loader = DataLoader(adv_data, batch_size=batch_size, shuffle=True)
    # print("attack {} model".format(arch))
    print("attack {} model".format(arch), file=f)
    _, _, _, sur_conmetrix = executeAttack(
        adv_loader, adv_model, device, file=f)
    EDR_sur = sur_conmetrix[1][0] / np.sum(sur_conmetrix[1])

    # print("attack fsnet model")
    print("attack fsnet model", file=f)
    con_metrix = attackFsnet(
        5, gamma, f, advdata_filename, target_model_filename, device=device)
    f.close()
    EDR = con_metrix[1][0] / np.sum(con_metrix[1])
    print("sur_edr: {}, target_edr:{}".format(EDR_sur, EDR))
    return EDR_sur, EDR


if __name__ == '__main__':
    EDR_list = []
    for gamma in range(0, 800, 5):
        sur_edr, edr = attackGamma(float(gamma))
        EDR_list.append((gamma, sur_edr, edr))
    np.save("../autoencoder_fsnet_TrickBot_edr_gamma_list.npy", EDR_list)
