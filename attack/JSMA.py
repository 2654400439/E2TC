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
from substituteModel.AutoEncoder import AutoEncoder
from TargetModel.FSNet.FSNet import FSNet
from substituteModel.RNN import RNN
from TargetModel.FSNet.dataset import C2Data
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from attack.excute import attackFsnet
from advertorch.attacks import JacobianSaliencyMapAttack, GradientSignAttack, CarliniWagnerL2Attack, LinfBasicIterativeAttack


def jsmaAttack(model: nn.Module, num_class: int, dataloader: DataLoader, device, target_class: int, filename, clip_min=0.0, clip_max=1.0, loss_fn=None, theta=1.0, gamma=1.0, modify_times=10):
    """

    :return:
    """
    model = model.to(device)
    adv_dataset = []
    jsma = JacobianSaliencyMapAttack(
        model, num_class, clip_min=clip_min, clip_max=clip_max, loss_fn=loss_fn, theta=theta, gamma=gamma, modify_times=modify_times)
    fgsm = GradientSignAttack(model, loss_fn=loss_fn, eps=gamma,
                              clip_min=clip_min, clip_max=clip_max, targeted=True)
    cw = CarliniWagnerL2Attack(model, num_classes=num_class, confidence=1.0,
                               targeted=True, clip_min=clip_min, clip_max=clip_max)
    bim = LinfBasicIterativeAttack(model, eps=gamma, nb_iter=500, eps_iter=1., targeted=True, clip_min=0, clip_max=1600)
    batch_x_clone = None
    adv_x = None
    for batch_x, batch_y in tqdm(dataloader):
        batch_x = batch_x.float().to(device)
        batch_x_clone = batch_x.clone()
        batch_y = batch_y.to(device)
        target = torch.from_numpy(np.array([target_class] * batch_x.shape[0]))
        target = target.to(device)
        # adv_x = jsma.perturb(batch_x, target)
        # adv_x = fgsm.perturb(batch_x, target)
        adv_x = bim.perturb(batch_x, target)
        # adv_x = cw.perturb(batch_x, target)

        adv_dataset += torch.cat((adv_x, batch_y), dim=1).detach().cpu().numpy().tolist()
        # adv_dataset.append(adv_x[0].detach().cpu().numpy().tolist() + [torch.argmax(batch_y[0]).item()])
    print("origin sample:\n {}".format(batch_x_clone[-1]))
    print("adversarial sample:\n {}".format(adv_x[-1]))
    adv_dataset = np.array(adv_dataset, dtype=np.int64)
    print(adv_dataset.shape)
    print("save filename:{}".format(filename))
    np.save(filename, adv_dataset)
    return adv_dataset


def main(theta, gamma):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # generate(device, 'rnn')

    arch = "autoencoder"
    botname = "Gozi"
    normal = "CTUNone"
    target_arch = "fsnet"
    sample_size = 580
    adv_model_filename =  "../modelFile/subtitute_{}_{}_{}.pkt".format(arch, botname, normal)
    target_model_filename = "../modelFile/target_{}_{}_{}.pkt".format(target_arch, botname, normal)
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
    modify_times = 20
    loss_fn = nn.CrossEntropyLoss()
    dataset = C2Data(botname, number=sample_size, sequenceLen=30)
    dataloader = DataLoader(dataset, batch_size=256,
                            shuffle=True, drop_last=False)
    f = open('../adversarialData/result.txt', 'w')
    print("theta: {}, gamma: {}".format(theta, gamma))
    print("theta: {}, gamma: {}".format(theta, gamma), file=f)
    advdata_filename = "../adversarialData/advdata_{}_{}_{}_{}_{}_{}.npy".format(arch, botname, normal, theta, gamma, modify_times)
    adv_dataset = jsmaAttack(adv_model, 2, dataloader, device, target_class=target_class, filename=advdata_filename,
                             loss_fn=loss_fn, clip_min=60, clip_max=1600, theta=theta, gamma=gamma, modify_times=modify_times)

    batch_size = 64
    adv_data = AdversarialC2Data(advdata_filename, target_class=target_class, keep_target=True)
    adv_loader = DataLoader(adv_data, batch_size=batch_size, shuffle=True)
    print("attack {} model".format(arch))
    print("attack {} model".format(arch), file=f)
    executeAttack(adv_loader, adv_model, device, file=f)

    print("attack fsnet model")
    print("attack fsnet model", file=f)
    attackFsnet(theta, gamma, f, advdata_filename, target_model_filename, device=device)
    f.close()


if __name__ == '__main__':
# for gamma in range(2, 11, 2):
#     for theta in range(11, 21, 1):
    main(5, 200.)