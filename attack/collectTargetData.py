"""
Date: 2022-03-09
Author: sunhanwu@iie.ac.cn
Desc: Collect the labels output by the target model for training the subtitute model
"""

import torch
from TargetModel.FSNet.FSNet import FSNet
from TargetModel.FSNet.dataset import C2Data
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

def collectData(model:FSNet, dataloader, device, filename):
    """

    :param model:
    :param dataloader:
    :param filenale:
    :return:
    """
    collectData = []
    for batch_x, batch_y in tqdm(dataloader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        z_e = model.encode(batch_x)
        z_d, D = model.decode(z_e)
        z_dense = model.dense(z_e, z_d)
        # z_dense = F.softmax(z_dense)
        z_dense = torch.argmax(z_dense, dim=1, keepdim=True)
        collect = torch.cat((batch_x, z_dense), dim=1)
        collectData += collect.detach().cpu().numpy().tolist()
    collectData = np.array(collectData)
    np.save(filename, collectData)
    return collectData



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load('../modelFile/target_fsnet.pkt')
    fsnet = FSNet(state['param'])
    fsnet.load_state_dict(state['model_dict'])
    fsnet = fsnet.to(device)
    c2data = C2Data(number=200)
    batch_size = 32
    dataloader = DataLoader(c2data, batch_size, shuffle=True, drop_last=False)
    data = collectData(fsnet, dataloader, device, "../adversarialData/collectionData.npy")
    print(data.shape)
