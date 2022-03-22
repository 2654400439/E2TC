"""
Date: 2022-03-07
Author: sunhanwu@iie.ac.cn
Desc: some utils func
"""
import torch
import torch.nn as nn


def save_model(model, optimizer, param, hyper, FPR, filename):
    """
    save fsnet model
    :param model:
    :param optimizer:
    :param filename:
    :return:
    """
    state = {
        'model_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "param": param,
        'hyper': hyper,
        'FPR': FPR
    }
    print("save model: {}".format(filename))
    torch.save(state, filename)

