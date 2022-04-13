"""
Date: 2022-03-22
Author: sunhanwu@iie.ac.cn
Desc: gan based attack, use the subtitute as the D, and G generate the adversarial samples
"""
import torch
import torch.nn as nn
from substituteModel.AutoEncoder import AutoEncoder
class Generator(nn.Module):
    """
    generate the adversarial sample
    """
    def __init__(self, param):
        super(Generator, self).__init__()


