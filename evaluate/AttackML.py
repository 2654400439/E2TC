import numpy as np
import pandas as pd
from TargetModel.FSNet.dataset import C2Data as MTASequence
from utils.MTACICFlowMeter import MTACICFlowMeter as MTAStatistic
from TargetModel.TargetLR import TargetLR
from TargetModel.TargetDT import TargetDT
from TargetModel.TargetSVM import TargetSVM
from TargetModel.TargetIF import TargetIF

def tarinMlModel(arch, param, dataset, botname):
    """

    :param arch: 模型架构
    :param param:
    :param dataset:
    :param botname:
    :return:
    """



