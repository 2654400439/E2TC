"""
Date: 2022-04-13
Author: sunhanwu@iie.ac.cn
Desc: Calc the information gain of features
"""
import numpy as np
import pandas as pd
import json

# class InformationGain():
#     """
#     信息增益
#     """
#     def __init__(self, X, y):
#         self.X = X
#         self.y = y
#         self.totalSampleCount = X.shape[0]
#         self.totalSystemEntropy = 0
#         self.totalClassCountDict = {}
#         self.nonzeroPostion = X.T.nonzero()
#         self.igResult = []
#         self.wordExistSampleCount = 0
#         self.wordExistClassDict = {}
#         self.iter()
#
#     def get_result(self):
#         return self.igResult
#
#     def cal_total_system_entropy(self):
#         # 计算每个类别有多少
#         for label in self.y:
#             if label not in self.totalClassCountDict:
#                 self.totalClassCountDict[label] = 1
#             else:
#                 self.totalClassCountDict[label] += 1
#         for cls in self.totalClassCountDict:
#             probs = self.totalClassCountDict[cls] / float(self.totalSampleCount)
#             self.totalSystemEntropy -= probs * np.log(probs)
#
#     # 遍历nonzeroPosition，逐步计算出每个wor的信息增益
#     def iter(self):
#         self.cal_total_system_entropy()
#
#         pre = 0
#         for i in range(len(self.nonzeroPostion[0])):
#             if i != 0 and self.nonzeroPostion[0][i] != pre:
#                 for notappear in range(pre + 1, self.nonzeroPostion[0][i]):
#                     self.igResult.append(0.0)
#                 ig = self.cal_infomation_gain()
#                 self.igResult.append(ig)
#                 self.wordExistSampleCount = 0
#                 self.wordExistClassDict = {}
#                 pre = self.nonzeroPostion[0][i]
#             self.wordExistSampleCount += 1
#             yclass = self.y[self.nonzeroPostion[1][i]]
#             if yclass not in self.wordExistClassDict:
#                 self.wordExistClassDict[yclass] = 1
#             else:
#                 self.wordExistClassDict[yclass] += 1
#         # 计算最后一个特征的ig
#         ig = self.cal_infomation_gain()
#         self.igResult.append(ig)
#
#     def cal_infomation_gain(self):
#         x_exist_entropy = 0
#         x_nonexist_entropy = 0
#         for cls in self.wordExistClassDict:
#             probs = self.wordExistClassDict[cls] / float(self.wordExistSampleCount)
#             x_exist_entropy -= probs * np.log(probs)
#             probs = (self.totalClassCountDict[cls] - self.wordExistClassDict[cls]) / float(self.totalSampleCount - self.wordExistSampleCount)
#             if probs == 0:
#                 x_nonexist_entropy = 0
#             else:
#                 x_nonexist_entropy -= probs * np.log(probs)
#
#         for cls in self.totalClassCountDict:
#             if cls not in self.wordExistClassDict:
#                 probs = self.totalClassCountDict[cls] / float(self.totalSampleCount - self.wordExistSampleCount)
#                 x_nonexist_entropy -= probs * np.log(probs)
#         ig = self.totalSystemEntropy - ((self.wordExistSampleCount / float(self.totalSampleCount)) * x_exist_entropy +
#                                         ((self.totalSampleCount-self.wordExistSampleCount)/float(self.totalSampleCount)*x_nonexist_entropy))
#         return ig

import numpy as np
import pandas as pd
import math


class InformationGain():
    def __init__(self, feature, label):
        feature = np.array(feature)
        num_of_feature = np.shape(feature)[1]
        num_of_label = len(label)
        temp_ent = 0
        temp_condition_ent = 0
        information_gain_ratio = 0
        shanno_ent = []
        condition_ent = []
        information_gain_list = []
        information_gain_ratio_list = []

        for i in set(label):
            temp_ent += -(label.count(i) / num_of_label) * math.log(label.count(i) / num_of_label)
        for i in range(num_of_feature):
            feature1 = feature[:, i]
            sorted_feature = sorted(feature1)
            threshold = [(sorted_feature[inde - 1] + sorted_feature[inde]) / 2 for inde in range(len(feature1)) if
                         inde != 0]
            thre_set = set(threshold)
            if float(max(feature1)) in thre_set:
                thre_set.remove(float(max(feature1)))
            if min(feature1) in thre_set:
                thre_set.remove(min(feature1))
            information_gain = 0
            for thre in thre_set:
                lower = [label[s] for s in range(len(feature1)) if feature1[s] < thre]
                highter = [label[s] for s in range(len(feature1)) if feature1[s] > thre]
                H_l = 0
                for l in set(lower):
                    H_l += -(lower.count(l) / len(lower)) * math.log(lower.count(l) / len(lower))
                H_h = 0
                for h in set(highter):
                    H_h += -(highter.count(h) / len(highter)) * math.log(highter.count(h) / len(highter))
                temp_condition_ent = len(lower) / num_of_label * H_l + len(highter) / num_of_label * H_h
                gain = temp_ent - temp_condition_ent
                information_gain = max(information_gain, gain)
                information_gain_ratio = information_gain / temp_ent
            shanno_ent.append(temp_ent)
            condition_ent.append(temp_condition_ent)
            information_gain_list.append(information_gain)
            information_gain_ratio_list.append(information_gain_ratio)
            self.shannoEnt = shanno_ent[0]
            self.conditionEnt = condition_ent
            self.InformationGain = information_gain_list
            self.InformationGainRatio = information_gain_ratio_list
    def getEnt(self):
        return self.shannoEnt

    def getConditionEnt(self):
        return self.conditionEnt

    def getInformationGain(self):
        return self.InformationGain

    def getInformationGainRatio(self):
        return self.InformationGainRatio

if __name__ == '__main__':
    malwareFamily = ['Dridex', 'Gozi', 'Quakbot', 'Tofsee', 'TrickBot']
    ig_data = {}
    for cls in malwareFamily:
        path = "/home/sunhanwu/datasets/MTA/cicflownpy/{}.npy".format(cls)
        print("load {} data".format(cls))
        data = np.load(path, allow_pickle=True)
        data = pd.DataFrame(data)
        X = data.iloc[:, :-1].values.tolist()
        y = data.iloc[:, -1].values.tolist()
        ig = InformationGain(X, y)
        result = ig.getInformationGain()
        print("calc {} information gain done.".format(cls))
        ig_data[cls] = result
    with open("/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/result/MTA_Statistic_IG.json", 'w') as f:
        json.dump(ig_data, f)
