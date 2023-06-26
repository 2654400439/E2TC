"""
Date: 2022-06-22
Author: sunhanwu@iie.ac.cn
Desc: extract kdd features from datasets
"""
from FeatureExtractor import FeatureExtractor
import os

class KDDFeatureExtractor(FeatureExtractor):
    """
    https://github.com/AI-IDS/kdd99_feature_extractor
    extract kdd features
    """
    def __init__(self, extractor:str, jobs=8):
        """
        constructor
        :param extractor: extractor path
        :param config: extractor param
        """
        super(KDDFeatureExtractor, self).__init__(extractor, jobs)

    def __extractOneFile(self, input_file:str, output_file:str):
        """
        extract one pcap to features
        :param input_file: pcap file path
        :param output_file: csv file path
        :return:
        """
        print("[KDD Feature Extractor] {} => {}".format(input_file, output_file))
        cmd = "{} {} > {}".format(self.extractor, input_file, output_file)
        # os.system(cmd)

    def extract(self, src:str, dst:str):
        """
        extract features
        :param src: pcap path or pacps dir path
        :param dst: csv file output path
        :return: None
        """
        if not os.path.exists(dst):
            os.mkdir(dst)
        if src.endswith("pcap"):
            output_file = os.path.join(dst, src.split('/')[-1].replace('pcap', 'csv'))
            self.__extractOneFile(src, output_file)
        else:
            filenames = [os.path.join(src, x) for x in os.listdir(src)]
            outputfiles = [os.path.join(dst, x.split('/')[-1].replace('pcap', 'csv')) for x in filenames]
            print(KDDFeatureExtractor.__extractOneFile)
            super().multiProcess(KDDFeatureExtractor.__extractOneFile, filenames, outputfiles)




if __name__ == '__main__':
    kdd = KDDFeatureExtractor("/home/sunhanwu/tools/kdd99_feature_extractor/build/src/kdd99extractor", jobs=-1)
    kdd.extract("/home/sunhanwu/datasets/MTA/labeldata/Adware/", "/home/sunhanwu/Adware/")
