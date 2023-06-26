"""
Date: 2022-06-22
Author: sunhanwu@iie.ac.cn
Desc: Base class for all feature extractors
"""
from joblib import Parallel, delayed

class FeatureExtractor():
    """
    the base class for feature extractor
    """
    def __init__(self, extrator, jobs=-1):
        self.extractor = extrator
        self.jobs = jobs

    def extract(self, src:str, dst:str):
        """
        extract features
        :param src: pcap path or pacps dir path
        :param dst: csv file output path
        :return: None
        """
        raise "not implemented"

    def __extractOneFile(self, input_file:str, output_file:str):
        """
        extract one pcap to features
        :param input_file: pcap file path
        :param output_file: csv file path
        :return:
        """
        raise "not implemented"

    def multiProcess(self, extractOneFile, filenames:list, outputfiles:list):
        """
        :param extractOneFile:
        :param filenames:
        :param output:
        :return:
        """
        Parallel(n_jobs=self.jobs)(
            (delayed(extractOneFile)(input_file, output_file) for input_file, output_file in zip(filenames, outputfiles))
        )


