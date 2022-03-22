"""
Date: 2022-03-02
Author: sunhanwu@iie.ac.cn
Desc: split all tls flows from pcap and use https://sslbl.abuse.ch/ blacklist to label them
"""
from pcap_splitter.splitter import PcapSplitter
import os
import pandas as pd
from tqdm import tqdm

def splitFlows(srcPath, dstPath):
    """
    use tool(pcap-splitter:https://github.com/shramos/pcap-splitter) to split tls flows
    :param filePath: pcap file path
    :return: None
    """
    filenames = [os.path.join(srcPath, x) for x in os.listdir(srcPath)]
    for file in filenames:
        ps = PcapSplitter(file)
        print(ps.split_by_session(dstPath, pkts_bpf_filter="port 443"))

def labelByJa3(filename, blackListDict: dict, dataset="MTA"):
    """
    use ja3 tool to generate ja3 digest
    :param filepath: pcap files path
    :return:
    """
    cmd = "ja3 --json {}".format(filename)
    r = os.popen(cmd)
    result = r.read()
    r.close()
    if len(result) == 0:
        return "None"
    result = eval(result)
    if len(result) == 0:
        return "None"
    ja3_digest = result[0]['ja3_digest']
    malicious_family = blackListDict.get(ja3_digest, "None")
    # print("{}: {}".format(filename, malicious_family))

    # rename
    index = 0
    if os.path.exists("/home/sunhanwu/datasets/{}/labeldata/{}".format(dataset, malicious_family)):
        index = len(os.listdir("/home/sunhanwu/datasets/{}/labeldata/{}".format(dataset, malicious_family))) + 1
        pass
    else:
        os.mkdir("/home/sunhanwu/datasets/{}/labeldata/{}".format(dataset, malicious_family))
        index = 1
    new_filename = "/".join(filename.split('/')[:-2]) + "/labeldata/{}/{}.pcap".format(malicious_family, index)
    if os.path.exists(new_filename):
        return malicious_family
    mv = "mv {} {}".format(filename, new_filename)
    # print(mv)
    os.system(mv)
    return malicious_family



def loadBlackList(filename):
    """
    load the ssl black list
    :param filename: black list file, csv file
    :return: dict, key: ja3 digest, value: malicious family
    """
    data = pd.read_csv(filename)
    resutl = {}
    for index, row in data.iterrows():
        resutl[row['ja3_md5']] = row['Listingreason']
    return resutl

def loadSSLBlackList(filename):
    """
    load the https://sslbl.abuse.ch/blacklist/sslblacklist.csv
    :param filename:
    :return:
    """
    data = pd.read_csv(filename)
    result = {}
    for index, row in data.iterrows():
        result[row['SHA1']] = row['Listingreason'].split(' ')[0]
    return result

def parseFingerprinnt(pcapfile):
    """
    use certgrep parse the certificate fingerprint from pcap
    :param pcapfile:
    :return: a list of sha1
    """
    certgrepPath = "/home/sunhanwu/tools/certgrep/dist/certgrep-linux-amd64"
    cmd = "{} -p {} --log-to-stdout 2>/dev/null | grep fingerprint".format(certgrepPath, pcapfile)
    r = os.popen(cmd)
    resutl = r.readlines()
    r.close()
    sha1List = []
    for item in resutl:
        for key_value in item.split(' '):
            if 'fingerprint' in key_value:
                sha1List.append(key_value.split(':')[1])
    return sha1List

def labelBySHA1(filename, blackListDict: dict, dataset="MTA"):
    """

    :param filename:
    :param blackListDict:
    :param dataset:
    :return:
    """
    sha1List = parseFingerprinnt(filename)
    label = "Normal"
    for item in sha1List:
        if item in blackListDict.keys():
            label = blackListDict[item]
            print("{}:{}".format(filename, label))




if __name__ == '__main__':
    # blackList = loadSSLBlackList('./sslblacklist.csv')
    # filenames = [os.path.join("/home/sunhanwu/datasets/MTA-bot-2014/flows/", x) for x in os.listdir("/home/sunhanwu/datasets/MTA/flows/")]
    # for file in filenames:
    #     labelBySHA1(file, blackList)
    # parseFingerprinnt('/home/sunhanwu/datasets/MTA/labeldata/Adware/100.pcap')
    # splitFlows("/home/sunhanwu/datasets/MTA/pcaps/", "/home/sunhanwu/datasets/MTA/flows2")
    filenames = [os.path.join("/home/sunhanwu/datasets/MTA/flows2/", x) for x in os.listdir("/home/sunhanwu/datasets/MTA/flows2/")]
    family_sta = {}
    blackListDict = loadBlackList("./ja3_fingerprints.csv")
    for file in tqdm(filenames):
    # for file in filenames:
        family = labelByJa3(file, blackListDict, dataset="MTA")
        if family in family_sta.keys():
            family_sta[family] += 1
        else:
            family_sta[family] = 1
    print(family_sta)
