"""
Date: 2022-03-02
Author: sunhanwu@iie.ac.cn
Desc: label the iscx-bot-2014 dataset,
      Use the ip that comes with the dataset as the label,
      because the effect of using the ssl blacklist label is not good
"""
import pandas as pd
import os
from tqdm import tqdm


def loadAttackIps():
    """
    load the malicious ips and the mapping between ip and bot family
    :return: a list of malicious ips, a dict, key: ip, value: bot family
    """
    with open('./iscx-bot-2014-malicious-ips.txt', 'r') as f:
        malicious_ips = [x.strip() for x in f.readlines()]
    mapping_df = pd.read_csv('ip2type.csv')
    mapping = {}
    for index, row in mapping_df.iterrows():
        if row['ip'] not in mapping.keys():
            mapping[row['ip']] = row['type']
        else:
            continue
    return malicious_ips, mapping

def labelISCXData(filename, malicious_ips, mapping):
    """
    label the pcap as a type, Only when two IPs in a flow are not in the malicious IP list,
    it belongs to normal, otherwise the malicious bot family is found according to the mapping,
    and the one that cannot be found belongs to the others family
    Note: All flows are TLS flows with ssl handshake
    :param filename: pcap file
    :param malicious_ips: list of the malicious ips
    :param mapping: dict of the mapping between ip and bot famliy
    :return: famliy type: str
    """
    tshark = "tshark -r {} -T fields -e ip.src -e ip.dst | head -n 1".format(filename)
    r = os.popen(tshark)
    tshark_result = r.readlines()
    r.close()
    if len(tshark_result) == 0:
        return "None"
    tshark_result = tshark_result[0].strip().split('\t')
    if (tshark[0] not in malicious_ips) and (tshark_result[1] not in malicious_ips):
        famliy = "Normal"
    else:
        if tshark[0] in malicious_ips:
            malicious_ip = tshark_result[0]
        if tshark[1] in malicious_ips:
            malicious_ip = tshark_result[1]
        if malicious_ip in mapping.keys():
            famliy = mapping[malicious_ip]
        else:
            famliy = "Others"

    # mv
    if os.path.exists("/home/sunhanwu/datasets/iscx-bot-2014/labeldata/{}".format(famliy)):
        index = len(os.listdir("/home/sunhanwu/datasets/iscx-bot-2014/labeldata/{}/".format(famliy))) + 1
    else:
        os.mkdir("/home/sunhanwu/datasets/iscx-bot-2014/labeldata/{}".format(famliy))
        index = 1
    newfilename = "/home/sunhanwu/datasets/iscx-bot-2014/labeldata/{}/{}.pcap".format(famliy, index)
    mv = "mv {} {}".format(filename, newfilename)
    os.system(mv)
    return famliy





if __name__ == '__main__':
    malicious_ips, mapping = loadAttackIps()
    filenames = [os.path.join("/home/sunhanwu/datasets/iscx-bot-2014/labeldata/None/", x) for x in os.listdir("/home/sunhanwu/datasets/iscx-bot-2014/labeldata/None/")]
    statistic = {}
    for file in tqdm(filenames):
        family = labelISCXData(file, malicious_ips, mapping)
        if family in statistic.keys():
            statistic[family] += 1
        else:
            statistic[family] = 1
    print(statistic)