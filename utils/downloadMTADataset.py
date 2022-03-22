"""
Date: 2022-03-01
Author: sunhanwu@iie.ac.cn
Desc: download all the pcaps from https://malware-traffic-analysis.net and unzip them
"""
import requests
from lxml import etree
import re
import os


def getAllUrls():
    """
    Desc: get all the pcaps url from target website
    :return: a list of the data url
    """
    # Different target urls in different years
    targetUrl = "https://www.malware-traffic-analysis.net/2022/index.html"
    response = requests.get(targetUrl)
    if response.status_code != 200:
        raise "Connection Error"
    html = etree.HTML(response.text)
    results = []
    yearsUls = html.xpath('//*[@id="main_content"]/div[1]/ul')
    for i in range(1, len(yearsUls) + 1):
        yearDetailsUrls = html.xpath('//*[@id="main_content"]/div[1]/ul[{}]/li'.format(i))
        for j in range(1, len(yearDetailsUrls) + 1):
            url = html.xpath('//*[@id="main_content"]/div[1]/ul[{}]/li[{}]/a[1]/@href'.format(i, j))[0]
            if 'isc.sans.edu' in url:
                continue
            url = "https://malware-traffic-analysis.net/" + targetUrl.split("/")[-2] + "/" + url
            results.append(url)
    return results


def getAPcap(url):
    """
    get pcap file url, use wget to download the file and unzip it
    :param url: pcap file web url
    :return:
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise "Connection Error"
    html = response.text
    pcapLinks = re.findall(r'<a\sclass="menu_link"\shref="([a-zA-Z0-9\-]+)\.pcap\.zip">.*?pcap\.zip</a>', html, re.S)
    for item in pcapLinks:
        url_ = "/".join(url.split("/")[:-1])
        downloadLink = url_ + "/" + item + ".pcap.zip"
        # wget download file
        if os.path.exists("/home/sunhanwu/datasets/MTA/pcaps2/" + downloadLink.split('/')[-1]):
            continue
        wget = "wget -P /home/sunhanwu/datasets/MTA/pcaps2/ " + downloadLink
        print(wget)
        os.system(wget)

        # unzip file
        unzip = "unzip -P infected /home/sunhanwu/datasets/MTA/pcaps2/" + downloadLink.split('/')[-1] + " -d /home/sunhanwu/datasets/MTA/pcaps2/"
        print(unzip)
        os.system(unzip)


if __name__ == '__main__':
    urls = getAllUrls()
    for url in urls:
        print(url)
        getAPcap(url)
