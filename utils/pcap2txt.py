"""
Date: 2022-03-03
Author: sunhanwu@iie.ac.cn
Desc: use tshark parse the pcap, export some useful fields to csv file
"""
import os
import multiprocessing as mp

# data path config dict
DATA_PATH = {
    'text': "/home/sunhanwu/datasets/MTA/text/Ransomware",
    'pcap': "/home/sunhanwu/datasets/MTA/labeldata/Ransomware"
}

# tshark config dict
DEFAULT_TSHARK_CONFIG = {
        'T': 'fields',
        'filename': [],
        'fields': [
                # frame
                # 'frame.number','frame.time_epoch','frame.len',
                'frame.time_epoch','frame.len',
                # eth
                # 'eth.src','eth.dst',
                # ip
                # 'ip.src','ip.dst','ip.proto',
                # gtp
                # 'gtp','gtp.teid',
                # ppp
                # 'ppp',
                # tcp
                # 'tcp.stream','tcp.srcport','tcp.dstport','tcp.flags.syn','tcp.flags','tcp.window_size_value','tcp.window_size_scalefactor','tcp.window_size','tcp.payload',
                # http
                # 'http','http.request','http.request','http.request.uri','http.request.method','http.user_agent','http.response','http.response.code',
                # ssl
                # 'ssl.record','ssl','ssl.record.content_type','ssl.record.length','ssl.handshake.type','ssl.handshake.ciphersuite','ssl.handshake.extensions_server_name',
                # ssh
                # 'ssh',
                # udp
                # 'udp.stream','udp.srcport','udp.dstport',
                # DNS
                # 'dns','dns.flags.response','dns.qry.name','dns.resp.ttl','dns.time',
                # GQUIC
                # 'gquic',
                # ssdp
                # 'ssdp',
                # ntp
                # 'ntp.flags.mode'
        ],
        'Y':'not _ws.malformed.expert'
}

class HandlePcap():
    def __init__(self):
        self.filename = None
        self.command = None
        self.T = 'fields'
        self.Fields = []
        self.OutputPath = None
        self.Y = None

    def _Set_T(self,T):
        self.T = T

    def _Set_Fields(self,Fields):
        self.Fields = Fields

    def _Set_OutPut(self,Output):
        self.OutputPath = Output

    def _Set_Filename(self,filename):
        self.filename = filename

    def _Set_Y(self,Y):
        self.Y = Y

    def FormatCommand(self):
        """Generate tshark command based on parameters"""
        if self.T == 'json': #以json格式输出
            self.command = 'tshark -r {} -T {} -Y "{}" -x > {} '.format(self.filename,self.T,self.Y,self.OutputPath)
        elif self.T == 'fields': #以字段格式输出
            self.command = 'tshark -r {} -T {} -Y "{}" '.format(self.filename,self.T,self.Y)
            for field in self.Fields:
                self.command += ' -e {} '.format(field)
            self.command += ' > {}'.format(self.OutputPath)
    def _ScanPcap(self,filename):
        """Processing a single pcap file"""
        assert self.T != None
        assert self.Fields != None
        self.filename = filename
        # self.filename = filename[:-5].replace('.', '_') + ".pcap"
        self.OutputPath = '{}.csv'.format(os.path.join(DATA_PATH['text'], "_".join(self.filename.split('/')[-1].split('.')[:-1])))
        self.FormatCommand()
        print("[  Tshark  ] %s"%self.command)
        os.system(self.command)

    def ScanPcaps(self,Paras:dict,jobs = 5):
        """Processing tcap files with tshark"""
        """use jobs control the number of the process"""
        self._Set_T(Paras['T'])
        self._Set_Y(Paras['Y'])
        self._Set_Fields(Paras['fields'])
        self._Set_Filename(Paras['filenames'])
        index = 0
        while(index + jobs) <= len(Paras['filenames']):
            pool = mp.Pool(processes=jobs)
            pool.map(self._ScanPcap,Paras['filenames'][index:index + jobs])
            pool.close() #不允许再增加新的进程
            pool.join() #等待进程所有进程跑完
            index = index + jobs
        pool = mp.Pool(processes=len(Paras['filenames']) - index)
        pool.map(self._ScanPcap,Paras['filenames'][index:])
        pool.close() #不允许再增加新的进程
        pool.join() #等待进程所有进程跑完

if __name__ == '__main__':
    TSHARK_CONFIG = DEFAULT_TSHARK_CONFIG.copy()
    TSHARK_CONFIG['filenames'] = [os.path.join(DATA_PATH['pcap'], x) for x in os.listdir(DATA_PATH['pcap'])]
    handle = HandlePcap()
    handle.ScanPcaps(TSHARK_CONFIG, jobs=32)