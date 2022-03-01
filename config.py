import numpy as np
kitsune_config = {
    # maximum size for any autoencoder in the ensemble layer
    "maxAE": 10,
    #the number of instances taken to learn the feature mapping (the ensemble's architecture)
    "FMgrace": 5000,
    #the number of instances used to train the anomaly detector (ensemble itself)
    "ADgrace": 50000,
    # the number of packets from the input file to process
    "packet_limit": 1400000,
    # learning_rate
    "lr": 0.1,
    # the pcap, pcapng, or tsv file which you wish to process.
    #TODO: need pcap/pcapng/tsv file path
    "path": "/home/sunhanwu/datasets/kitsune/OS_Scan/OS_Scan_pcap.pcapng.tsv"
}