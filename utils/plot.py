"""
Date: 2022-03-22
Author: sunhanwu@iie.ac.cn
Desc: plot
"""
import matplotlib.pyplot as plt
import numpy as np


def plt_f1():
    families = [
        'Dridex',
        'Gozi',
        'Quakbot',
        'Tofsee'
    ]
    x = np.array([0,1,2,3])
    total_width, n = 0.8, 4
    width = total_width / n
    x = x - width - width

    fsnet = [0.9947,1.,0.9844,0.9842]
    autoender = [0.995, 0.96, 0.9448, 0.9201]
    fsnet_after_attack = [0.7779, 0.7333, 0.6736, 0.6423]
    autoender_after_attack = [0.5285, 0.3322, 0.6570, 0.4857]
    plt.bar(x, fsnet, width=width, label='target model')
    plt.bar(x+width, autoender, width, label='subtitute model')
    plt.bar(x+width*2, autoender_after_attack, width, label='subtitute after attack')
    plt.bar(x+width*3, fsnet_after_attack, width, label='target after attack')
    # plt.xlim([-1, 4])
    plt.title('F1 value before/after adversarial sample attack')
    plt.xlabel('botnet family')
    plt.ylabel('f1')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.xticks(x+width, families)
    plt.show()

def plt_edr():
    families = [
        'Dridex',
        'Gozi',
        'Quakbot',
        'Tofsee'
    ]
    x = np.array([0,1,2,3])
    total_width, n = 0.4, 2
    width = total_width / n

    fsnet = [0.825, 0.95, 0.58, 0.8]
    autoender = [0.885, 1.0, 0.59, 0.845]
    plt.bar(x, fsnet, width=width, label='target model')
    plt.bar(x+width, autoender, width, label='subtitute model')
    plt.ylim([0, 1])
    plt.xlabel('botnet family')
    plt.ylabel('EDR(Escape Detection Rate)')
    plt.title("adversarial samples Escape Detection Rate")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.xticks(x+width, families)
    plt.show()

def plot_edr_gamma():
    plt.figure(dpi=100, figsize=(24, 8))
    truncate = 80
    rnn_fsnet_quackbot = np.load("../rnn_fsnet_Quakbot_edr_gamma_list.npy")[:truncate]
    rnn_fsnet_TrickBot = np.load("../rnn_fsnet_TrickBot_edr_gamma_list.npy")[:truncate]
    dnn_fsnet_TrickBot = np.load("../dnn_fsnet_TrickBot_edr_gamma_list.npy")[:truncate]
    autoencoder_fsnet_TrickBot = np.load("../autoencoder_fsnet_TrickBot_edr_gamma_list.npy")[:truncate]

    plt.plot([x[0] for x in dnn_fsnet_TrickBot], [x[1] for x in dnn_fsnet_TrickBot], label="Quakbot:dnn edr",
             marker='s', color='#00B04E', linestyle='-', markersize=3, linewidth='1')
    plt.plot([x[0] for x in dnn_fsnet_TrickBot], [x[2] for x in dnn_fsnet_TrickBot], label="Quakbot:fsnet edr",
             marker="*", color="#00B04E", linestyle='-.', markersize=3, linewidth='1')

    plt.plot([x[0] for x in autoencoder_fsnet_TrickBot], [x[1] for x in autoencoder_fsnet_TrickBot], label="TrickBot:autoencoder edr",
             marker='s', color='#F7921E', linestyle='-', markersize=3, linewidth='1')
    plt.plot([x[0] for x in autoencoder_fsnet_TrickBot], [x[2] for x in autoencoder_fsnet_TrickBot], label="TrickBot:fsnet edr",
             marker="*", color="#F7921E", linestyle='-.', markersize=3, linewidth='1')

    plt.plot([x[0] for x in rnn_fsnet_TrickBot], [x[1] for x in rnn_fsnet_TrickBot], label="TrickBot:rnn edr",
             marker='s', color='red', linestyle='-', markersize=3, linewidth='1')
    plt.plot([x[0] for x in rnn_fsnet_TrickBot], [x[2] for x in rnn_fsnet_TrickBot], label="TrickBot:fsnet edr",
             marker="*", color="red", linestyle='-.', markersize=3, linewidth='1')
    plt.legend()
    plt.show()

def plot_edr_targets():
    """

    :return:
    """
    plt.figure(dpi=100, figsize=(24, 8))
    truncate = 80
    # todo: 需要补充rnn和autoencoder作为替代模型的迁移攻击下效果
    rnn_fsnet_TrickBot = np.load("../rnn_fsnet_TrickBot_edr_gamma_list.npy")[:truncate]
    rnn_svm_TrickBot = np.load("../rnn_svm_TrickBot_edr_gamma_list.npy")[:truncate]
    rnn_lr_TrickBot = np.load("../rnn_lr_TrickBot_edr_gamma_list.npy")[:truncate]
    rnn_dt_TrickBot = np.load("../rnn_dt_TrickBot_edr_gamma_list.npy")[:truncate]
    rnn_rf_TrickBot = np.load("../rnn_rf_TrickBot_edr_gamma_list.npy")[:truncate]
    rnn_knn_TrickBot = np.load("../rnn_knn_TrickBot_edr_gamma_list.npy")[:truncate]
    rnn_dnn_TrickBot = np.load("../rnn_dnn_TrickBot_edr_gamma_list.npy")[:truncate]
    rnn_lstm_TrickBot = np.load("../rnn_lstm_TrickBot_edr_gamma_list.npy")[:truncate]


    plt.plot([x[0] for x in rnn_fsnet_TrickBot], [x[1] for x in rnn_fsnet_TrickBot], label="TrickBot:rnn edr",
             marker='s', color='red', linestyle='-', markersize=3, linewidth='1')
    plt.plot([x[0] for x in rnn_fsnet_TrickBot], [x[2] for x in rnn_fsnet_TrickBot], label="TrickBot:fsnet edr",
             marker="*", color="blue", linestyle='-.', markersize=3, linewidth='1')

    plt.plot([x[0] for x in rnn_svm_TrickBot], [x[1] for x in rnn_svm_TrickBot], label="TrickBot:svm edr",
             marker='s', color='green', linestyle='-.', markersize=3, linewidth='1')

    plt.plot([x[0] for x in rnn_lr_TrickBot], [x[1] for x in rnn_lr_TrickBot], label="TrickBot:lr edr",
             marker='s', color='yellow', linestyle='-.', markersize=3, linewidth='1')

    plt.plot([x[0] for x in rnn_dt_TrickBot], [x[1] for x in rnn_dt_TrickBot], label="TrickBot:dt edr",
             marker='s', color='cyan', linestyle='-.', markersize=3, linewidth='1')

    plt.plot([x[0] for x in rnn_rf_TrickBot], [x[1] for x in rnn_rf_TrickBot], label="TrickBot:rf edr",
             marker='s', color='magenta', linestyle='-.', markersize=3, linewidth='1')

    plt.plot([x[0] for x in rnn_knn_TrickBot], [x[1] for x in rnn_knn_TrickBot], label="TrickBot:knn edr",
             marker='s', color='purple', linestyle='-.', markersize=3, linewidth='1')

    plt.plot([x[0] for x in rnn_dnn_TrickBot], [x[1] for x in rnn_dnn_TrickBot], label="TrickBot:dnn edr",
             marker='s', color='brown', linestyle='-.', markersize=3, linewidth='1')

    plt.plot([x[0] for x in rnn_lstm_TrickBot], [x[1] for x in rnn_lstm_TrickBot], label="TrickBot:lstm edr",
             marker='s', color='olive', linestyle='-.', markersize=3, linewidth='1')
    plt.legend()
    # todo: 需要仔细商榷
    plt.title("")
    plt.xlabel("")
    plt.ylabel("")
    plt.savefig("../fig/test1.eps", dpi=200, format='eps')
    plt.show()

if __name__ == '__main__':
    # plt_edr()
    # plt_f1()
    # plot_edr_gamma()
    plot_edr_targets()