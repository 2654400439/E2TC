"""
Date: 2022-03-22
Author: sunhanwu@iie.ac.cn
Desc: plot
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    truncate = 32
    rnn_fsnet_Tofsee = np.load("../rnn_fsnet_Tofsee_edr_gamma_list.npy")[:truncate]
    rnn_fsnet_Dridex = np.load("../rnn_fsnet_Dridex_edr_gamma_list.npy")[:truncate]
    rnn_fsnet_quackbot = np.load("../rnn_fsnet_Quakbot_edr_gamma_list.npy")[:truncate]
    rnn_fsnet_TrickBot = np.load("../rnn_fsnet_TrickBot_edr_gamma_list.npy")[:truncate]
    rnn_fsnet_Gozi = np.load("../rnn_fsnet_Gozi_edr_gamma_list.npy")[:truncate]

    maxLength = 1600
    plt.plot([x[0]/maxLength for x in rnn_fsnet_Tofsee], [x[1] for x in rnn_fsnet_Tofsee],
             marker='s', color='cyan', linestyle='-', markersize=4, linewidth='2')
    plt.plot([x[0]/maxLength for x in rnn_fsnet_Tofsee], [x[2] for x in rnn_fsnet_Tofsee],
             marker="^", color="cyan", linestyle=':', markersize=6, linewidth='2')

    plt.plot([x[0]/maxLength for x in rnn_fsnet_Dridex], [x[1] for x in rnn_fsnet_Dridex],
             marker='s', color='purple', linestyle='-', markersize=4, linewidth='2')
    plt.plot([x[0]/maxLength for x in rnn_fsnet_Dridex], [x[2] for x in rnn_fsnet_Dridex],
             marker="^", color="purple", linestyle=':', markersize=6, linewidth='2')

    plt.plot([x[0]/maxLength for x in rnn_fsnet_quackbot], [x[1] for x in rnn_fsnet_quackbot],
             marker='s', color='#00B04E', linestyle='-', markersize=4, linewidth='2')
    plt.plot([x[0]/maxLength for x in rnn_fsnet_quackbot], [x[2] for x in rnn_fsnet_quackbot],
             marker="^", color="#00B04E", linestyle=':', markersize=6, linewidth='2')

    plt.plot([x[0]/maxLength for x in rnn_fsnet_TrickBot], [x[1] for x in rnn_fsnet_TrickBot],
             marker='s', color='#F7921E', linestyle='-', markersize=4, linewidth='2')
    plt.plot([x[0]/maxLength for x in rnn_fsnet_TrickBot], [x[2] for x in rnn_fsnet_TrickBot],
             marker="^", color="#F7921E", linestyle=':', markersize=6, linewidth='2')

    plt.plot([x[0]/maxLength for x in rnn_fsnet_Gozi], [x[1] for x in rnn_fsnet_Gozi],
             marker='s', color='red', linestyle='-', markersize=4, linewidth='2')
    plt.plot([x[0]/maxLength for x in rnn_fsnet_Gozi], [x[2] for x in rnn_fsnet_Gozi],
             marker="^", color="red", linestyle=':', markersize=6, linewidth='2')
    labels = [
        Line2D([0], [0], color='cyan', lw=4, label="Tofsee"),
        Line2D([0], [0], color='purple', lw=4, label="Dridex"),
        Line2D([0], [0], color='#00B04E', lw=4, label="Quakbot"),
        Line2D([0], [0], color='#F7921E', lw=4, label="TrickBot"),
        Line2D([0], [0], color='red', lw=4, label="Gozi"),
        Line2D([0], [0], color='black', lw=2, linestyle="-", marker="s", markersize=4, label=r"$\psi_{s}^{rnn}$"),
        Line2D([0], [0], color='black', lw=2, linestyle=":", marker="^", markersize=6, label=r"$\psi_{t}^{FS-Net}$"),
    ]
    plt.legend(handles=labels, ncol=7)
    x_ticks = np.arange(0, 0.11, 0.01)
    plt.xticks(x_ticks)
    plt.ylabel("EDR")
    plt.xlabel(r"$\epsilon$")
    # plt.title(r"$\psi_{s}^{rnn}$-$\psi_{t}^{FS-Net}$")
    plt.savefig("../fig/edr_epsilon_a.eps", dpi=200, format='eps', bbox_inches="tight", pad_inches=0)
    plt.show()

def plot_edr_targets():
    """

    :return:
    """
    plt.figure(dpi=100, figsize=(24, 8))
    truncate = 100
    # todo: 需要补充rnn和autoencoder作为替代模型的迁移攻击下效果
    rnn_fsnet_TrickBot = np.load("../rnn_fsnet_TrickBot_edr_gamma_list.npy")[:truncate]
    rnn_svm_TrickBot = np.load("../rnn_svm_TrickBot_edr_gamma_list.npy")[:truncate]
    rnn_lr_TrickBot = np.load("../rnn_lr_TrickBot_edr_gamma_list.npy")[:truncate]
    rnn_dt_TrickBot = np.load("../rnn_dt_TrickBot_edr_gamma_list.npy")[:truncate]
    rnn_rf_TrickBot = np.load("../rnn_rf_TrickBot_edr_gamma_list.npy")[:truncate]
    rnn_knn_TrickBot = np.load("../rnn_knn_TrickBot_edr_gamma_list.npy")[:truncate]
    rnn_dnn_TrickBot = np.load("../rnn_dnn_TrickBot_edr_gamma_list.npy")[:truncate]
    rnn_lstm_TrickBot = np.load("../rnn_lstm_TrickBot_edr_gamma_list.npy")[:truncate]


    maxLength = 1600
    # plt.plot([x[0]/maxLength for x in rnn_fsnet_TrickBot], [x[1] for x in rnn_fsnet_TrickBot], label=r"$\psi_s^{LSTM}$",
    #          color='red', linestyle=':', linewidth='2')
    plt.plot([x[0]/maxLength for x in rnn_fsnet_TrickBot], [x[1] for x in rnn_fsnet_TrickBot], label=r"$\psi_s^{LSTM}$",
             marker='^', color='red', linestyle=':', markersize=4, linewidth='2')
    plt.plot([x[0]/maxLength for x in rnn_fsnet_TrickBot], [x[2] for x in rnn_fsnet_TrickBot], label=r"$\psi_t^{FS-Net}$",
             color="blue", linestyle='-', linewidth='2', marker='s', markersize=4)
    plt.plot([x[0]/maxLength for x in rnn_dnn_TrickBot], [x[1] for x in rnn_dnn_TrickBot], label=r"$\psi_t^{DNN}$",
             color='brown', linestyle='-', linewidth='2', marker='s', markersize=4)

    plt.plot([x[0]/maxLength for x in rnn_lstm_TrickBot], [x[1] for x in rnn_lstm_TrickBot], label=r"$\psi_t^{LSTM}$",
             color='olive', linestyle='-', linewidth='1', marker='s', markersize=4)

    plt.plot([x[0]/maxLength for x in rnn_svm_TrickBot], [x[1] for x in rnn_svm_TrickBot], label=r"$\psi_t^{SVM}$",
             marker='s', color='green', linestyle='-', markersize=4, linewidth='2')

    plt.plot([x[0]/maxLength for x in rnn_lr_TrickBot], [x[1] for x in rnn_lr_TrickBot], label=r"$\psi_t^{LR}$",
             marker='s', color='yellow', linestyle='-', markersize=4, linewidth='2')

    plt.plot([x[0]/maxLength for x in rnn_dt_TrickBot], [x[1] for x in rnn_dt_TrickBot], label=r"$\psi_t^{DT}$",
             marker='s', color='cyan', linestyle='-', markersize=4, linewidth='2')

    plt.plot([x[0]/maxLength for x in rnn_rf_TrickBot], [x[1] for x in rnn_rf_TrickBot], label=r"$\psi_t^{RF}$",
             marker='s', color='magenta', linestyle='-', markersize=4, linewidth='2')

    plt.plot([x[0]/maxLength for x in rnn_knn_TrickBot], [x[1] for x in rnn_knn_TrickBot], label=r"$\psi_t^{KNN}$",
             marker='s', color='purple', linestyle='-', markersize=4, linewidth='2')

    plt.axvline(0.1)
    x_ticks = np.arange(0,0.33, 0.02)
    plt.xticks(x_ticks)
    y_ticks = np.arange(0,1.1,0.1)
    plt.yticks(y_ticks)
    plt.legend(ncol=9, loc=[0.28,1.001])
    # todo: 需要仔细商榷
    # plt.title("")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"EDR")
    plt.savefig("../fig/edr_epsilon_b.eps", dpi=200, format='eps', bbox_inches="tight", pad_inches=0)
    plt.show()

if __name__ == '__main__':
    # plt_edr()
    # plt_f1()
    plot_edr_gamma()
    # plot_edr_targets()