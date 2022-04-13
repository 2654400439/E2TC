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


if __name__ == '__main__':
    plt_edr()
    # plt_f1()