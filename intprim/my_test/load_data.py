from __future__ import absolute_import, division, print_function

import numpy as np
import pickle
import random
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


MAX_LEN = 50


def data_gen(traj_seqs):
    data_dict = []
    random.shuffle(traj_seqs)
    for traj in traj_seqs:
        traj_len = len(traj['input'])
        traj_x = traj['input']
        traj_r = traj['output']
        traj_idx = traj['task_idx']
        one_data = dict(traj_x=traj_x, traj_r = traj_r, traj_lens=traj_len, task_idx=traj_idx)
        data_dict.append(one_data)

    return data_dict


def generate_data(file_name):
    pkl_file = open(file_name, 'rb')
    datasets = pickle.load(pkl_file)  # for python 2 to load the file generated by python 2
    print('number of trajectories:', len(datasets[0]))

    seqs = data_gen(datasets)

    visual_data(seqs)

    return seqs

def visual_data(data_dict):
    fig = plt.figure(0)
    ax = fig.gca(projection='3d')
    color = ['pink', 'brown', 'yellow', 'blue', 'green']
    for data in data_dict:
        traj_x = data['traj_x']
        traj_y = data['traj_r']
        id = data['task_idx']
        ax.plot(traj_x[:, 0], traj_x[:, 1], traj_x[:, 2], '-.', linewidth=1, color=color[id], label="human")
        ax.plot(traj_y[:, 0], traj_y[:, 1], traj_y[:, 2], '-.', linewidth=1, color='red',label="robot")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()





def main():
    # pkl_file = open('./reg_fmt_datasets.pkl','rb')
    # datasets = pickle.load(pkl_file)
    # print('length of tasks:', len(datasets))

    generate_data('../dataset/reg_fmt_datasets.pkl')


if __name__ == '__main__':
    main()