from __future__ import division
from enum import Enum
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_all_trajectories(train_trajectories, gen_trajectory, partial_observed_trajectory, mean_trajectory=None):
    fig = plt.figure("all trajectories")
    for trajectory in train_trajectories:
        plt.plot(trajectory[0], trajectory[1], "-", color="gray", alpha=0.3)
    start_partial = 0.0
    end_partial = float(partial_observed_trajectory.shape[1]) / (
            float(partial_observed_trajectory.shape[1]) + float(trajectory.shape[1]))

    plt.plot(partial_observed_trajectory[0], partial_observed_trajectory[1], color="#6ba3ff", label="Observed",
             linewidth=2.0)

    # gen mean, upper,lower bound
    plt.plot(gen_trajectory['mean'][0], gen_trajectory['mean'][1], "-", color="#ff6a6a", label="Generated mean", linewidth=2.0)
    plt.plot(gen_trajectory['up'][0], gen_trajectory['up'][1], "--", color="#fe5a6a", label="Generated up",
             linewidth=2.0)
    plt.plot(gen_trajectory['low'][0], gen_trajectory['low'][1], "--", color="#fe5a6a", label="Generated low",
             linewidth=2.0)

    if (mean_trajectory is not None):
        plt.plot(mean_trajectory[0], mean_trajectory[1], color="#85d87f", label="Mean")
    plt.legend()

    fig.suptitle('Probable trajectory')

    fig = plt.figure()

    obs_ratio = len(partial_observed_trajectory[0]) / len(train_trajectories[0][0])
    for index, degree in enumerate(partial_observed_trajectory):
        new_plot = plt.subplot(len(trajectory), 1, index + 1)

        domain = np.linspace(0, obs_ratio, len(partial_observed_trajectory[index]))
        new_plot.plot(domain, partial_observed_trajectory[index], color="#6ba3ff", label="Observed.")

        domain = np.linspace(0, 1, len(gen_trajectory['mean'][index]))
        new_plot.plot(domain, gen_trajectory['mean'][index], "-", color="#ff6a6a", label="Generated mean")
        new_plot.plot(domain, gen_trajectory['up'][index], "--", color="#fe5a6a", label="Generated up")
        new_plot.plot(domain, gen_trajectory['low'][index], "--", color="#fe5a6a", label="Generated low")


        if (mean_trajectory is not None):
            domain = np.linspace(0, 1, len(mean_trajectory[index]))
            new_plot.plot(domain, mean_trajectory[index], color="#85d87f", label="Mean.")

        new_plot.set_title('Trajectory for degree ' + str(index))
        new_plot.legend()

    plt.show()


# 3d plot
def plot_3d_trajectories(train_trajectories, gen_trajectory, partial_observed_trajectory, mean_trajectory=None, true_trajectory=None):
    fig = plt.figure("all trajectories")
    ax = fig.gca(projection = '3d')
    for trajectory in train_trajectories:
        #human
        ax.plot(trajectory[0], trajectory[1],trajectory[2], "-", color="gray", alpha=0.3)
        #robot
        ax.plot(trajectory[3], trajectory[4], trajectory[5], "-", color="gold", alpha=0.3)

    start_partial = 0.0
    end_partial = float(partial_observed_trajectory.shape[1]) / (
            float(partial_observed_trajectory.shape[1]) + float(trajectory.shape[1]))

    mean_gen=gen_trajectory['mean']
    up_gen=gen_trajectory['up']
    low_gen=gen_trajectory['low']

    ax.plot(mean_gen[0], mean_gen[1],mean_gen[2], "-", color="#ff6a6a", label=" Mean Predicted human", linewidth=2.0)
    ax.plot(mean_gen[3], mean_gen[4], mean_gen[5], "-", color="#ff6a6a", label=" Mean Generated robot",
            linewidth=2.0)

    ax.plot(up_gen[0], up_gen[1], up_gen[2], "--", color="#ff6a6a", label=" Up Predicted human", linewidth=2.0)
    ax.plot(up_gen[3], up_gen[4], up_gen[5], "--", color="#ff6a6a", label=" Up Generated robot",
            linewidth=2.0)

    ax.plot(low_gen[0], low_gen[1], low_gen[2], "--", color="#ff6a6a", label=" Low Predicted human", linewidth=2.0)
    ax.plot(low_gen[3], low_gen[4], low_gen[5], "--", color="#ff6a6a", label=" Low Generated robot",
            linewidth=2.0)

    if (mean_trajectory is not None):
        plt.plot(mean_trajectory[0], mean_trajectory[1],mean_trajectory[2], color="#85d87f", label="Mean")

    if (true_trajectory is not None):
        ax.plot(true_trajectory[0], true_trajectory[1], true_trajectory[2],
                color="black", label="True human", linewidth=2.0)
        ax.plot(true_trajectory[3], true_trajectory[4], true_trajectory[5],
                color="orange", label="True robot", linewidth=2.0)
    plt.legend()

    fig.suptitle('Probable trajectory')

    fig = plt.figure()

    # plot DOFs
    obs_ratio = len(partial_observed_trajectory[0]) / len(true_trajectory[0])
    for index, degree in enumerate(mean_gen):
        new_plot = plt.subplot(len(trajectory), 1, index + 1)

        domain = np.linspace(0, obs_ratio, len(partial_observed_trajectory[index]))
        new_plot.plot(domain, partial_observed_trajectory[index], color="#6ba3ff", label="Observed")

        domain = np.linspace(0, 1, len(mean_gen[index]))
        new_plot.plot(domain, mean_gen[index], "-", color="#ff6a6a", label="Generated mean")
        new_plot.plot(domain, up_gen[index], "--", color="#ff6a6a", label="Generated up")
        new_plot.plot(domain, low_gen[index], "--", color="#ff6a6a", label="Generated low")

        if (mean_trajectory is not None):
            domain = np.linspace(0, 1, len(mean_trajectory[index]))
            new_plot.plot(domain, mean_trajectory[index], color="#85d87f", label="Mean")

        if (true_trajectory is not None):
            domain = np.linspace(0, 1, len(true_trajectory[index]))
            new_plot.plot(domain, true_trajectory[index], color="yellow", label="Truth")

        new_plot.set_title('Trajectory for degree ' + str(index))
        new_plot.legend( bbox_to_anchor=(1.1, 1.05))

    plt.show()


def plot_phase(phase_list):
    fig = plt.figure("phase ratio")
    x=range(len(phase_list))
    plt.plot(phase_list["true_obs"],"-*",label="true phase")
    plt.plot(phase_list["pred_obs"],"-*", label="predict phase")
    plt.legend()
    # plt.show()


def plot_all_gens(gen_list):
    fig = plt.figure("gen trajs")
    for (id, traj_var) in enumerate(gen_list):
        traj = traj_var['mean']
        for index, degree in enumerate(traj):
            new_plot = plt.subplot(len(traj), 1, index + 1)
            domain = np.linspace(0, 1, len(traj[index]))
            new_plot.plot(domain, traj[index], "--", color="#ff6a6a", label="Generated"+str(id))
            new_plot.set_title('Trajectory for degree ' + str(index))
            new_plot.legend()


#TODO: print var
def plot_var(var_list):
    fig = plt.figure("var")
    var_matrix = np.asarray(var_list)
    var_matrix=var_matrix.T
    #ramdom select parameters
    domain = np.linspace(0, 1, var_matrix.shape[1])
    for i in range(1,var_matrix.shape[0],10):
        plt.plot(domain, var_matrix[i,:])

