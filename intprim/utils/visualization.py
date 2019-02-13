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

    plt.plot(gen_trajectory[0], gen_trajectory[1], "--", color="#ff6a6a", label="Generated", linewidth=2.0)
    plt.plot(partial_observed_trajectory[0], partial_observed_trajectory[1], color="#6ba3ff", label="Observed",
             linewidth=2.0)
    if (mean_trajectory is not None):
        plt.plot(mean_trajectory[0], mean_trajectory[1], color="#85d87f", label="Mean")
    plt.legend()

    fig.suptitle('Probable trajectory')

    fig = plt.figure()

    obs_ratio = len(partial_observed_trajectory[0]) / len(train_trajectories[0][0])
    for index, degree in enumerate(gen_trajectory):
        new_plot = plt.subplot(len(trajectory), 1, index + 1)

        domain = np.linspace(0, obs_ratio, len(partial_observed_trajectory[index]))
        new_plot.plot(domain, partial_observed_trajectory[index], color="#6ba3ff", label="Observed.")

        domain = np.linspace(0, 1, len(gen_trajectory[index]))
        new_plot.plot(domain, gen_trajectory[index], "--", color="#ff6a6a", label="Generated.")

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

    ax.plot(gen_trajectory[0], gen_trajectory[1],gen_trajectory[2], "--", color="#ff6a6a", label="Predicted human", linewidth=2.0)
    ax.plot(gen_trajectory[3], gen_trajectory[4], gen_trajectory[5], "--", color="#ff6a6a", label="Generated robot",
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
    obs_ratio = len(partial_observed_trajectory[0]) / len(train_trajectories[0][0])
    for index, degree in enumerate(gen_trajectory):
        new_plot = plt.subplot(len(trajectory), 1, index + 1)

        domain = np.linspace(0, obs_ratio, len(partial_observed_trajectory[index]))
        new_plot.plot(domain, partial_observed_trajectory[index], color="#6ba3ff", label="Observed")

        domain = np.linspace(0, 1, len(gen_trajectory[index]))
        new_plot.plot(domain, gen_trajectory[index], "--", color="#ff6a6a", label="Generated")

        if (mean_trajectory is not None):
            domain = np.linspace(0, 1, len(mean_trajectory[index]))
            new_plot.plot(domain, mean_trajectory[index], color="#85d87f", label="Mean")

        if (true_trajectory is not None):
            domain = np.linspace(0, 1, len(true_trajectory[index]))
            new_plot.plot(domain, true_trajectory[index], color="yellow", label="Truth")

        new_plot.set_title('Trajectory for degree ' + str(index))
        new_plot.legend()

    plt.show()