import numpy as np
from intprim import basis_model
from intprim import bayesian_interaction_primitives as bip
import intprim.utils.visualization as vis

import load_data

def seperate_data(data_dict):
    traj_x_set = []
    traj_r_set = []
    for one_data in data_dict:
        if one_data['task_idx']==1:
            xs = one_data['traj_x']
            rs = one_data['traj_r']
            traj_x_set.append(xs)
            traj_r_set.append(rs)

    return traj_x_set,traj_r_set


def record_data():
    test_num = 5

    #load dataset
    data_dict = load_data.generate_data('../dataset/reg_fmt_datasets.pkl')
    traj_x_set,traj_r_set = seperate_data(data_dict)
    # train_set = traj_set[0:-test_num]
    # test_set = traj_set[-test_num:]

    # Initialize a BIP object with 2 DOF named "X" and "Y" which are approximated by 8 basis functions.
    primitive = bip.BayesianInteractionPrimitive(6, ['hx', 'hy','hz','rx','ry','rz'], 10)

    phase_velocities = []

    train_trajectories = []  # for plot

    for (traj_x, traj_r) in zip(traj_x_set[0:-test_num],traj_r_set[0:-test_num]):
        train_trajectory = np.concatenate((traj_x[:,0:3],traj_r[:,0:3]), axis = 1).T
        primitive.add_demonstration(train_trajectory)
        phase_velocities.append(1.0 / train_trajectory.shape[1])

        train_trajectories.append(train_trajectory)

    # test
    for (test_x,test_r) in zip(traj_x_set[-test_num:],traj_r_set[-test_num:]):
        test_trajectory = np.concatenate((test_x[:,0:3],traj_r[:,0:3]), axis=1).T
        test_trajectory_partial = np.array(test_trajectory, copy = True)
        test_trajectory_partial[3:6,:] = 0.0
        observable_samples = 20

        # observation_noise = np.array([[0.2,0.0,0.0],[0.0,0.2,0.0],[0.0,0.0,0.2]])
        observation_noise = np.zeros((6,6), dtype = np.float64)
        np.fill_diagonal(observation_noise,0.3)
        observation_noise=observation_noise
        for i in range(3,6):
            observation_noise[i][i]=10000


        primitive.initialize_filter(phase_velocity=np.mean(phase_velocities), phase_var=np.var(phase_velocities))
        gen_trajectory, phase = primitive.generate_probable_trajectory_recursive(
            test_trajectory_partial[:, :observable_samples], observation_noise, num_samples=50 - observable_samples)

        mean_trajectory = primitive.get_mean_trajectory()

        # primitive.plot_partial_trajectory(gen_trajectory, test_trajectory_partial[:, :observable_samples],
        #                                   mean_trajectory)

        # plot training trajectory
        vis.plot_3d_trajectories(train_trajectories, gen_trajectory, test_trajectory[:, :observable_samples],
                                  mean_trajectory,test_trajectory)


def main():
    record_data()


if __name__ == '__main__':
    main()