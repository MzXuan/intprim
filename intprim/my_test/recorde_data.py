import numpy as np
from intprim import basis_model
from intprim import bayesian_interaction_primitives as bip

import load_data

def seperate_data(data_dict):
    traj_set = []
    y_set = []
    z_set = []
    for one_data in data_dict:
        if one_data['task_idx']==1:
            xs = one_data['traj_x']
            traj_set.append(xs)

    return traj_set



def record_data():
    test_num = 5

    #load dataset
    data_dict = load_data.generate_data('../dataset/reg_fmt_datasets.pkl')
    traj_set = seperate_data(data_dict)
    train_set = traj_set[0:-test_num]
    test_set = traj_set[-test_num:]

    # Initialize a BIP object with 2 DOF named "X" and "Y" which are approximated by 8 basis functions.
    primitive = bip.BayesianInteractionPrimitive(3, ['X', 'Y','Z'], 10)

    phase_velocities = []

    for traj in train_set:
        train_trajectory = traj[:,[0,1,2]].T
        primitive.add_demonstration(train_trajectory)
        phase_velocities.append(1.0 / train_trajectory.shape[1])


    # test
    for test_traj in test_set:
        test_trajectory_partial = test_traj[:,[0,1,2]].T
        observable_samples = 30

        observation_noise = np.array([[0.2,0.0,0.0],[0.0,0.2,0.0],[0.0,0.0,0.2]])

        primitive.initialize_filter(phase_velocity=np.mean(phase_velocities), phase_var=np.var(phase_velocities))
        gen_trajectory, phase = primitive.generate_probable_trajectory_recursive(
            test_trajectory_partial[:, :observable_samples], observation_noise, num_samples=50 - observable_samples)

        mean_trajectory = primitive.get_mean_trajectory()

        primitive.plot_partial_trajectory(gen_trajectory, test_trajectory_partial[:, :observable_samples],
                                          mean_trajectory)



def main():
    record_data()


if __name__ == '__main__':
    main()
