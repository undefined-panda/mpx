import numpy as np
from pathlib import Path
import utils.state_estimation
from utils.plot_data import *
from gym_quadruped.quadruped_env import QuadrupedEnv
import mujoco
from tqdm import tqdm
from rosbags.highlevel import AnyReader
from rosbags.typesys import get_typestore, Stores
from pathlib import Path
import numpy as np

def load_custom_dataset(dataset_path, sim_num):
    dataset = np.load(dataset_path)

    data = {"num_datasets": len(dataset["time"]),
            "num_datapoints": len(dataset["time"][0]),
            "dt": dataset["dt"][sim_num],
            "time": dataset["time"][sim_num],
            "base_pos": dataset["base_pos"][sim_num],
            "base_orient": dataset["base_orient"][sim_num],
            "base_vel" : dataset["base_vel"][sim_num],
            "base_ang_vel" : dataset["base_ang_vel"][sim_num],
            "base_acc" : dataset["base_acc"][sim_num],
            "joint_pos" : dataset["joint_pos"][sim_num],
            "joint_vel" : dataset["joint_vel"][sim_num],
            "joint_acc" : dataset["joint_acc"][sim_num],
            "joint_torque" : dataset["joint_torque"][sim_num],
            "contact_states" : dataset["contact_states"][sim_num],
            "contact_pos" : dataset["contact_pos"][sim_num],
            "contact_forces" : dataset["contact_forces"][sim_num]}
    
    print(f"Data loaded from {dataset_path}.")

    return data

def compute_pos_from_estimated_vel(velocities, dt):
    positions = []
    prev_pos = np.zeros((3,))
    for vel in velocities:
        next_pos = prev_pos + dt * vel
        positions.append(next_pos)
        prev_pos = next_pos
    
    return np.array(positions)

def estimation_error(ground_truth, estimation, title, subplot_title=None, plot=True, return_errors=False):
    errors = ground_truth - np.array(estimation)
    
    if plot:
        num_plots = ground_truth.shape[1]
        if subplot_title is None: titles = ["x", "y", "z"]
        fig, ax = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
        fig.suptitle(f"{title} Estimation Error over Time")

        for i in range(num_plots):
            ax[i].plot(errors[:, i])
            ax[i].set_title(titles[i]+" - Value")
            ax[i].set_xlabel("Time")
            ax[i].set_ylabel("Error")

        plt.show()
    
    if return_errors: return errors

def estimate_acc_from_contact_force(env, contact_states, contact_forces):
    m = float(np.sum(env.mjModel.body_mass))
    g = 9.81
    Fg = m * np.array([0, 0, g])

    force = np.zeros((3,))

    # sum jacobians of all legs that are in contact
    for i in range(4):
        c_i = contact_states[i]
        cf_i = contact_forces[i]

        force += c_i * cf_i

    acc = (force - Fg) / m

    return acc, force

def estimate_acc_from_estimated_contact_force(env, torque, contact_states):
    m = float(np.sum(env.mjModel.body_mass))
    g = 9.81
    Fg = m * np.array([0, 0, g])
    J_feet_linear = env.feet_jacobians(frame="world")

    force = np.zeros((3,))
    contact_forces = []

    # sum jacobians of all legs that are in contact
    for i in range(4):
        leg_name = env.legs_order[i]
        c_i = contact_states[i]

        J_lin = c_i * J_feet_linear[leg_name][:, 6:]
        force += J_lin @ torque
        contact_forces.append(force)

    acc = (force - Fg) / m

    return acc, force, contact_forces

def estimate_contact_state(contact_force, threshold):
    contact_force = np.array(contact_force)
    if contact_force.shape == (4,3): # x, y, z values for force
        mask = (np.sqrt(contact_force[:, 0]**2 + contact_force[:, 1]**2) <= contact_force[:, 2])
        contact_state = (contact_force != 0)[:, 0] & mask
    elif contact_force.shape == (4,): # single value for force
        contact_state = contact_force > threshold
    else:
        print(f"contact_force has invalid shape: {contact_force.shape}")
        return
    return contact_state

def run_state_estimation(dt,
                         base_orient,
                         base_ang_vel,
                         joint_pos,
                         joint_vel,
                         contact_pos,
                         Q, 
                         R,
                         base_pos=None,
                         base_acc=None,
                         joint_torque=None,
                         contact_forces=None,
                         contact_states=None,
                         contact_state_threshold=None):
    """_summary_

    Args:
        sim_num (int): _description_
        Q (float): Process noise. Smaller values mean trusting the model more
        R (float): Measurement Noise. Smaller values mean trusting the measurements more
    """
    
    # use first pos as init pos if given
    if base_pos is None:
        init_pos = np.zeros((3,))
    else:
        init_pos = base_pos[0]

    if base_acc is None: 
        base_acc = [None] * len(base_orient)
        print("Estimating base_acc.")
    if joint_torque is None: joint_torque = [None] * len(base_orient)
    if contact_forces is None: 
        contact_forces = [None] * len(base_orient)
        print("Estimating contact_force from contact_state and joint_torque")
    if contact_states is None: 
        contact_states = [None] * len(base_orient)
        print("Estimating contact state from contact_force and threshold")
    if contact_state_threshold is None: contact_state_threshold = [None] * len(base_orient)

    ekf = utils.state_estimation.EKF(init_pos=init_pos, dt=dt, Q_diag=Q, R_diag=R)

    # results of EKF prediction step
    pos_predict_sim = []
    vel_precict_sim = []
    P_predict_sim = []

    # results of leg odometry
    leg_odom_sim = []
    leg_odom_pos = []

    # results of EKF update step
    pos_update_sim = []
    vel_update_sim = []
    P_update_sim = []

    kalman_gain = []
    base_acc_est = []
    c_force_est = []
    z_tilde = []

    for i in tqdm(range(len(base_orient)), desc="Estimating state"):
        ekf.step(base_orient=base_orient[i],
                 base_acc=base_acc[i],
                 base_ang_vel=base_ang_vel[i],
                 joint_pos=joint_pos[i],
                 joint_vel=joint_vel[i],
                 joint_torque=joint_torque[i],
                 contact_states=contact_states[i],
                 contact_forces=contact_forces[i],
                 contact_pos=contact_pos[i],
                 contact_state_threshold=contact_state_threshold)

        pos_predict_sim.append(ekf.x_pred[:3])
        vel_precict_sim.append(ekf.x_pred[3:])
        P_predict_sim.append(ekf.P_pred)

        leg_odom_sim.append(ekf.z)
        leg_odom_pos.append(ekf.leg_odom_pos)
        base_acc_est.append(ekf.base_acc)
        c_force_est.append(ekf.c_force)

        pos_update_sim.append(ekf.x[:3])
        vel_update_sim.append(ekf.x[3:])
        P_update_sim.append(ekf.P)
        kalman_gain.append(ekf.K)
        z_tilde.append(ekf.z_tilde)

    result = {"pos_predict": np.array(pos_predict_sim),
              "vel_predict": np.array(vel_precict_sim),
              "P_predict": np.array(P_predict_sim),
              "leg_odom": np.array(leg_odom_sim),
              "leg_odom_pos": np.array(leg_odom_pos),
              "pos_update": np.array(pos_update_sim),
              "vel_update": np.array(vel_update_sim),
              "P_update": np.array(P_update_sim),
              "kalman_gain": np.array(kalman_gain),
              "z_tilde": np.array(z_tilde),
              "base_acc_est": np.array(base_acc_est),
              "c_force_est": np.array(c_force_est)
              }
    
    return result
