from tqdm import tqdm
from utils.state_estimation import KF
import numpy as np
from pathlib import Path

def run_state_estimation(dt,
                         base_orient,
                         base_ang_vel,
                         joint_pos,
                         joint_vel,
                         contact_pos,
                         Q, 
                         R,
                         init_pos=None,
                         base_acc=None,
                         joint_acc=None,
                         joint_torque=None,
                         contact_forces=None,
                         contact_states=None,
                         contact_state_threshold=None,
                         result_dir=None,
                         file_name=None):
    """_summary_

    Args:
        sim_num (int): _description_
        Q (float): Process noise. Smaller values mean trusting the model more
        R (float): Measurement Noise. Smaller values mean trusting the measurements more
    """
    
    # use first pos as init pos if given
    if init_pos is None:
        init_pos = np.zeros((3,))

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
    if joint_acc is None: joint_acc = [None] * len(joint_acc)

    kf = KF(init_pos=init_pos, dt=dt, Q_diag=Q, R_diag=R)

    # results of EKF prediction step
    pos_predict_sim = []
    vel_precict_sim = []
    ang_vel_predict_sim = []
    orient_predict_sim = []
    P_predict_sim = []

    # results of leg odometry
    leg_odom_sim = []
    leg_odom_pos = []

    # results of EKF update step
    pos_update_sim = []
    vel_update_sim = []
    ang_vel_update_sim = []
    orient_update_sim = []
    P_update_sim = []

    kalman_gain = []
    base_acc_est = []
    base_acc2_est = []
    base_acc3_est = []
    c_force_est = []
    z_tilde = []

    for i in tqdm(range(len(base_orient)), desc="Estimating state"):
        kf.step(base_orient=base_orient[i],
                base_acc=base_acc[i],
                base_ang_vel=base_ang_vel[i],
                joint_pos=joint_pos[i],
                joint_vel=joint_vel[i],
                joint_acc=joint_acc[i],
                joint_torque=joint_torque[i],
                contact_states=contact_states[i],
                contact_forces=contact_forces[i],
                contact_pos=contact_pos[i],
                contact_state_threshold=contact_state_threshold)

        pos_predict_sim.append(kf.get_pos("predict"))
        vel_precict_sim.append(kf.get_lin_vel("predict"))
        ang_vel_predict_sim.append(kf.get_ang_vel("predict"))
        orient_predict_sim.append(kf.get_orient("predict"))
        P_predict_sim.append(kf.P_pred)

        pos_update_sim.append(kf.get_pos())
        vel_update_sim.append(kf.get_lin_vel())
        ang_vel_update_sim.append(kf.get_ang_vel())
        orient_update_sim.append(kf.get_orient())
        P_update_sim.append(kf.P)

        leg_odom_sim.append(kf.leg_odom_vel)
        leg_odom_pos.append(kf.leg_odom_pos)
        base_acc_est.append(kf.base_acc)
        base_acc2_est.append(kf.base_acc2)
        base_acc3_est.append(kf.base_acc3)
        c_force_est.append(kf.c_force)

        kalman_gain.append(kf.K)
        z_tilde.append(kf.z_tilde)

    result = {"pos_predict": np.array(pos_predict_sim),
              "vel_predict": np.array(vel_precict_sim),
              "ang_vel_predict": np.array(ang_vel_predict_sim),
              "orient_predict": np.array(orient_predict_sim),
              "P_predict": np.array(P_predict_sim),
              "leg_odom": np.array(leg_odom_sim),
              "leg_odom_pos": np.array(leg_odom_pos),
              "pos_update": np.array(pos_update_sim),
              "vel_update": np.array(vel_update_sim),
              "ang_vel_update": np.array(ang_vel_update_sim),
              "orient_update":np.array(orient_update_sim),
              "P_update": np.array(P_update_sim),
              "kalman_gain": np.array(kalman_gain),
              "z_tilde": np.array(z_tilde),
              "base_acc_est": np.array(base_acc_est),
              "base_acc2_est": np.array(base_acc2_est),
              "base_acc3_est": np.array(base_acc3_est),
              "c_force_est": np.array(c_force_est)
              }
    
    if result_dir and file_name:
        save_path = Path.cwd().parent / result_dir
        save_path.mkdir(parents=True, exist_ok=True)
        np.savez(f"{save_path}/{file_name}.npz", **result)
        print(f"Result saved in {save_path}/{file_name}")
    
    return result, kf
