from utils.leg_odometry import LegOdom
from tqdm import tqdm
import numpy as np
from utils.state_estimation import KF
from utils.kf_utils import quat_to_rot, get_inertia_matrix, get_jacobian
from utils.dynamics_model import *
from felan.models.log_chol_cadelac_pot_param import CaDeLaCLogChol, get_config_from_dict
from felan.train import load_model_fn
import jax.numpy as jnp
import jax
from collections import deque

def run_state_estimation(dt,
                         base_orient,
                         base_ang_vel,
                         joint_pos,
                         joint_vel,
                         joint_acc,
                         Q,
                         R,
                         base_vel=None,
                         base_acc=None,
                         joint_torque=None,
                         contact_forces=None,
                         contact_states=None,
                         contact_state_threshold=None,
                         model_name="aliengo",
                         L1=100,
                         L2=10000, # based on paper: L2 = L1²
                         contact_thresholds=np.array([15,15,15,15]),
                         est_mode=1,
                         cadelac_path=None,
                         tau_diff=None):
    """Run the Kalman Filter state estimation.

    est_mode:
    - 1 = estimation of pos, lin vel and ang vel (base acc as control input)
    - 2 = estimation of pos, lin vel, ang vel and contact force (identity in A)
    - 3 = estimation of pos, lin vel, ang vel and contact force (contact state diagonal block matrix in A)
    - 4 = estimation of pos, lin vel, ang vel and contact force (identity in A) with base acc inside KF
    """

    num_data = len(base_ang_vel)

    # results of EKF prediction step
    pos_predict_sim = []
    vel_predict_sim = []
    ang_vel_predict_sim = []
    c_force_predict_sim = []
    orient_predict_sim = []

    # results of EKF update step
    pos_update_sim = []
    vel_update_sim = []
    ang_vel_update_sim = []
    c_force_update_sim = []
    orient_update_sim = []

    c_force_measurement = []
    c_force_estimation = []
    c_state_estimation = []
    leg_odom_vel = []

    kf = KF(dt=dt, Q_diag=Q, R_diag=R, est_mode=est_mode)
    leg_odom = LegOdom(model_name=model_name)
    gm_observer = GMContactObserver(dt, L1, L2, contact_thresholds)
    gm_observer.f_hat_history = []
    if contact_forces is not None:
        single_force_val = contact_forces[0].shape == (4,)

    # for single print
    base_acc_info = True
    c_force_info = True
    c_state_info = True

    time_window = 0
    history_buf = deque(maxlen=time_window) if cadelac is not None else None
    if cadelac_path is not None:
        params, hyper = load_model_fn(cadelac_path.name, cadelac_path.parent)
        nn_config = get_config_from_dict(hyper)
        model = CaDeLaCLogChol(hyper['nv_dof'], nn_config)
        cadelac = jax.jit(model.apply)
        time_window = hyper["time_window"]
        history = None

    def build_feature_vector(base_orient_i, base_vel_i, base_ang_vel_i, base_pos_z_i, diff_tau_i):
        return np.concatenate([base_orient_i, base_vel_i, base_ang_vel_i, base_pos_z_i, diff_tau_i])

    for i in tqdm(range(num_data), desc="Running state estimation"):
        orient = base_orient[i]
        J_b, J_w = get_jacobian(leg_odom.env, orient, joint_pos[i], joint_vel[i])
        if cadelac_path is not None:
            if i > 0:
                history_buf.append(build_feature_vector(base_orient[i-1], base_vel[i-1], base_ang_vel[i-1], kf.get_pos()[2], tau_diff[i]))

        if cadelac_path is None or len(history_buf) < time_window:
            inertia_matrix = get_inertia_matrix(leg_odom.env)
            qfrc_bias = leg_odom.env.mjData.qfrc_bias.copy()
        else:
            history = jnp.array(history_buf)[None, ...]
            q = np.concatenate([kf.get_pos(), base_orient[i]]) # convert orient to euler
            qd = np.concatenate([kf.get_lin_vel(), base_ang_vel[i]]) 
            qdd = jnp.array(joint_acc[i])[None, ...]

            tau_pred, dEdt, extras = cadelac(params, q, qd, qdd, history)
            inertia_matrix = np.asarray(extras["M"])[0]
            qfrc_bias = np.asarray(extras["qfrc_bias"])[0]

        # estimate contact state if not given, either with thresholding singular contact force value or based on momentum
        if contact_states is None:
            if contact_forces is not None:
                if single_force_val: # singular value for each foot (prob. z-component)
                    if c_state_info:
                        print("Estimating contact state based on threshold")
                        c_state_info = False
                    c_state = estimate_contact_states(contact_forces[i], contact_state_threshold)
                else:
                    raise ValueError("Error for contact state estimation.")
            else:
                if c_state_info:
                    print("Estimating contact state based on momentum")
                    c_state_info = False
                J_w_stacked = np.vstack([J_w[leg][:, :] for leg in leg_odom.env.legs_order])
                c_state, f_hat = gm_observer.step(vel=np.concatenate([kf.get_lin_vel(), base_ang_vel[i], joint_vel[i]]),
                                                  M=inertia_matrix,
                                                  joint_torque=joint_torque[i],
                                                  J=J_w_stacked,
                                                  qfrc_bias=qfrc_bias)
                c_state_estimation.append(c_state)
                gm_observer.f_hat_history.append(f_hat)
        else:
            c_state = contact_states[i]

        # using leg odometry to produce measurement for linear velocity
        leg_odom.compute_leg_odometry(dt=dt,
                                      base_orient=orient,
                                      base_ang_vel=base_ang_vel[i],
                                      qdot=joint_vel[i],
                                      joint_pos=joint_pos[i],
                                      J_b=J_b,
                                      contact_state=c_state)
        
        # estimate x,y,z-contact force
        if contact_forces is None or single_force_val:
            if c_force_info:
                print("Estimating contact forces with joint torque")
                c_force_info = False
            c_force = estimate_contact_forces(joint_torque=joint_torque[i],
                                              contact_state=c_state,
                                              legs_order=leg_odom.env.legs_order,
                                              J_w=J_w)
            c_force_measurement.append(c_force)
        else:
            c_force = contact_forces[i]

        # estimate base accelaration based on floating base dynamics
        if base_acc is None:
            if base_acc_info:
                print("Estimating base accelaration with dynamics")
                base_acc_info = False
            base_acc_i = estimate_acc_from_contact_force_v3(joint_acc=joint_acc[i],
                                                            contact_forces=c_force,
                                                            contact_states=c_state,
                                                            contact_pos_b=leg_odom.p_b,
                                                            orient=quat_to_rot(orient),
                                                            M=inertia_matrix,
                                                            qfrc_bias=qfrc_bias)
        else:
            base_acc_i = base_acc[i]

        if est_mode == 1:
            kf.predict(u=base_acc_i)
            kf.update(z=np.concatenate([leg_odom.state.vel, base_ang_vel[i]]))
        elif est_mode in [2,3]:
            if est_mode == 3: 
                kf.update_A_contact_force(contact_state=c_state)
            kf.predict(u=base_acc_i)
            kf.update(z=np.concatenate([leg_odom.state.vel, base_ang_vel[i], c_force.flatten()]))
        elif est_mode == 4:
            kf.update_A_B_contact_forces(quat_to_rot(orient), leg_odom.p_b, c_state, inertia_matrix, qfrc_bias)
            kf.predict(u=np.concatenate([joint_acc[i], [1]]))
            kf.update(z=np.concatenate([leg_odom.state.vel, base_ang_vel[i], c_force.flatten()]))
        else:
            raise ValueError(f"est_mode of {est_mode} is not known.")

        pos_predict_sim.append(kf.get_pos("predict"))
        vel_predict_sim.append(kf.get_lin_vel("predict"))
        ang_vel_predict_sim.append(kf.get_ang_vel("predict"))
        if est_mode in [2,3,4]: c_force_predict_sim.append(kf.get_contact_force("predict"))

        pos_update_sim.append(kf.get_pos())
        vel_update_sim.append(kf.get_lin_vel())
        ang_vel_update_sim.append(kf.get_ang_vel())
        if est_mode in [2,3,4]: c_force_update_sim.append(kf.get_contact_force())

        leg_odom_vel.append(leg_odom.state.vel)

    result = {"pos_predict": np.array(pos_predict_sim),
              "vel_predict": np.array(vel_predict_sim),
              "ang_vel_predict": np.array(ang_vel_predict_sim),
              "c_force_predict": np.array(c_force_predict_sim),
              "pos_update": np.array(pos_update_sim),
              "vel_update": np.array(vel_update_sim),
              "ang_vel_update": np.array(ang_vel_update_sim),
              "c_force_update": np.array(c_force_update_sim),

              "c_force_meas": np.array(c_force_measurement),
              "c_state_est": np.array(c_state_estimation),
              "f_hat_history": np.array(gm_observer.f_hat_history),
              "leg_odom_vel": np.array(leg_odom_vel)
              }

    return result
