from utils.leg_odometry import LegOdom, LegOdomJax
from tqdm import tqdm
import numpy as np
import jax.numpy as jnp
from utils.state_estimation import KF, KFJax
from utils.kf_utils import quat_to_rot, quat_to_rot_jax

def run_state_estimation(dt,
                          base_orient,
                          base_ang_vel,
                          joint_pos,
                          joint_vel,
                          joint_acc,
                          joint_torque,
                          contact_states,
                          Q,
                          R,
                          init_pos=None,
                          contact_forces=None,
                          contact_state_threshold=None,
                          contact_coupling="dynamics",
                          contact_force_decay="constant",
                          model_name="aliengo"):
    """Run the Kalman Filter state estimation over a full trajectory.

    Args:
        Q (float): process noise. Smaller values mean trusting the model more.
        R (float): measurement noise. Smaller values mean trusting the measurements more.
        init_pos (np.ndarray | None): initial base position. Defaults to the origin.
        contact_forces (np.ndarray | None): per-foot contact force. If None, it is
            estimated from joint torque and contact state via the dynamics model.
        contact_state_threshold (float | None): contact force threshold used when
            contact states need to be estimated from contact forces.
        contact_coupling (str): see KF.__init__ - how contact force couples into
            lin_vel/ang_vel ("dynamics", "direct" or "identity").
        contact_force_decay (str): see KF.__init__ - behaviour of the C_FORCE->C_FORCE
            block of A ("constant" or "contact_gated").
    """

    num_data = len(base_orient)

    if init_pos is None: init_pos = np.zeros(3)
    if contact_forces is None: contact_forces = [None] * num_data

    # results of EKF prediction step
    pos_predict_sim = []
    vel_predict_sim = []
    ang_vel_predict_sim = []
    orient_predict_sim = []
    c_force_predict_sim = []

    # results of EKF update step
    pos_update_sim = []
    vel_update_sim = []
    ang_vel_update_sim = []
    orient_update_sim = []
    c_force_update_sim = []

    leg_odom_vel_sim = []
    c_force_obs_sim = []  # raw measurement fed into the filter (J^T tau estimate, unless ground truth contact_forces was given)

    kf = KF(dt=dt, Q_diag=Q, R_diag=R, contact_coupling=contact_coupling, contact_force_decay=contact_force_decay)
    kf.x[kf.POS] = init_pos
    leg_odom = LegOdom(model_name=model_name, init_state=np.concatenate([init_pos, np.zeros(3)]))

    for i in tqdm(range(num_data), desc="Running state estimation"):
        orient_est = kf.get_orient()  # filter's own current orientation estimate, used for kinematics

        leg_odom.compute_leg_odometry(dt=dt,
                                       base_orient=orient_est,
                                       base_ang_vel=base_ang_vel[i],
                                       qdot=joint_vel[i],
                                       joint_torque=joint_torque[i],
                                       joint_pos=joint_pos[i],
                                       contact_state=contact_states[i],
                                       contact_force=contact_forces[i],
                                       contact_state_threshold=contact_state_threshold)

        kf.update_A_B_contact_forces(leg_odom.env, orient_est, leg_odom.p_b, leg_odom.contact_states)

        kf.predict(u=np.concatenate([joint_acc[i], [1.0]]))
        kf.update(z=np.concatenate([quat_to_rot(base_orient[i]).flatten(),
                                     leg_odom.state.vel,
                                     base_ang_vel[i],
                                     leg_odom.contact_forces.flatten()]))

        pos_predict_sim.append(kf.get_pos("predict"))
        vel_predict_sim.append(kf.get_lin_vel("predict"))
        ang_vel_predict_sim.append(kf.get_ang_vel("predict"))
        orient_predict_sim.append(kf.get_orient("predict"))
        c_force_predict_sim.append(kf.get_contact_force("predict"))

        pos_update_sim.append(kf.get_pos())
        vel_update_sim.append(kf.get_lin_vel())
        ang_vel_update_sim.append(kf.get_ang_vel())
        orient_update_sim.append(kf.get_orient())
        c_force_update_sim.append(kf.get_contact_force())

        leg_odom_vel_sim.append(leg_odom.state.vel)
        c_force_obs_sim.append(leg_odom.contact_forces)

    result = {"pos_predict": np.array(pos_predict_sim),
              "vel_predict": np.array(vel_predict_sim),
              "ang_vel_predict": np.array(ang_vel_predict_sim),
              "orient_predict": np.array(orient_predict_sim),
              "c_force_predict": np.array(c_force_predict_sim),
              "pos_update": np.array(pos_update_sim),
              "vel_update": np.array(vel_update_sim),
              "ang_vel_update": np.array(ang_vel_update_sim),
              "orient_update": np.array(orient_update_sim),
              "c_force_update": np.array(c_force_update_sim),
              "leg_odom": np.array(leg_odom_vel_sim),
              "c_force_obs": np.array(c_force_obs_sim),
              }

    return result, kf


def run_state_estimation_jax(dt,
                              base_orient,
                              base_ang_vel,
                              joint_pos,
                              joint_vel,
                              joint_acc,
                              joint_torque,
                              contact_states,
                              Q,
                              R,
                              init_pos=None,
                              contact_forces=None,
                              contact_state_threshold=None,
                              contact_coupling="dynamics",
                              contact_force_decay="constant",
                              model_name="aliengo"):
    """JAX equivalent of run_state_estimation (see there for the argument docs).

    Uses KFJax/LegOdomJax instead of KF/LegOdom, so the filter math runs with
    jax.numpy and jax.jit. MuJoCo-bound parts (env stepping, mass matrix, jacobians)
    stay NumPy either way, since MuJoCo's classic Python bindings are not JAX-jittable.
    """

    num_data = len(base_orient)

    if init_pos is None: init_pos = np.zeros(3)
    if contact_forces is None: contact_forces = [None] * num_data

    # results of EKF prediction step
    pos_predict_sim = []
    vel_predict_sim = []
    ang_vel_predict_sim = []
    orient_predict_sim = []
    c_force_predict_sim = []

    # results of EKF update step
    pos_update_sim = []
    vel_update_sim = []
    ang_vel_update_sim = []
    orient_update_sim = []
    c_force_update_sim = []

    leg_odom_vel_sim = []
    c_force_obs_sim = []  # raw measurement fed into the filter (J^T tau estimate, unless ground truth contact_forces was given)

    kf = KFJax(dt=dt, Q_diag=Q, R_diag=R, contact_coupling=contact_coupling, contact_force_decay=contact_force_decay)
    kf.x = kf.x.at[kf.POS].set(jnp.asarray(init_pos))
    leg_odom = LegOdomJax(model_name=model_name, init_state=np.concatenate([init_pos, np.zeros(3)]))

    for i in tqdm(range(num_data), desc="Running state estimation (jax)"):
        orient_est = kf.get_orient()  # filter's own current orientation estimate, used for kinematics

        leg_odom.compute_leg_odometry(dt=dt,
                                       base_orient=orient_est,
                                       base_ang_vel=base_ang_vel[i],
                                       qdot=joint_vel[i],
                                       joint_torque=joint_torque[i],
                                       joint_pos=joint_pos[i],
                                       contact_state=contact_states[i],
                                       contact_force=contact_forces[i],
                                       contact_state_threshold=contact_state_threshold)

        kf.update_A_B_contact_forces(leg_odom.env, orient_est, leg_odom.p_b, leg_odom.contact_states)

        kf.predict(u=jnp.concatenate([jnp.asarray(joint_acc[i]), jnp.ones(1)]))
        kf.update(z=jnp.concatenate([quat_to_rot_jax(jnp.asarray(base_orient[i])).flatten(),
                                      leg_odom.state.vel,
                                      jnp.asarray(base_ang_vel[i]),
                                      leg_odom.contact_forces.flatten()]))

        pos_predict_sim.append(kf.get_pos("predict"))
        vel_predict_sim.append(kf.get_lin_vel("predict"))
        ang_vel_predict_sim.append(kf.get_ang_vel("predict"))
        orient_predict_sim.append(kf.get_orient("predict"))
        c_force_predict_sim.append(kf.get_contact_force("predict"))

        pos_update_sim.append(kf.get_pos())
        vel_update_sim.append(kf.get_lin_vel())
        ang_vel_update_sim.append(kf.get_ang_vel())
        orient_update_sim.append(kf.get_orient())
        c_force_update_sim.append(kf.get_contact_force())

        leg_odom_vel_sim.append(leg_odom.state.vel)
        c_force_obs_sim.append(leg_odom.contact_forces)

    result = {"pos_predict": np.array(pos_predict_sim),
              "vel_predict": np.array(vel_predict_sim),
              "ang_vel_predict": np.array(ang_vel_predict_sim),
              "orient_predict": np.array(orient_predict_sim),
              "c_force_predict": np.array(c_force_predict_sim),
              "pos_update": np.array(pos_update_sim),
              "vel_update": np.array(vel_update_sim),
              "ang_vel_update": np.array(ang_vel_update_sim),
              "orient_update": np.array(orient_update_sim),
              "c_force_update": np.array(c_force_update_sim),
              "leg_odom": np.array(leg_odom_vel_sim),
              "c_force_obs": np.array(c_force_obs_sim),
              }

    return result, kf
