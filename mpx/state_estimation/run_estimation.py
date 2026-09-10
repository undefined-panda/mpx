from state_estimation.kalman_filter import KF
from tqdm import tqdm
import numpy as np
from state_estimation.kf_utils import get_jacobian, get_inertia_matrix, convert_orient, build_kinematic_model
from state_estimation.dynamics import _get_base_acc, estimate_contact_forces, compute_contact_force_dynamics
from state_estimation.contact_obsever import GMContactObserver
from state_estimation.leg_odometry import LegOdom
from state_estimation.evaluate import validate_data
from state_estimation.logging_utils import print_section

def run_estimation(dt, 
                   data, 
                   Q, 
                   R,
                   include_ang_vel=True,
                   include_contact_force=True,
                   init_state=None,
                   base_acc_source="internal",
                   P0=None,
                   model_name="aliengo", 
                   L1=100, 
                   L2=10000, 
                   contact_thresholds=[15,15,15,15],
                   cadelan_path=None):

    validate_data(data)

    if base_acc_source not in ["external", "internal"]:
        raise ValueError(f"Invalid value for base_acc_source: {base_acc_source} (only 'internal' or 'external')")

    if (not include_contact_force) and (base_acc_source == "internal"):
        raise ValueError(f"Invalid parameter configuration:\ninclude_contact_force={include_contact_force} and base_acc_source={base_acc_source}")
    
    if cadelan_path is None:
        return _run_estimation_numpy(dt, data, Q, R, include_ang_vel, include_contact_force, init_state, 
                                     base_acc_source, P0, model_name, L1, L2, contact_thresholds)
    else:
        return _run_estimation_jax(dt, data, Q, R, include_ang_vel, include_contact_force, init_state, 
                                   base_acc_source, P0, model_name, L1, L2, contact_thresholds, cadelan_path)

def _run_estimation_numpy(dt, data, Q, R, include_ang_vel, include_contact_force, init_state, 
                          base_acc_source, P0, model_name, L1, L2, contact_thresholds):
    """Run state estimation using a Kalman Filter.

    Toggle addition of angular velocity to the state. If added, rigid body dynamics is used for 
    acceleration estimation

    Toggle addition of contact force to the state. Based on base_acc_source, contact force is estimated
    either ...
    """

    mass_est = False
    num_data = data["num_datapoints"]
    missing = []

    # use leg odometry for linear velocity measurement
    leg_odom = LegOdom(model_name)
    prev_leg_odom_pos, pre_leg_odom_vel = np.zeros(3), np.zeros(3)
    legs_order = leg_odom.env.legs_order
    mass = leg_odom.env.mjModel.body_mass.sum()

    # use generalized momentum observer for contact state estimation
    estimate_contact_state = False
    if data.get("contact_states") is None:
        missing.append("contact_state - Estimated based on momentum")
        estimate_contact_state = True
        gm_observer = GMContactObserver(dt, L1, L2, contact_thresholds)
        gm_observer.f_hat_history = []

    # use relationship with joint torque for contact force estimation
    estimate_contact_force = False
    if data.get("contact_forces") is None:
        missing.append("contact_forces - Estimated with joint torque")
        estimate_contact_force = True

    # use Newton's second law of motion or rigid body dynamics for base acc estimation
    estimate_base_acc = False
    if data.get("base_acc") is None:
        estimate_base_acc = True
        base_acc_est_type = "Rigid-Body-Dynamics" if include_ang_vel else "Newton"
        missing.append(f"base_acc - Estimated with dynamics based on {base_acc_est_type} ({base_acc_source})")

    base_acc_source = base_acc_source if estimate_base_acc else "external"
    A, B, H = build_kinematic_model(dt, include_ang_vel, include_contact_force, base_acc_source)
    kf = KF(dt, A, B, H, Q, R, P0, init_state)

    pos_update, vel_update, ang_vel_update, contact_force_update = [], [], [], []
    leg_odom_vel, base_acc_est, contact_state_est, contact_force_est = [], [], [], []

    if len(missing) > 0: print_section("Missing inputs - Estimating internally", missing)

    for i in tqdm(range(num_data), desc="Running state estimation"):
        # -- Unpack Inputs ---
        orient_quat, orient_R = convert_orient(data["base_orient"][i])
        ang_vel = data["base_ang_vel"][i]
        joint_pos = data["joint_pos"][i]
        joint_vel = data["joint_vel"][i]
        joint_acc = data["joint_acc"][i]
        joint_torque = data["joint_torque"][i]

        # --- Robot Quantities ---
        J_b, J_w = get_jacobian(leg_odom.env, orient_quat, joint_pos, joint_vel, kf.x[3:6], ang_vel)
        inertia_matrix = get_inertia_matrix(leg_odom.env)
        qfrc_bias = leg_odom.env.mjData.qfrc_bias.copy()

        # --- Contact State ---
        if not estimate_contact_state:
            contact_state = np.asarray(data["contact_states"][i])
        else:
            J_w_stacked = np.stack([J_w[leg] for leg in legs_order]).reshape(12, -1)
            contact_state, f_hat = gm_observer.step(vel=np.concatenate([kf.x[3:6], ang_vel, joint_vel]),
                                                    M=inertia_matrix,
                                                    joint_torque=joint_torque,
                                                    J=J_w_stacked,
                                                    qfrc_bias=qfrc_bias)
            contact_state_est.append(contact_state)
            gm_observer.f_hat_history.append(f_hat)

        # --- Contact Force ---
        if not estimate_contact_force:
            contact_force = data["contact_forces"][i]
        else:
            contact_force = estimate_contact_forces(joint_torque=joint_torque, 
                                                    contact_states=contact_state, 
                                                    jacobians=J_w, 
                                                    legs_order=legs_order)
            contact_force_est.append(contact_force)

        # --- Leg Odometry ---
        new_leg_odom_pos, new_leg_odom_vel = leg_odom.motion_estimation(dt=dt, base_orient=orient_R, base_ang_vel=ang_vel, 
                                                                        joint_pos=joint_pos, joint_vel=joint_vel, J_b=J_b, contact_state=contact_state, 
                                                                        prev_pos=prev_leg_odom_pos, prev_vel=pre_leg_odom_vel)
        contact_pos_b = leg_odom.p_b
        prev_leg_odom_pos, pre_leg_odom_vel = new_leg_odom_pos, new_leg_odom_vel

        # --- Dynamics ---
        if estimate_base_acc:
            if base_acc_source == "external":
                base_acc = _get_base_acc(orient_R, joint_acc, contact_force, contact_state,
                                         contact_pos_b, mass, inertia_matrix, qfrc_bias,
                                         base_acc_est, base_acc_est_type, mass_est)
                control_input = base_acc
            else: # internal
                J_new, coupling, bias = compute_contact_force_dynamics(orient_R, contact_pos_b, contact_state, inertia_matrix, qfrc_bias)
                kf.update_process_model(J_new, coupling, bias)
                control_input = np.concatenate([joint_acc, [1.0]])
        else:
            base_acc = data["base_acc"][i]
            control_input = base_acc

        # --- Measurements ---
        measurements = [new_leg_odom_vel]
        if include_ang_vel: measurements.append(ang_vel)
        if include_contact_force: measurements.append(contact_force.flatten())

        # --- Kalman Filter ---
        kf.predict(u=control_input)
        kf.update(z=np.hstack(measurements))

        # --- Log Results ---
        leg_odom_vel.append(new_leg_odom_vel)
        pos_update.append(kf.x[0:3])
        vel_update.append(kf.x[3:6])
        if include_ang_vel: ang_vel_update.append(kf.x[6:9])
        if include_contact_force: contact_force_update.append(kf.x[9:21])

    result = {"pos_update": np.array(pos_update),
              "vel_update": np.array(vel_update),
              "ang_vel_update": np.array(ang_vel_update),
              "contact_force_update": np.array(contact_force_update),
              "leg_odom_vel": np.array(leg_odom_vel),
              "base_acc_est": np.array(base_acc_est),
              "contact_state_est": np.array(contact_state_est),
              "contact_force_est": np.array(contact_force_est),
              "legs_order": legs_order,
              }

    return result

def _run_estimation_jax(dt, data, Q, R, include_ang_vel, include_contact_force, init_state, 
                        base_acc_source, P0, model_name, L1, L2, contact_thresholds, cadelan_path):
    """Run state estimation using a Kalman Filter, written in JAX.
    """

    pass
