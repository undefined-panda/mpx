from state_estimation.kalman_filter import KF, KF_JAX
from tqdm import tqdm
import numpy as np
from state_estimation.kf_utils import *
from state_estimation.dynamics import _get_base_acc, estimate_contact_forces, compute_contact_force_dynamics
from state_estimation.contact_obsever import GMContactObserver
from state_estimation.leg_odometry import LegOdom
from state_estimation.evaluate import validate_data
from state_estimation.logging_utils import print_section
import jax
import jax.numpy as jnp
from felan.models.cadelan_pot_param import CaDeLaN, get_config_from_dict
from felan.train import load_model_fn
import time

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
                   use_jax=False,
                   cadelan_path=None):

    validate_data(data)

    if base_acc_source not in ["external", "internal"]:
        raise ValueError(f"Invalid value for base_acc_source: {base_acc_source} (only 'internal' or 'external')")

    if (not include_contact_force) and (base_acc_source == "internal"):
        raise ValueError(f"Invalid parameter configuration:\ninclude_contact_force={include_contact_force} and base_acc_source={base_acc_source}")

    use_jax = use_jax or (cadelan_path is not None)
    xp = jnp if use_jax else np
    num_data = data["num_datapoints"]
    missing = []

    # --- Setup ---
    estimate_flags = {}
    leg_odom = LegOdom(model_name, xp=xp) # use leg odometry for linear velocity measurement
    
    # use generalized momentum observer for contact state estimation
    estimate_contact_state = False
    gm_observer = None
    if data.get("contact_states") is None:
        missing.append("contact_state - Estimated based on momentum")
        estimate_contact_state = True
        gm_observer = GMContactObserver(dt, L1, L2, contact_thresholds, xp=xp)
    estimate_flags["estimate_contact_state"] = estimate_contact_state

    # use relationship with joint torque for contact force estimation
    estimate_contact_force = False
    if data.get("contact_forces") is None:
        missing.append("contact_forces - Estimated with joint torque")
        estimate_contact_force = True
    estimate_flags["estimate_contact_force"] = estimate_contact_force

    # use Newton's second law of motion or rigid body dynamics for base acc estimation
    estimate_base_acc = False
    if data.get("base_acc") is None:
        estimate_base_acc = True
        base_acc_est_type = "Rigid-Body-Dynamics" if include_ang_vel else "Newton"
        missing.append(f"base_acc - Estimated with dynamics based on {base_acc_est_type} ({base_acc_source})")
        estimate_flags["base_acc_est_type"] = base_acc_est_type
    estimate_flags["estimate_base_acc"] = estimate_base_acc
    base_acc_source = base_acc_source if estimate_flags["estimate_base_acc"] else "external"

    if len(missing) > 0: print_section("Missing inputs - Estimating internally", missing)
    
    if not use_jax:
        return _run_estimation_numpy(dt, data, Q, R, include_ang_vel, include_contact_force, init_state, 
                                     base_acc_source, P0, estimate_flags, num_data, leg_odom, gm_observer)
    else:
        return _run_estimation_jax(dt, data, Q, R, include_ang_vel, include_contact_force, init_state, 
                                   base_acc_source, P0, estimate_flags, num_data, leg_odom, gm_observer, cadelan_path)

def _run_estimation_numpy(dt, data, Q, R, include_ang_vel, include_contact_force, init_state, 
                          base_acc_source, P0, estimate_flags, num_data, leg_odom, gm_observer):
    """Run state estimation using a Kalman Filter.

    Toggle addition of angular velocity to the state. If added, rigid body dynamics is used for 
    acceleration estimation

    Toggle addition of contact force to the state. Based on base_acc_source, contact force is estimated
    either ...
    """

    print("Running with Numpy")

    A, B, H = build_kinematic_model(dt, include_ang_vel, include_contact_force, base_acc_source)
    kf = KF(dt, A, B, H, Q, R, P0, init_state)

    prev_leg_odom_pos, prev_leg_odom_vel = leg_odom.init_pos, leg_odom.init_vel
    legs_order = leg_odom.env.legs_order
    mass = leg_odom.env.mjModel.body_mass.sum()
    
    if estimate_flags["estimate_contact_state"]:
        state = gm_observer.state

    pos_update, vel_update, ang_vel_update, contact_force_update = [], [], [], []
    leg_odom_vel, base_acc_est, contact_state_est, contact_force_est = [], [], [], []

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
        if not estimate_flags["estimate_contact_state"]:
            contact_state = np.asarray(data["contact_states"][i])
        else:
            J_w_stacked = J_w.reshape(12, -1)
            state, contact_state, f_hat = gm_observer.step(state=state,
                                                           vel=np.concatenate([kf.x[3:6], ang_vel, joint_vel]),
                                                           M=inertia_matrix,
                                                           joint_torque=joint_torque,
                                                           J=J_w_stacked,
                                                           qfrc_bias=qfrc_bias)
            contact_state_est.append(contact_state)

        # --- Contact Force ---
        if not estimate_flags["estimate_contact_force"]:
            contact_force = data["contact_forces"][i]
        else:
            contact_force = estimate_contact_forces(joint_torque=joint_torque, 
                                                    contact_states=contact_state, 
                                                    jacobians=J_w, 
                                                    legs_order=legs_order)
            contact_force_est.append(contact_force)

        # --- Leg Odometry ---
        new_leg_odom_pos, new_leg_odom_vel, contact_pos_b = leg_odom.motion_estimation(dt=dt, base_orient=orient_R, base_ang_vel=ang_vel, 
                                                                                       joint_pos=joint_pos, joint_vel=joint_vel, J_b=J_b, contact_state=contact_state, 
                                                                                       prev_pos=prev_leg_odom_pos, prev_vel=prev_leg_odom_vel)
        prev_leg_odom_pos, prev_leg_odom_vel = new_leg_odom_pos, new_leg_odom_vel

        # --- Dynamics ---
        if estimate_flags["estimate_base_acc"]:
            if base_acc_source == "external":
                base_acc = _get_base_acc(orient_R, joint_acc, contact_force, contact_state,
                                         contact_pos_b, mass, inertia_matrix, qfrc_bias,
                                         estimate_flags["base_acc_est_type"])
                base_acc_est.append(base_acc)
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
                        base_acc_source, P0, estimate_flags, num_data, leg_odom, gm_observer, cadelan_path):
    """Run state estimation using a Kalman Filter, written in JAX.
    """
    print("Running with JAX")
    jax.config.update("jax_compilation_cache_dir", "./jax_cache")
    t0 = time.perf_counter()

    # Kalman Filter
    A, B, H = build_kinematic_model(dt, include_ang_vel, include_contact_force, base_acc_source)
    A, B, H = map(jnp.asarray, (A, B, H))
    state_size = A.shape[0]
    kf = KF_JAX(dt, state_size, H, Q, R)
    x = jnp.zeros(state_size) if init_state is None else jnp.array(init_state)
    P = jnp.diag(P0) if P0 is not None else kf.Q.copy()

    # Leg Odometry
    prev_leg_odom_pos, prev_leg_odom_vel = leg_odom.init_pos, leg_odom.init_vel
    legs_order = leg_odom.env.legs_order
    mass = leg_odom.env.mjModel.body_mass.sum()
    mjx_model = leg_odom.mjx_model

    orient_quat, orient_R, orient_euler = [], [], []
    for orient in data["base_orient"]:
        quat, R = convert_orient(orient)
        euler = quat_to_euler(orient)
        orient_quat.append(quat)
        orient_R.append(R)
        orient_euler.append(euler)

    data_jax = {"orient_quat": jnp.asarray(orient_quat),
                "orient_R": jnp.asarray(orient_R),
                "orient_euler": jnp.asarray(orient_euler),
                "base_ang_vel": jnp.asarray(data["base_ang_vel"]),
                "joint_pos": jnp.asarray(data["joint_pos"]),
                "joint_vel": jnp.asarray(data["joint_vel"]),
                "joint_acc": jnp.asarray(data["joint_acc"]),
                "joint_torque": jnp.asarray(data["joint_torque"])}

    carry0 = {"x": x,
              "P": P,
              "A": A,
              "B": B,
              "prev_leg_odom_pos": prev_leg_odom_pos,
              "prev_leg_odom_vel": prev_leg_odom_vel}

    if not estimate_flags["estimate_base_acc"]: 
        data_jax["base_acc"] = jnp.asarray(data["base_acc"])
    if not estimate_flags["estimate_contact_force"]: 
        data_jax["contact_forces"] = jnp.asarray(data["contact_forces"])
    if estimate_flags["estimate_contact_state"]:
        state = gm_observer.state
        carry0["observer_state"] = state
    else:
        data_jax["contact_states"] = jnp.asarray(data["contact_states"])

    if cadelan_path is not None:
        # tau_diff is the difference between tau from the dataset and tau_nominal
        print("Loading CaDeLaN model")
        params, hyper = load_model_fn(cadelan_path.name, cadelan_path.parent)
        nn_config = get_config_from_dict(hyper)
        model = CaDeLaN(hyper['nv_dof'], nn_config)
        time_window = hyper["time_window"]
        history_stride = int(hyper.get("history_stride", 1))
        if hyper["history_input"] == "joint":
            feature_dim = 30
            init_history_vector = build_feature_vector(data["joint_pos"][0], data["joint_vel"][0], jnp.zeros(6))
        else:
            feature_dim = 16
            init_history_vector = build_feature_vector(data["base_orient"][0], data["base_vel"][0], data["base_ang_vel"][0], jnp.zeros(6))

        # initialize history as a buffer with copys of first value, fill it in each step
        carry0["history"] = jnp.tile(init_history_vector[None, :], (time_window, 1))
        carry0["hist_step"] = jnp.int32(0)
        carry0["base_acc"] = jnp.zeros(6)

        data_jax["tau"] = (data["tau_m"]+data["tau_c"]+data["tau_g"])[..., :6]
        data_jax["tau_nom"] = (data["tau_m_nom"]+data["tau_c_nom"]+data["tau_g_nom"])[..., :6]

    def step(carry, xs):
        ang_vel = xs["base_ang_vel"]
        orient_quat = xs["orient_quat"]
        orient_R = xs["orient_R"]
        orient_euler = xs["orient_euler"]
        joint_pos = xs["joint_pos"]
        joint_vel = xs["joint_vel"]
        joint_acc = xs["joint_acc"]
        joint_torque = xs["joint_torque"]

        # --- Robot Quantities ---
        J_b, J_w, mjx_data = get_jacobian_mjx(mjx_model, orient_quat, joint_pos, joint_vel, 
                                              carry["x"][3:6], ang_vel, leg_odom._foot_geom_ids_j, leg_odom._foot_body_ids_j)
        inertia_matrix = get_inertia_matrix_mjx(mjx_model, mjx_data)
        qfrc_bias = mjx_data.qfrc_bias

        # --- CaDeLaN ---
        if cadelan_path is not None:
            q = jnp.concatenate([carry["x"][0:3], orient_euler])[None, ...]
            qd = jnp.concatenate([carry["x"][3:6], ang_vel])[None, ...]
            qdd = carry["base_acc"][None, ...]
            history = carry["history"][None, ...]

            tau_diff_pred, _, extras = model.apply(params, q, qd, qdd, history)
            M_res = extras["M"][0]
            qfrc_res = extras["qfrc_bias"][0].reshape(-1)

            # use nominal inertia and bias forces
            inertia_matrix = inertia_matrix.at[:6, :6].add(M_res)
            qfrc_bias = qfrc_bias.at[:6].add(qfrc_res)            

        # --- Contact State ---
        if not estimate_flags["estimate_contact_state"]:
            contact_state = xs["contact_states"]
        else:
            J_w_stacked = J_w.reshape(12, -1)
            state, contact_state, f_hat = gm_observer.step(state=carry["observer_state"],
                                                           vel=jnp.concatenate([carry["x"][3:6], ang_vel, joint_vel]),
                                                           M=inertia_matrix,
                                                           joint_torque=joint_torque,
                                                           J=J_w_stacked,
                                                           qfrc_bias=qfrc_bias)

        # --- Contact Force ---
        if not estimate_flags["estimate_contact_force"]:
            contact_force = xs["contact_forces"]
        else:
            contact_force = estimate_contact_forces(joint_torque=joint_torque, 
                                                    contact_states=contact_state, 
                                                    jacobians=J_w, 
                                                    legs_order=legs_order,
                                                    xp=jnp)

        # --- Leg Odometry ---
        new_leg_odom_pos, new_leg_odom_vel, contact_pos_b = leg_odom.motion_estimation(dt=dt, base_orient=orient_R, base_ang_vel=ang_vel, 
                                                                                       joint_pos=joint_pos, joint_vel=joint_vel, J_b=J_b, contact_state=contact_state, 
                                                                                       prev_pos=carry["prev_leg_odom_pos"], prev_vel=carry["prev_leg_odom_vel"])
        prev_leg_odom_pos, prev_leg_odom_vel = new_leg_odom_pos, new_leg_odom_vel

        # --- Dynamics ---
        A, B = carry["A"], carry["B"]
        if estimate_flags["estimate_base_acc"]:
            if (base_acc_source == "external") or (cadelan_path is not None):
                base_acc = _get_base_acc(orient_R, joint_acc, contact_force, contact_state,
                                         contact_pos_b, mass, inertia_matrix, qfrc_bias,
                                         estimate_flags["base_acc_est_type"], xp=jnp)
                control_input = base_acc # gets overwritten for 'internal' + CaDeLaN case, to carry base_acc
            if base_acc_source == "internal":
                J_new, coupling, bias = compute_contact_force_dynamics(orient_R, contact_pos_b, contact_state, inertia_matrix, qfrc_bias, xp=jnp)
                A, B = kf.update_process_model(carry["A"], carry["B"], J_new, coupling, bias)
                control_input = jnp.concatenate([joint_acc, jnp.array([1.0])])
        else:
            base_acc = xs["base_acc"]
            control_input = base_acc

        # --- Measurements ---
        measurements = [new_leg_odom_vel]
        if include_ang_vel: measurements.append(ang_vel)
        if include_contact_force: measurements.append(contact_force.flatten())

        # --- Kalman Filter ---
        x_pred, P_pred = kf.predict(x=carry["x"], A=A, u=control_input, B=B, P=carry["P"])
        x, P = kf.update(x_pred=x_pred, P_pred=P_pred, z=jnp.hstack(measurements))

        new_carry = {"x": x,
                     "P": P,
                     "A": A,
                     "B": B,
                     "prev_leg_odom_pos": prev_leg_odom_pos,
                     "prev_leg_odom_vel": prev_leg_odom_vel}
        if estimate_flags["estimate_contact_state"]: new_carry["observer_state"] = state
        
        ys = {"pos_update": x[0:3],
              "vel_update": x[3:6],
              "leg_odom_vel": new_leg_odom_vel
              }
        
        if cadelan_path is not None:
            tau_diff = xs["tau"] - xs["tau_nom"]
            feature = build_feature(feature_dim, joint_pos, joint_vel, tau_diff, orient_quat, x[3:6], ang_vel)
            new_carry["history"] = update_history(carry["history"], feature, carry["hist_step"], history_stride)
            new_carry["hist_step"] = carry["hist_step"] + 1
            ys["tau_diff_pred"] = tau_diff_pred
            new_carry["base_acc"] = base_acc

        # --- Log Results ---
        if include_ang_vel: ys["ang_vel_update"] = x[6:9]
        if include_contact_force: ys["contact_force_update"] = x[9:21]
        if estimate_flags["estimate_base_acc"] and ((base_acc_source == "external") or (cadelan_path is not None)): ys["base_acc_est"] = base_acc
        if estimate_flags["estimate_contact_force"]: ys["contact_force_est"] = contact_force
        if estimate_flags["estimate_contact_state"]: ys["contact_state_est"] = contact_state

        return new_carry, ys
    
    print(f"Preparation setup finished after {time.perf_counter() - t0:.2f}s")

    _, result = jax.lax.scan(step, carry0, data_jax, length=num_data)
    print(f"JAX Lax Scan finished after {time.perf_counter() - t0:.2f}s")

    result["legs_order"] = legs_order
    
    return result
