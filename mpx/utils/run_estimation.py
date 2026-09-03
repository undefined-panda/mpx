import numpy as np
from utils.leg_odometry import LegOdom, LegOdom_JAX, compute_leg_odometry_step_jax
from tqdm import tqdm
from utils.state_estimation import KF, KF_JAX
from utils.kf_utils import quat_to_rot, get_inertia_matrix, get_jacobian, quat_to_euler
from utils.dynamics_model import GMContactObserver, GMContactObserver_JAX, estimate_contact_states, estimate_contact_forces_v2, estimate_acc_from_contact_force_v4
from felan.models.log_chol_cadelac_pot_param import CaDeLaCLogChol, get_config_from_dict
from felan.train import load_model_fn
import jax
import jax.numpy as jnp

def run_state_estimation(dt,
                         base_orient,
                         base_ang_vel,
                         joint_pos,
                         joint_vel,
                         joint_acc,
                         Q,
                         R,
                         base_acc=None,
                         base_vel=None,
                         base_pos=None,
                         joint_torque=None,
                         contact_forces=None,
                         contact_states=None,
                         contact_state_threshold=None,
                         model_name="aliengo",
                         L1=100,
                         L2=10000, # based on paper: L2 = L1²
                         contact_thresholds=[15,15,15,15],
                         est_mode=1,
                         cadelac_path=None,
                         tau=None,
                         tau_nominal=None,
                         oracle_M_res=None,
                         oracle_qfrc_res=None):
    """Run the Kalman Filter state estimation.

    est_mode:
    - 1 = estimation of pos, lin vel and ang vel (base acc as control i xput)
    - 2 = estimation of pos, lin vel, ang vel and contact force (identity in A)
    - 3 = estimation of pos, lin vel, ang vel and contact force (contact state diagonal block matrix in A)
    - 4 = estimation of pos, lin vel, ang vel and contact force (identity in A) with base acc inside KF

    If CaDeLaC path is given, it runs with JAX, else Numpy.

    oracle_M_res / oracle_qfrc_res (nur ohne cadelac_path): wahre Residuen statt eines
    gelernten Modells -- oracle_M_res (6,6) wird pro Schritt auf den Basisblock der
    nominalen Traegheitsmatrix addiert, oracle_qfrc_res (N,6) auf qfrc_bias[:6].
    Gleiche Stellen wie die CaDeLaC-Korrektur im JAX-Pfad (GM-Observer, Beschleunigungs-
    schaetzung, KF-Update), damit die Leiter Nominal / CaDeLaC / Oracle fair vergleicht.
    Das ist die Obergrenze dessen, was ein perfektes Netz dem Filter bringen kann.
    """
    if cadelac_path is None:
        return _run_state_estimation_numpy(
            dt=dt, base_orient=base_orient, base_ang_vel=base_ang_vel, joint_pos=joint_pos,
            joint_vel=joint_vel, joint_acc=joint_acc, Q=Q, R=R, base_acc=base_acc, joint_torque=joint_torque, 
            contact_forces=contact_forces, contact_states=contact_states, contact_state_threshold=contact_state_threshold,
            model_name=model_name, L1=L1, L2=L2, contact_thresholds=contact_thresholds, est_mode=est_mode,
            oracle_M_res=oracle_M_res, oracle_qfrc_res=oracle_qfrc_res)

    if oracle_M_res is not None or oracle_qfrc_res is not None:
        raise ValueError("oracle_M_res/oracle_qfrc_res nur ohne cadelac_path verwenden (entweder Netz oder Oracle).")

    return _run_state_estimation_cadelac(
        dt=dt, base_orient=base_orient, base_ang_vel=base_ang_vel, joint_pos=joint_pos,
        joint_vel=joint_vel, joint_acc=joint_acc, Q=Q, R=R, base_acc=base_acc, base_vel=base_vel,
        base_pos=base_pos, joint_torque=joint_torque, contact_forces=contact_forces,
        contact_states=contact_states, contact_state_threshold=contact_state_threshold,
        model_name=model_name, L1=L1, L2=L2, contact_thresholds=contact_thresholds, est_mode=est_mode,
        cadelac_path=cadelac_path, tau=tau, tau_nominal=tau_nominal)


def _run_state_estimation_numpy(dt, base_orient, base_ang_vel, joint_pos, joint_vel, joint_acc, Q, R,
                                 base_acc, joint_torque, contact_forces,
                                 contact_states, contact_state_threshold, model_name, L1, L2,
                                 contact_thresholds, est_mode,
                                 oracle_M_res=None, oracle_qfrc_res=None):
    num_data = len(base_ang_vel)

    pos_predict_sim, vel_predict_sim, ang_vel_predict_sim, c_force_predict_sim = [], [], [], []
    pos_update_sim, vel_update_sim, ang_vel_update_sim, c_force_update_sim = [], [], [], []
    c_force_measurement, c_state_estimation, leg_odom_vel = [], [], []

    kf = KF(dt=dt, Q_diag=Q, R_diag=R, est_mode=est_mode)
    leg_odom = LegOdom(model_name=model_name)
    gm_observer = GMContactObserver(dt, L1, L2, contact_thresholds)
    gm_observer.f_hat_history = []

    single_force_val = (contact_forces is not None) and (contact_forces[0].shape == (4,))
    base_acc_info = c_force_info = c_state_info = True
    base_acc_i = np.zeros((6,))

    for i in tqdm(range(num_data), desc="Running state estimation"):
        orient = base_orient[i]
        orient_rot = quat_to_rot(orient, enable_jax=False)
        J_b, J_w = get_jacobian(leg_odom.env, orient, joint_pos[i], joint_vel[i])

        inertia_matrix = get_inertia_matrix(leg_odom.env)
        qfrc_bias = leg_odom.env.mjData.qfrc_bias.copy()

        # Oracle-Residuen: an derselben Stelle addiert wie die CaDeLaC-Korrektur im
        # JAX-Pfad, VOR allen drei Verwendungen (GM-Observer, Acc-Schaetzung, KF).
        if oracle_M_res is not None:
            inertia_matrix[:6, :6] += oracle_M_res
        if oracle_qfrc_res is not None:
            qfrc_bias[:6] += oracle_qfrc_res[i]

        # --- Contact State ---
        if contact_states is None:
            if contact_forces is not None:
                if single_force_val:
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
                J_w_stacked = np.stack([J_w[leg] for leg in leg_odom.env.legs_order])
                c_state, f_hat = gm_observer.step(vel=np.concatenate([kf.get_lin_vel(), base_ang_vel[i], joint_vel[i]]),
                                                  M=inertia_matrix,
                                                  joint_torque=joint_torque[i],
                                                  J=J_w_stacked.reshape(12, -1),
                                                  qfrc_bias=qfrc_bias)
                c_state_estimation.append(c_state)
                gm_observer.f_hat_history.append(f_hat)
        else:
            c_state = np.asarray(contact_states[i])

        # --- Leg Odometry ---
        leg_odom.compute_leg_odometry(dt=dt,
                                      base_orient=orient,
                                      base_ang_vel=base_ang_vel[i],
                                      qdot=joint_vel[i],
                                      joint_pos=joint_pos[i],
                                      J_b=J_b,
                                      contact_state=c_state)

        # --- Contact Force ---
        if contact_forces is None or single_force_val:
            if c_force_info:
                print("Estimating contact forces with joint torque")
                c_force_info = False
            J_w_stacked = np.stack([J_w[leg] for leg in leg_odom.env.legs_order])
            c_force = estimate_contact_forces_v2(joint_torque=joint_torque[i],
                                              contact_state=c_state,
                                              J_w_stacked=J_w_stacked,
                                              enable_jax=False)
            c_force_measurement.append(c_force)
        else:
            c_force = contact_forces[i]

        # --- Base Accelaration ---
        if base_acc is None:
            if base_acc_info:
                print("Estimating base accelaration with dynamics")
                base_acc_info = False
            base_acc_i = estimate_acc_from_contact_force_v4(joint_acc=joint_acc[i],
                                                            contact_forces=c_force,
                                                            contact_states=c_state,
                                                            contact_pos_b=leg_odom.p_b,
                                                            orient=orient_rot,
                                                            M=inertia_matrix,
                                                            qfrc_bias=qfrc_bias,
                                                            enable_jax=False)
        else:
            base_acc_i = np.asarray(base_acc[i])

        # --- Kalman Filter ---
        if est_mode == 1:
            kf.predict(u=base_acc_i)
            kf.update(z=np.concatenate([leg_odom.state.vel, base_ang_vel[i]]))
        elif est_mode in [2,3]:
            if est_mode == 3:
                kf.update_A_contact_force(contact_state=c_state)
            kf.predict(u=base_acc_i)
            kf.update(z=np.concatenate([leg_odom.state.vel, base_ang_vel[i], c_force.flatten()]))
        elif est_mode == 4:
            kf.update_A_B_contact_forces(orient_rot, leg_odom.p_b, c_state, inertia_matrix, qfrc_bias)
            kf.predict(u=np.concatenate([np.asarray(joint_acc[i]), np.array([1.0])]))
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
              "leg_odom_vel": np.array(leg_odom_vel),
              }

    return result

def _precompute_mujoco_stage(leg_odom, base_orient, joint_pos, joint_vel):
    """Precompute every per-step, MuJoCo-derived quantity for the whole trajectory in
    one fast, plain-numpy pass. These quantities (feet jacobians, foot positions, full
    mass matrix, bias forces) are pure functions of the externally given trajectory
    (base_orient[i], joint_pos[i], joint_vel[i]) -- NOT of the recursive filter state --
    so they can be computed once upfront instead of interleaved with the recursive
    filter, which is where nearly all of the CaDeLaC-path overhead came from (per-step
    eager JAX dispatch + host/device transfers around each MuJoCo call). Same MuJoCo
    call pattern/order as the original per-iteration loop, just hoisted into a batch.
    """
    num_data = len(base_orient)
    legs = leg_odom.env.legs_order
    nv = leg_odom.env.mjModel.nv

    J_b = np.zeros((num_data, 4, 3, nv))
    J_w = np.zeros((num_data, 4, 3, nv))
    p_b = np.zeros((num_data, 4, 3))
    M = np.zeros((num_data, nv, nv))
    qfrc_bias = np.zeros((num_data, nv))

    for i in range(num_data):
        J_b_i, J_w_i = get_jacobian(leg_odom.env, base_orient[i], joint_pos[i], joint_vel[i])
        for k, leg in enumerate(legs):
            J_b[i, k] = J_b_i[leg]
            J_w[i, k] = J_w_i[leg]

        M[i] = get_inertia_matrix(leg_odom.env)
        qfrc_bias[i] = leg_odom.env.mjData.qfrc_bias.copy()

        p_b[i] = leg_odom.compute_foot_positions_B(joint_pos[i])

    return {"J_b": J_b, "J_w": J_w, "p_b": p_b, "M": M, "qfrc_bias": qfrc_bias}

def _run_state_estimation_cadelac(dt, base_orient, base_ang_vel, joint_pos, joint_vel, joint_acc, Q, R,
                                   base_acc, base_vel, base_pos, joint_torque, contact_forces,
                                   contact_states, contact_state_threshold, model_name, L1, L2,
                                   contact_thresholds, est_mode, cadelac_path, tau, tau_nominal):
    num_data = len(base_ang_vel)
    kf_ref = KF_JAX(dt=dt, Q_diag=Q, R_diag=R, est_mode=est_mode)
    POS, LIN_VEL, ANG_VEL, C_FORCE = kf_ref.POS, kf_ref.LIN_VEL, kf_ref.ANG_VEL, kf_ref.C_FORCE
    Q_mat, R_mat, H_mat = kf_ref.Q, kf_ref.R, kf_ref.H

    leg_odom = LegOdom_JAX(model_name=model_name)
    gm_observer = GMContactObserver_JAX(dt, L1, L2, contact_thresholds)

    single_force_val = (contact_forces is not None) and (contact_forces[0].shape == (4,))
    if contact_states is None and contact_forces is not None and not single_force_val:
        raise ValueError("Error for contact state estimation.")
    use_gm_observer = (contact_states is None) and (contact_forces is None)
    use_threshold = (contact_states is None) and (contact_forces is not None) and single_force_val
    use_external_base_acc = base_acc is not None
    use_external_c_force = (contact_forces is not None) and (not single_force_val)

    if use_gm_observer:
        print("Estimating contact state based on momentum")
    elif use_threshold:
        print("Estimating contact state based on threshold")
    if not use_external_c_force:
        print("Estimating contact forces with joint torque")
    if not use_external_base_acc:
        print("Estimating base accelaration with dynamics")

    print("Loading CaDeLaC model")
    params, hyper = load_model_fn(cadelac_path.name, cadelac_path.parent)
    nn_config = get_config_from_dict(hyper)
    model = CaDeLaCLogChol(hyper['nv_dof'], nn_config)
    time_window = hyper["time_window"]
    # The training windows are strided: `time_window` points spaced `history_stride`
    # samples apart, i.e. a span of time_window*history_stride samples. The ring
    # buffer below must be advanced at that same rate -- pushing every step would
    # give the model a window covering only 1/stride of the wall-clock span it was
    # trained on, which is a different input distribution. Models trained before
    # this option existed carry no key and default to the old every-step behaviour.
    history_stride = int(hyper.get("history_stride", 1))

    feature_dim = int(params["params"]["lstm"]["lstm_layer_0"]["ii"]["kernel"].shape[0])

    # ---- Stage A: MuJoCo precompute over the FULL trajectory (fast, plain numpy) ----
    stage_a = _precompute_mujoco_stage(leg_odom, np.asarray(base_orient), np.asarray(joint_pos), np.asarray(joint_vel)) # muss umgebaut werden, sodass nix precomputed ist

    # ---- Cast trajectory + Stage A outputs to device once (not per-iteration) ----
    base_orient_j = jnp.asarray(base_orient)
    base_ang_vel_j = jnp.asarray(base_ang_vel)
    joint_pos_j = jnp.asarray(joint_pos)
    joint_vel_j = jnp.asarray(joint_vel)
    joint_acc_j = jnp.asarray(joint_acc)
    use_external_base_vel = base_vel is not None # if not given, use KF prediction
    use_external_base_pos = base_pos is not None # if not given, use KF prediction
    base_vel_j = jnp.asarray(base_vel) if use_external_base_vel else jnp.zeros((num_data, 3))
    base_pos_j = jnp.asarray(base_pos) if use_external_base_pos else jnp.zeros((num_data, 3))
    joint_torque_j = jnp.asarray(joint_torque) if joint_torque is not None else None
    contact_forces_j = jnp.asarray(contact_forces) if contact_forces is not None else None
    contact_states_j = jnp.asarray(contact_states) if contact_states is not None else None
    base_acc_j = jnp.asarray(base_acc) if base_acc is not None else None
    tau_j = jnp.asarray(tau)

    J_b_j = jnp.asarray(stage_a["J_b"])
    J_w_j = jnp.asarray(stage_a["J_w"])
    p_b_j = jnp.asarray(stage_a["p_b"])
    M_mujoco_j = jnp.asarray(stage_a["M"])
    qfrc_bias_mujoco_j = jnp.asarray(stage_a["qfrc_bias"])
    tau_nominal_mujoco_j = jnp.asarray(tau_nominal)

    def shift1(arr):
        # shift array by one position and use the first one for index 0 and 1, to get previous values for history
        return jnp.concatenate([arr[0:1], arr[:-1]], axis=0)

    base_orient_prev = shift1(base_orient_j)
    base_vel_prev = shift1(base_vel_j)
    base_ang_vel_prev = shift1(base_ang_vel_j)
    base_pos_z_prev = shift1(base_pos_j[:, 2])
    joint_pos_prev_j = shift1(joint_pos_j)
    joint_vel_prev_j = shift1(joint_vel_j)

    def build_feature_vector(*args):
        return jnp.concatenate([jnp.atleast_1d(jnp.asarray(a)) for a in args])

    # tau_diff is the difference between tau from the dataset and tau_nominal
    def build_feature(base_orient_prev_i, base_vel_prev_i, base_ang_vel_prev_i, base_pos_z_prev_i,
                       joint_pos_prev_i, joint_vel_prev_i, tau_diff_i):
        if feature_dim == 30:  # input_values="joint"
            return build_feature_vector(joint_pos_prev_i, joint_vel_prev_i, tau_diff_i)
        elif feature_dim == 17:  # input_values="base_pos_z"
            return build_feature_vector(base_orient_prev_i, base_vel_prev_i, base_ang_vel_prev_i,
                                         base_pos_z_prev_i, tau_diff_i)
        elif feature_dim == 16:  # input_values="base"
            return build_feature_vector(base_orient_prev_i, base_vel_prev_i, base_ang_vel_prev_i, tau_diff_i)
        else:
            raise ValueError(f"Invalid feature dim: {feature_dim}")

    # ---- Build the xs (scanned-input) pytree, covering the FULL trajectory ----
    xs = {
        "base_orient": base_orient_j,
        "base_orient_prev": base_orient_prev,
        "base_ang_vel": base_ang_vel_j,
        "base_ang_vel_prev": base_ang_vel_prev,
        "joint_pos": joint_pos_j,
        "joint_pos_prev_j": joint_pos_prev_j,
        "joint_vel": joint_vel_j,
        "joint_vel_prev_j": joint_vel_prev_j,
        "joint_acc": joint_acc_j,
        "tau": tau_j,
        "J_b": J_b_j,
        "J_w": J_w_j,
        "p_b": p_b_j,
        "M_mujoco": M_mujoco_j,
        "qfrc_bias_mujoco": qfrc_bias_mujoco_j,
        "tau_nominal_mujoco": tau_nominal_mujoco_j
    }
    if joint_torque_j is not None:
        xs["joint_torque"] = joint_torque_j
    if contact_states_j is not None:
        xs["contact_states"] = contact_states_j
    if contact_forces_j is not None:
        xs["contact_forces"] = contact_forces_j
    if base_acc_j is not None:
        xs["base_acc"] = base_acc_j
    if use_external_base_vel:
        xs["base_vel_prev"] = base_vel_prev
    if use_external_base_pos:
        xs["base_pos_z_prev"] = base_pos_z_prev

    # make CaDeLaC usable from step 0 with 'time_window' copies of the first feature vector, gradually changed with new values
    feature_0 = build_feature(base_orient_prev[0],
                               base_vel_prev[0] if use_external_base_vel else jnp.zeros(3),
                               base_ang_vel_prev[0],
                               base_pos_z_prev[0] if use_external_base_pos else jnp.zeros(()),
                               joint_pos_prev_j[0], joint_vel_prev_j[0], jnp.zeros(6))

    x_size = kf_ref.x.shape[0]

    # recursive state passed through the scan
    carry0 = {
        "x": jnp.zeros(x_size),
        "P": Q_mat,
        "A": kf_ref.A,
        "B": kf_ref.B,
        "leg_pos": jnp.zeros(3),
        "leg_vel": jnp.zeros(3),
        "history_j": jnp.tile(feature_0[None, :], (time_window, 1)),
        "tau_diff_prev": jnp.zeros(6),
        "hist_step": jnp.int32(0),
    }

    if not use_external_base_acc:
        carry0["base_acc_i"] = jnp.zeros(6)
    if use_gm_observer:
        carry0["p_hat"] = jnp.zeros(18)
        carry0["f_hat"] = jnp.zeros(12)
        carry0["f_hat_prev"] = jnp.zeros(12)

    def step(carry, xs_i):
        orient = xs_i["base_orient"]
        orient_rot = quat_to_rot(orient, enable_jax=True)

        # --- CaDeLaC feature-history buffer ---
        base_vel_prev_i = xs_i["base_vel_prev"] if use_external_base_vel else carry["x"][LIN_VEL]
        base_pos_z_prev_i = xs_i["base_pos_z_prev"] if use_external_base_pos else carry["x"][POS][2]
        feature = build_feature(xs_i["base_orient_prev"], base_vel_prev_i, xs_i["base_ang_vel_prev"],
                                 base_pos_z_prev_i, xs_i["joint_pos_prev_j"], xs_i["joint_vel_prev_j"],
                                 carry["tau_diff_prev"])
        # Advance only every `history_stride`-th step so the buffer spans the same
        # wall-clock window as the training data (jnp.where, not a Python branch --
        # this runs inside lax.scan).
        rolled_history_j = jnp.roll(carry["history_j"], -1, axis=0).at[-1].set(feature)
        do_push = (carry["hist_step"] % history_stride) == 0
        new_history_j = jnp.where(do_push, rolled_history_j, carry["history_j"])
        new_hist_step = carry["hist_step"] + 1

        # nominal inertia matrix, biases and torque from MuJoCo
        M_nom = xs_i["M_mujoco"]
        qfrc_nom = xs_i["qfrc_bias_mujoco"]
        tau_nom = xs_i["tau_nominal_mujoco"]
        tau_i = xs_i["tau"]

        q_in = jnp.concatenate([carry["x"][POS], quat_to_euler(orient, True)])[None, ...]
        qd_in = jnp.concatenate([carry["x"][LIN_VEL], xs_i["base_ang_vel"]])[None, ...]
        qdd_src = carry["base_acc_i"] if not use_external_base_acc else xs_i["base_acc"]
        qdd_in = qdd_src[None, ...]
        history_in = new_history_j[None, ...]

        # residual predictions
        tau_diff_pred, _, extras = model.apply(params, q_in, qd_in, qdd_in, history_in)
        M_res = extras["M"][0]
        qfrc_res = extras["qfrc_bias"][0].reshape(-1)

        # true_value = residual + nominal
        inertia_matrix = M_nom.at[:6, :6].add(M_res)
        qfrc_bias = qfrc_nom.at[:6].add(qfrc_res)

        # --- Contact state ---
        f_hat_out = None
        p_hat_new = f_hat_new = f_filtered = None
        if not use_gm_observer and not use_threshold:
            c_state = xs_i["contact_states"]
        elif use_gm_observer:
            J_w_flat = xs_i["J_w"].reshape(12, -1)
            p_hat_new, f_hat_new, f_filtered, c_state = gm_observer._step_impl(
                carry["p_hat"], carry["f_hat"], carry["f_hat_prev"],
                vel=jnp.concatenate([carry["x"][LIN_VEL], xs_i["base_ang_vel"], xs_i["joint_vel"]]),
                M=inertia_matrix, joint_torque=xs_i["joint_torque"], J=J_w_flat, qfrc_bias=qfrc_bias)
            f_hat_out = f_filtered
        else:  # use_threshold
            c_state = estimate_contact_states(xs_i["contact_forces"], contact_state_threshold)

        # --- Leg odometry ---
        new_leg_pos, new_leg_vel = compute_leg_odometry_step_jax(
            orient_rot, xs_i["base_ang_vel"], xs_i["joint_vel"], xs_i["p_b"],
            xs_i["J_b"], c_state, dt, carry["leg_pos"])

        # --- Contact force ---
        if use_external_c_force:
            c_force = xs_i["contact_forces"]
        else:
            c_force = estimate_contact_forces_v2(xs_i["joint_torque"], c_state, xs_i["J_w"], enable_jax=True)

        # --- Base acceleration ---
        if use_external_base_acc:
            base_acc_i = xs_i["base_acc"]
        else:
            base_acc_i = estimate_acc_from_contact_force_v4(
                xs_i["joint_acc"], c_force, c_state, xs_i["p_b"], orient_rot,
                inertia_matrix, qfrc_bias, enable_jax=True)

        # --- Kalman filter ---
        A, B = carry["A"], carry["B"]
        if est_mode == 3:
            A = KF_JAX._update_A_cf_impl(A, c_state)
        if est_mode == 4:
            A, B = KF_JAX._update_AB_impl(dt, A, B, orient_rot, xs_i["p_b"], c_state,
                                          inertia_matrix, qfrc_bias, use_full_M=True)
            u = jnp.concatenate([xs_i["joint_acc"], jnp.array([1.0])])
        else:
            u = base_acc_i
        x_pred, P_pred = KF_JAX._predict_impl(A, B, Q_mat, carry["P"], carry["x"], u)
        if est_mode == 1:
            z = jnp.concatenate([new_leg_vel, xs_i["base_ang_vel"]])
        else:
            z = jnp.concatenate([new_leg_vel, xs_i["base_ang_vel"], c_force.flatten()])
        x, P = KF_JAX._update_impl(H_mat, R_mat, P_pred, x_pred, z)

        new_carry = {
            "x": x, "P": P, "A": A, "B": B,
            "leg_pos": new_leg_pos, "leg_vel": new_leg_vel,
            "history_j": new_history_j,
            "tau_diff_prev": tau_i - tau_nom,
            "hist_step": new_hist_step,
        }
        if not use_external_base_acc:
            new_carry["base_acc_i"] = base_acc_i
        if use_gm_observer:
            new_carry["p_hat"] = p_hat_new
            new_carry["f_hat"] = f_hat_new
            new_carry["f_hat_prev"] = f_filtered

        ys_i = {
            "pos_predict": x_pred[POS], "vel_predict": x_pred[LIN_VEL], "ang_vel_predict": x_pred[ANG_VEL],
            "pos_update": x[POS], "vel_update": x[LIN_VEL], "ang_vel_update": x[ANG_VEL],
            "leg_odom_vel": new_leg_vel,
            # nominal = payload-free MuJoCo model, residual = raw network output,
            # corrected = what the filter actually uses (nominal + residual).
            "inertia_matrix_nominal": M_nom[:6, :6],
            "inertia_matrix_residual": M_res,
            "inertia_matrix_corrected": inertia_matrix[:6, :6],
            "qfrc_bias_nominal": qfrc_nom[:6],
            "qfrc_bias_residual": qfrc_res,
            "qfrc_bias_corrected": qfrc_bias[:6],
            "tau_diff_pred": tau_diff_pred
        }
        if est_mode in (2, 3, 4):
            ys_i["c_force_predict"] = x_pred[C_FORCE].reshape((4, 3))
            ys_i["c_force_update"] = x[C_FORCE].reshape((4, 3))
        if not use_external_c_force:
            ys_i["c_force_meas"] = c_force
        if use_gm_observer:
            ys_i["c_state_est"] = c_state
            ys_i["f_hat_history"] = f_hat_out

        return new_carry, ys_i

    print("Running state estimation (CaDeLaC, compiled scan)")
    run_scan = jax.jit(lambda carry0, xs: jax.lax.scan(step, carry0, xs))
    _, ys = run_scan(carry0, xs)

    def get(field):
        return ys[field] if field in ys else jnp.array([])

    result = {
        "pos_predict": get("pos_predict"),
        "vel_predict": get("vel_predict"),
        "ang_vel_predict": get("ang_vel_predict"),
        "c_force_predict": get("c_force_predict"),
        "pos_update": get("pos_update"),
        "vel_update": get("vel_update"),
        "ang_vel_update": get("ang_vel_update"),
        "c_force_update": get("c_force_update"),
        "c_force_meas": get("c_force_meas"),
        "c_state_est": get("c_state_est"),
        "f_hat_history": get("f_hat_history"),
        "leg_odom_vel": get("leg_odom_vel"),
        # "nominal"   = payload-free MuJoCo model
        # "residual"  = raw network output M~(z) / b~(z), i.e. the payload alone
        # "corrected" = nominal + residual, the dynamics the filter actually runs on
        "inertia_matrix": {"nominal": list(ys["inertia_matrix_nominal"]),
                           "residual": list(ys["inertia_matrix_residual"]),
                           "corrected": list(ys["inertia_matrix_corrected"])},
        "qfrc_bias": {"nominal": list(ys["qfrc_bias_nominal"]),
                      "residual": list(ys["qfrc_bias_residual"]),
                      "corrected": list(ys["qfrc_bias_corrected"])},
        "tau_diff_pred": get("tau_diff_pred"),
        "hyper": hyper
    }

    return result
