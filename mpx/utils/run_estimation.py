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
                         tau_diff=None):
    """Run the Kalman Filter state estimation.

    est_mode:
    - 1 = estimation of pos, lin vel and ang vel (base acc as control i xput)
    - 2 = estimation of pos, lin vel, ang vel and contact force (identity in A)
    - 3 = estimation of pos, lin vel, ang vel and contact force (contact state diagonal block matrix in A)
    - 4 = estimation of pos, lin vel, ang vel and contact force (identity in A) with base acc inside KF

    Without a CaDeLaC checkpoint (`cadelac_path=None`), this runs a plain-numpy,
    MuJoCo-in-the-loop recursive filter. With a CaDeLaC checkpoint, MuJoCo's per-step
    kinematics/dynamics outputs are precomputed once for the whole trajectory (they only
    depend on the given trajectory data, not on the filter's own recursive state), and
    the recursive filter (KF + leg odometry + contact estimation + CaDeLaC) runs as a
    single compiled `jax.lax.scan` with no MuJoCo/eager-JAX interleaving per step.
    """
    if cadelac_path is None:
        return _run_state_estimation_numpy(
            dt=dt, base_orient=base_orient, base_ang_vel=base_ang_vel, joint_pos=joint_pos,
            joint_vel=joint_vel, joint_acc=joint_acc, Q=Q, R=R, base_acc=base_acc, joint_torque=joint_torque, 
            contact_forces=contact_forces, contact_states=contact_states, contact_state_threshold=contact_state_threshold,
            model_name=model_name, L1=L1, L2=L2, contact_thresholds=contact_thresholds, est_mode=est_mode)

    return _run_state_estimation_cadelac(
        dt=dt, base_orient=base_orient, base_ang_vel=base_ang_vel, joint_pos=joint_pos,
        joint_vel=joint_vel, joint_acc=joint_acc, Q=Q, R=R, base_acc=base_acc, base_vel=base_vel,
        base_pos=base_pos, joint_torque=joint_torque, contact_forces=contact_forces,
        contact_states=contact_states, contact_state_threshold=contact_state_threshold,
        model_name=model_name, L1=L1, L2=L2, contact_thresholds=contact_thresholds, est_mode=est_mode,
        cadelac_path=cadelac_path, tau_diff=tau_diff)


def _run_state_estimation_numpy(dt, base_orient, base_ang_vel, joint_pos, joint_vel, joint_acc, Q, R,
                                 base_acc, joint_torque, contact_forces,
                                 contact_states, contact_state_threshold, model_name, L1, L2,
                                 contact_thresholds, est_mode):
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
                                   contact_thresholds, est_mode, cadelac_path, tau_diff):
    num_data = len(base_ang_vel)

    # Reference instances, used only to read off correctly-constructed constant
    # matrices/state-slice definitions (KF_JAX.__init__'s A/B/Q/R/H branching on
    # est_mode). Their mutating instance methods are never called during the scan --
    # the step functions call KF_JAX's/GMContactObserver_JAX's pure implementations
    # directly, threading state explicitly through the scan carry instead.
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
    feature_dim = 30
    kf_warmup = time_window
    switch_idx = min(2 * time_window, num_data)

    # ---- Stage A: MuJoCo precompute over the FULL trajectory (fast, plain numpy) ----
    stage_a = _precompute_mujoco_stage(leg_odom, np.asarray(base_orient), np.asarray(joint_pos), np.asarray(joint_vel))

    # ---- Cast trajectory + Stage A outputs to device once (not per-iteration) ----
    base_orient_j = jnp.asarray(base_orient)
    base_ang_vel_j = jnp.asarray(base_ang_vel)
    joint_pos_j = jnp.asarray(joint_pos)
    joint_vel_j = jnp.asarray(joint_vel)
    joint_acc_j = jnp.asarray(joint_acc)
    base_vel_j = jnp.asarray(base_vel) if base_vel is not None else jnp.zeros((num_data, 3))
    base_pos_j = jnp.asarray(base_pos) if base_pos is not None else jnp.zeros((num_data, 3))
    tau_diff_j = jnp.asarray(tau_diff)
    joint_torque_j = jnp.asarray(joint_torque) if joint_torque is not None else None
    contact_forces_j = jnp.asarray(contact_forces) if contact_forces is not None else None
    contact_states_j = jnp.asarray(contact_states) if contact_states is not None else None
    base_acc_j = jnp.asarray(base_acc) if base_acc is not None else None

    J_b_j = jnp.asarray(stage_a["J_b"])
    J_w_j = jnp.asarray(stage_a["J_w"])
    p_b_j = jnp.asarray(stage_a["p_b"])
    M_mujoco_j = jnp.asarray(stage_a["M"])
    qfrc_bias_mujoco_j = jnp.asarray(stage_a["qfrc_bias"])

    def shift1(arr):
        # "value from step i-1", edge-padded at i=0 (never actually read there,
        # since accumulate_mask[0] is False given kf_warmup >= 1).
        return jnp.concatenate([arr[0:1], arr[:-1]], axis=0)

    base_orient_prev = shift1(base_orient_j)
    base_vel_prev = shift1(base_vel_j)
    base_ang_vel_prev = shift1(base_ang_vel_j)
    base_pos_z_prev = shift1(base_pos_j[:, 2])
    base_acc_prev_j = shift1(base_acc_j) if base_acc_j is not None else None
    joint_pos_prev_j = shift1(joint_pos_j)
    joint_vel_prev_j = shift1(joint_vel_j)

    accumulate_mask = jnp.arange(num_data) > kf_warmup

    def build_feature_vector(*args):
        return jnp.concatenate([jnp.atleast_1d(jnp.asarray(a)) for a in args])

    # ---- Build the xs (scanned-input) pytree, sliced per phase below ----
    xs_full = {
        "base_orient": base_orient_j,
        "base_orient_prev": base_orient_prev,
        "base_ang_vel": base_ang_vel_j,
        "base_ang_vel_prev": base_ang_vel_prev,
        "base_vel_prev": base_vel_prev,
        "base_pos_z_prev": base_pos_z_prev,
        "tau_diff": tau_diff_j,
        "joint_pos": joint_pos_j,
        "joint_pos_prev_j": joint_pos_prev_j,
        "joint_vel": joint_vel_j,
        "joint_vel_prev_j": joint_vel_prev_j,
        "joint_acc": joint_acc_j,
        "J_b": J_b_j,
        "J_w": J_w_j,
        "p_b": p_b_j,
        "M_mujoco": M_mujoco_j,
        "qfrc_bias_mujoco": qfrc_bias_mujoco_j,
        "accumulate_mask": accumulate_mask,
    }
    if joint_torque_j is not None:
        xs_full["joint_torque"] = joint_torque_j
    if contact_states_j is not None:
        xs_full["contact_states"] = contact_states_j
    if contact_forces_j is not None:
        xs_full["contact_forces"] = contact_forces_j
    if base_acc_j is not None:
        xs_full["base_acc"] = base_acc_j
        xs_full["base_acc_prev"] = base_acc_prev_j

    def slice_xs(lo, hi):
        return {k: v[lo:hi] for k, v in xs_full.items()}

    xs1 = slice_xs(0, switch_idx)
    xs2 = slice_xs(switch_idx, num_data)

    # ---- Carry (recursive state threaded through both scan phases) ----
    x_size = kf_ref.x.shape[0]
    carry0 = {
        "x": jnp.zeros(x_size),
        "P": Q_mat,
        "A": kf_ref.A,
        "B": kf_ref.B,
        "leg_pos": jnp.zeros(3),
        "leg_vel": jnp.zeros(3),
        "history_j": jnp.zeros((time_window, feature_dim)),
        "history_count": jnp.int32(0),
    }
    if not use_external_base_acc:
        carry0["base_acc_i"] = jnp.zeros(6)
    carry0["tau_pred_prev"] = jnp.zeros(6)
    if use_gm_observer:
        carry0["p_hat"] = jnp.zeros(18)
        carry0["f_hat"] = jnp.zeros(12)
        carry0["f_hat_prev"] = jnp.zeros(12)

    def make_step(cadelac_active):
        def step(carry, xs_i):
            orient = xs_i["base_orient"]
            orient_rot = quat_to_rot(orient, enable_jax=True)

            # --- CaDeLaC feature-history buffer ---
            # feature = build_feature_vector(xs_i["base_orient_prev"],
            #                                 xs_i["base_vel_prev"],
            #                                 xs_i["base_ang_vel_prev"],
            #                                 xs_i["base_pos_z_prev"],
            #                                 xs_i["tau_diff"])
            # Autoregressive feedback: `tau_diff` is not a measurement -- it's the
            # residual torque (true payload-affected dynamics minus nominal MuJoCo
            # dynamics) that CaDeLaC itself is trained to predict as `tau_pred`. Once
            # CaDeLaC is active, feeding the recorded dataset's ground-truth `tau_diff`
            # here would leak information a real deployment never has; the model must
            # feed back its OWN previous prediction (`carry["tau_pred_prev"]`) instead.
            # Before CaDeLaC has run even once (phase 1 / warmup), no prediction exists
            # yet, so the history is bootstrapped from the ground-truth `tau_diff` --
            # unavoidable, and standard practice for autoregressive models.
            tau_feature = carry["tau_pred_prev"] if cadelac_active else xs_i["tau_diff"]
            feature = build_feature_vector(xs_i["joint_pos_prev_j"],
                                            xs_i["joint_vel_prev_j"],
                                            tau_feature)
            rolled = jnp.roll(carry["history_j"], -1, axis=0).at[-1].set(feature)
            if cadelac_active:
                # Once active, every step accumulates unconditionally (matches the
                # original loop: by construction this phase only ever contains
                # indices where `i > kf_warmup` already holds).
                new_history_j = rolled
                new_history_count = carry["history_count"] + 1
            else:
                mask_i = xs_i["accumulate_mask"]
                new_history_j = jnp.where(mask_i, rolled, carry["history_j"])
                new_history_count = jnp.where(mask_i, carry["history_count"] + 1, carry["history_count"])

            # --- Inertia matrix / qfrc_bias source ---
            # CaDeLaC/DeLaN predicts a RESIDUAL correction (base-only, 6x6 / 6-dim),
            # not the absolute mass matrix/bias -- it's trained on `diff_tau` (the
            # difference between the true, payload-affected dynamics and the nominal
            # MuJoCo dynamics), so its output must be ADDED onto the nominal MuJoCo
            # values, not substituted for them. The nominal values are exactly what
            # Stage A already precomputed (`leg_odom.env` has no knowledge of the true,
            # randomized payload mass, so it already IS the nominal/no-payload model,
            # evaluated at the real joint trajectory) -- no separate MuJoCo query
            # needed here. Only the base-base block [:6,:6]/[:6] is corrected; the
            # leg-related coupling stays purely nominal, since CaDeLaC never models it.
            M_nom = xs_i["M_mujoco"]
            qfrc_nom = xs_i["qfrc_bias_mujoco"]
            if cadelac_active:
                q_in = jnp.concatenate([carry["x"][POS], quat_to_euler(orient, True)])[None, ...]
                qd_in = jnp.concatenate([carry["x"][LIN_VEL], xs_i["base_ang_vel"]])[None, ...]
                qdd_src = carry["base_acc_i"] if not use_external_base_acc else xs_i["base_acc_prev"]
                qdd_in = qdd_src[None, ...]
                history_in = new_history_j[None, ...]
                tau_pred, dEdt, extras = model.apply(params, q_in, qd_in, qdd_in, history_in)
                M_res = extras["M"][0]
                qfrc_res = extras["qfrc_bias"][0].reshape(-1)
                inertia_matrix = M_nom.at[:6, :6].add(M_res)
                qfrc_bias = qfrc_nom.at[:6].add(qfrc_res)
                # Drop-in slot for the next step's `tau_diff` feature (see above).
                # Shape/sign/unit correspondence between `tau_pred` and the ground-
                # truth `tau_diff` it replaces is NOT yet verified -- only reshaped
                # defensively so this plugs into the (6,) feature slot. Revisit if the
                # filter behaves oddly once CaDeLaC is active.
                new_tau_pred_prev = jnp.reshape(tau_pred, (-1,))
            else:
                inertia_matrix = M_nom
                qfrc_bias = qfrc_nom
                new_tau_pred_prev = carry["tau_pred_prev"]

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
                # inertia_matrix/qfrc_bias are always the full 18x18/18-dim nominal
                # MuJoCo values (optionally corrected in the base block by CaDeLaC's
                # residual, see above) -- never a bare 6x6 CaDeLaC-only matrix -- so
                # the leg-coupling block H_BL is always populated and usable.
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
                "history_j": new_history_j, "history_count": new_history_count,
                "tau_pred_prev": new_tau_pred_prev,
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
            }
            if est_mode in (2, 3, 4):
                ys_i["c_force_predict"] = x_pred[C_FORCE].reshape((4, 3))
                ys_i["c_force_update"] = x[C_FORCE].reshape((4, 3))
            if not use_external_c_force:
                ys_i["c_force_meas"] = c_force
            if use_gm_observer:
                ys_i["c_state_est"] = c_state
                ys_i["f_hat_history"] = f_hat_out
            if cadelac_active:
                # "cadelac" here means the corrected (nominal + residual) base block
                # actually fed into the filter -- comparable in shape/meaning to the
                # pure-nominal "mujoco" entry, not the raw residual by itself.
                ys_i["inertia_matrix_mujoco"] = M_nom[:6, :6]
                ys_i["inertia_matrix_cadelac"] = inertia_matrix[:6, :6]
                ys_i["qfrc_bias_mujoco"] = qfrc_nom[:6]
                ys_i["qfrc_bias_cadelac"] = qfrc_bias[:6]

            return new_carry, ys_i

        return step

    step1 = make_step(cadelac_active=False)
    step2 = make_step(cadelac_active=True)

    @jax.jit
    def _run_both_phases(carry0, xs1, xs2):
        carry1, ys1 = jax.lax.scan(step1, carry0, xs1)
        carry2, ys2 = jax.lax.scan(step2, carry1, xs2)
        return ys1, ys2

    print("Running state estimation (CaDeLaC, compiled scan)")
    ys1, ys2 = _run_both_phases(carry0, xs1, xs2)

    def cat(field):
        parts = []
        if field in ys1:
            parts.append(ys1[field])
        if field in ys2:
            parts.append(ys2[field])
        return jnp.concatenate(parts, axis=0) if parts else jnp.array([])

    result = {
        "pos_predict": cat("pos_predict"),
        "vel_predict": cat("vel_predict"),
        "ang_vel_predict": cat("ang_vel_predict"),
        "c_force_predict": cat("c_force_predict"),
        "pos_update": cat("pos_update"),
        "vel_update": cat("vel_update"),
        "ang_vel_update": cat("ang_vel_update"),
        "c_force_update": cat("c_force_update"),
        "c_force_meas": cat("c_force_meas"),
        "c_state_est": cat("c_state_est"),
        "f_hat_history": cat("f_hat_history"),
        "leg_odom_vel": cat("leg_odom_vel"),
    }
    if "inertia_matrix_mujoco" in ys2:
        result["inertia_matrix"] = {"mujoco": list(ys2["inertia_matrix_mujoco"]),
                                     "cadelac": list(ys2["inertia_matrix_cadelac"])}
        result["qfrc_bias"] = {"mujoco": list(ys2["qfrc_bias_mujoco"]),
                                "cadelac": list(ys2["qfrc_bias_cadelac"])}
    else:
        result["inertia_matrix"] = {"mujoco": [], "cadelac": []}
        result["qfrc_bias"] = {"mujoco": [], "cadelac": []}

    return result
