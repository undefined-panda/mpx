import numpy as np
from state_estimation.logging_utils import _print_rmse_table, _print_drift_table, _plot_estimation, _print_cs_table

REQUIRED_KEYS = {
    "base_orient":   "Base orientation (quaternion or rotation matrix) — for frame transforms",
    "base_ang_vel":  "Base angular velocity in base frame — for leg odometry (Eq. 12.21)",
    "joint_pos":     "Joint positions — for forward kinematics of feet",
    "joint_vel":     "Joint velocities — for leg odometry velocity measurement",
    "joint_torque":  "Joint torques — for contact force estimation",
    # "contact_states": "Per-leg contact booleans — masks swing legs from measurements",
}

def validate_data(data, required=REQUIRED_KEYS):
    """Check that data contains everything run_estimation needs.
    """
    missing = [k for k in required if data.get(k) is None]
    if missing:
        details = "\n".join(f"  - {k}: {required[k]}" for k in missing)
        raise ValueError(
            f"run_estimation is missing {len(missing)} required input(s):\n{details}"
        )

def rmse(gt_value, est_value, axis=0):
    return np.sqrt(np.mean((gt_value - est_value)**2, axis=axis))

def rmse_total(gt_value, est_value):
    err = est_value - gt_value
    return np.sqrt(np.mean(np.sum(err**2, axis=1)))

def compute_rmse_state(results):
    """Add rmse and total rmse to results dict.
    """
    rmse_results = results.copy()
    for key, val in results.items():
        gt, est = val["gt"], val["est"]
        rmse_results[key]["rmse"] = rmse(gt, est, axis=0)
        rmse_results[key]["total"] = rmse_total(gt, est)

    return rmse_results

def compute_drift_err(pos_gt, pos_est):
    traj_len = np.sum(np.linalg.norm(np.diff(pos_gt, axis=0), axis=1))
    ate = np.sqrt(np.mean(np.sum((pos_est - pos_gt)**2, axis=1)))
    drift_rate = ate / traj_len * 100

    return {"traj_len": traj_len, "ate": ate, "drift_rate": drift_rate}

def compute_cs_metrics(results):
    cs_results = results.copy()
    for key, val in results.items():
        gt, est = val["gt"], val["est"]
        tp = np.sum( gt &  est)
        fp = np.sum(~gt &  est)
        fn = np.sum( gt & ~est)

        prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else np.nan
        ratio = gt.mean() * 100

        cs_results[key]["precision"] = prec
        cs_results[key]["recall"] = rec
        cs_results[key]["f1"] = f1
        cs_results[key]["ratio"] = ratio

    return cs_results

def evaluate_estimation(result, data, print_metrics=True, plot_results=True):
    """Print per-axis and total RMSE for base state estimation."""
    state_results = {
        "Position" :        {"unit": "m",   "gt": data["base_pos"], "est": result["pos_update"]},
        "Linear Velocity" : {"unit": "m/s", "gt": data["base_vel"], "est": result["vel_update"]},
    }
    legs_order = result["legs_order"]

    # add other state parameters
    ang_vel_est = np.asarray(result["ang_vel_update"])
    if len(ang_vel_est) > 0:
        state_results["Angular Velocity"] = {"unit": "rad/s", "gt": data["base_ang_vel"], "est": result["ang_vel_update"]}

    cf_est = np.asarray(result["contact_force_update"]) # (N, 12)
    if len(cf_est) > 0:
        cf_gt  = np.asarray(data["contact_forces"]) # (N, 4, 3)
        for i, leg in enumerate(legs_order):
            state_results[f"Contact Force {leg}"] = {"unit": "N", "gt": cf_gt[:, i, :], "est": cf_est[:, i*3:(i+1)*3]}

    # log base acc estimation
    dynamics_results = {}
    base_acc_est = np.asarray(result["base_acc_est"])
    if len(base_acc_est) > 0:
        dynamics_results["Linear Acceleration"] = {"unit": "m/s^2", "gt": data["base_acc"][:, :3], "est": result["base_acc_est"][:, :3]}
        if base_acc_est.shape[1] == 6:
            dynamics_results["Angular Acceleration"] = {"unit": "m/s^2", "gt": data["base_acc"][:, 3:], "est": result["base_acc_est"][:, 3:]}

    cs_est_results = {}
    cs_est = np.asarray(result["contact_state_est"])
    if len(cs_est) > 0:
        cs_gt = np.asarray(data["contact_states"])
        for i, leg in enumerate(legs_order):
            cs_est_results[leg] = {"gt": cs_gt[:, i], "est": cs_est[:, i]}

    if print_metrics:
        # RMSE
        rmse_state = compute_rmse_state(state_results)
        _print_rmse_table(rmse_state, "RMSE — State Estimation")
        if len(dynamics_results) > 0:
            rmse_dynamics = compute_rmse_state(dynamics_results)
            _print_rmse_table(rmse_dynamics, "RMSE - Dynamics Estimation")

        # Drift
        drift_err = compute_drift_err(state_results["Position"]["gt"], state_results["Position"]["est"])
        _print_drift_table(drift_err)

        # Contact State
        if len(cs_est_results) > 0:
            cs_metrics = compute_cs_metrics(cs_est_results)
            _print_cs_table(cs_metrics)

    if plot_results:
        time = data["time"]
        for key, val in state_results.items():
            _plot_estimation(gt=val["gt"], est=val["est"], time=time, name=key, unit=val["unit"])

