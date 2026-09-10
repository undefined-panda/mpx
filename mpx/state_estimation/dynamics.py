import numpy as np
from state_estimation.kf_utils import skew

def estimate_contact_forces(joint_torque, contact_states, jacobians, legs_order):
    """Estimate ground reaction force at each foot from joint torques.

    Inverts the static relation  τ_i = J_iᵀ · F_i  per leg, giving F_i = (J_iᵀ)⁻¹ · τ_i
    """
    J_stack = np.stack([jacobians[leg][:, 6 + 3*i : 6 + 3*(i+1)].T for i, leg in enumerate(legs_order)])
    tau = joint_torque.reshape(4, 3)
    forces = -np.linalg.solve(J_stack, tau[...,None]).squeeze(-1)
    return forces * np.asarray(contact_states)[:, None]

def _get_base_acc(orient, joint_acc, contact_force, contact_state,
                  p_b, mass, inertia_matrix, qfrc_bias,
                  base_acc_est, base_acc_est_type, mass_est):
    match base_acc_est_type:
        case "Newton":
            base_acc = compute_base_acc_newton_euler(mass, contact_state, contact_force)
        case "Rigid-Body-Dynamics":
            base_acc = compute_base_acceleration_full_dynamics(joint_acc, contact_force, contact_state, p_b, orient, inertia_matrix, qfrc_bias, not mass_est)
        case _:
            raise ValueError(f"Unknown base acc estimation type: {base_acc_est_type}")
    base_acc_est.append(base_acc)

    return base_acc

def compute_base_acc_newton_euler(mass, contact_states, contact_forces, g=9.81):
    """Estimate base linear acceleration from contact forces via Newton's 2nd law.

    a = (Σ_i c_i · (F_i - m·g)) / m
    """

    total_force = (contact_states[:, None] * contact_forces).sum(axis=0)
    gravity = mass * np.array([0.0, 0.0, g])

    return (total_force - gravity) / mass

def compute_base_acceleration_full_dynamics(joint_acc, contact_force, contact_state, contact_pos_b, orient, M, qfrc_bias, include_coupling=True):
    """Estimate base linear and angular acceleration from floating-base rigid-body dynamics.
    """
    contact_state = np.asarray(contact_state)
    cf_b = contact_force @ orient # rotate contact force to body frame to match MuJoCo convention

    lin = np.sum(contact_state[:, None] * contact_force, axis=0) # sum of contact forces
    ang = np.sum(contact_state[:, None] * np.cross(contact_pos_b, cf_b), axis=0) # sum of torques
    force = np.concatenate([lin, ang])

    H_B = M[:6, :6] # base inertia matrix
    H_BL = M[:6, 6:18] # coupling between base and legs

    coupling = np.zeros(6) # 6x6 Base-only (CaDeLaC): no leg-coupling
    if include_coupling: coupling = H_BL @ joint_acc # full 18x18-inertia (MuJoCo)        

    rhs = -coupling - qfrc_bias[:6] + force
    return np.linalg.solve(H_B, rhs)

def compute_contact_force_dynamics(orient, contact_pos_b, contact_state, M, qfrc_bias):
    """Returns (J_new, coupling, bias) — the three blocks that go into A/B."""
    H_B  = M[:6, :6]
    H_BL = M[:6, 6:18]
    J_full = np.zeros((6, 12))
    for i in range(4):
        if contact_state[i]:
            p_w = orient @ contact_pos_b[i]
            J_full[:3, i*3:(i+1)*3] = np.eye(3)
            J_full[3:, i*3:(i+1)*3] = skew(p_w) @ orient.T # convert bottom part to base frame
    J_new    = np.linalg.solve(H_B, J_full)
    coupling = np.linalg.solve(H_B, -H_BL)
    bias     = np.linalg.solve(H_B, -qfrc_bias[:6].reshape(-1, 1))
    return J_new, coupling, bias
