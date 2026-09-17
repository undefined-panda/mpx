import numpy as np
from state_estimation.kf_utils import skew

def estimate_contact_forces(joint_torque, contact_states, jacobians, legs_order, xp=np):
    """Estimate ground reaction force at each foot from joint torques.

    Inverts the static relation  τ_i = J_iᵀ · F_i  per leg, giving F_i = (J_iᵀ)⁻¹ · τ_i
    """
    J_stack = xp.stack([jacobians[i][:, 6 + 3*i : 6 + 3*(i+1)].T for i in range(len(legs_order))])
    tau = joint_torque.reshape(4, 3)
    forces = -xp.linalg.solve(J_stack, tau[...,None]).squeeze(-1)
    return forces * xp.asarray(contact_states)[:, None]

def _get_base_acc(orient, joint_acc, contact_force, contact_state,
                  p_b, mass, inertia_matrix, qfrc_bias,
                  base_acc_est_type, xp=np):
    match base_acc_est_type:
        case "Newton":
            base_acc = compute_base_acc_newton_euler(mass, contact_state, contact_force, xp=xp)
        case "Rigid-Body-Dynamics":
            base_acc = compute_base_acceleration_full_dynamics(joint_acc, contact_force, contact_state, p_b, orient, inertia_matrix, qfrc_bias, xp)
        case _:
            raise ValueError(f"Unknown base acc estimation type: {base_acc_est_type}")

    return base_acc

def compute_base_acc_newton_euler(mass, contact_states, contact_forces, g=9.81, xp=np):
    """Estimate base linear acceleration from contact forces via Newton's 2nd law.

    a = (Σ_i c_i · (F_i - m·g)) / m
    """

    total_force = (contact_states[:, None] * contact_forces).sum(axis=0)
    gravity = mass * xp.array([0.0, 0.0, g])

    return (total_force - gravity) / mass

def compute_base_acceleration_full_dynamics(joint_acc, contact_force, contact_state, contact_pos_b, orient, M, qfrc_bias, xp=np):
    """Estimate base linear and angular acceleration from floating-base rigid-body dynamics.
    """
    contact_state = xp.asarray(contact_state)
    cf_b = contact_force @ orient # rotate contact force to body frame to match MuJoCo convention

    lin = xp.sum(contact_state[:, None] * contact_force, axis=0) # sum of contact forces
    ang = xp.sum(contact_state[:, None] * xp.cross(contact_pos_b, cf_b), axis=0) # sum of torques
    force = xp.concatenate([lin, ang])

    H_B = M[:6, :6] # base inertia matrix
    H_BL = M[:6, 6:18] # coupling between base and legs

    if H_BL.shape[1] > 6: # full 18x18-inertia (MuJoCo)
        coupling = H_BL @ joint_acc
    else: # 6x6 Base-only (CaDeLaC): no leg-coupling
        coupling = xp.zeros(6)      

    rhs = -coupling - qfrc_bias[:6] + force
    return xp.linalg.solve(H_B, rhs)

def compute_contact_force_dynamics(orient, contact_pos_b, contact_state, M, qfrc_bias, xp=np):
    """Returns (J_new, coupling, bias) — the three blocks that go into A/B."""
    H_B  = M[:6, :6]
    H_BL = M[:6, 6:18]

    blocks = []
    for i in range(4):
        p_w = orient @ contact_pos_b[i]
        top = xp.eye(3)
        bot = skew(p_w, xp=xp) @ orient.T # convert bottom part to base frame
        block = xp.concatenate([top, bot], axis=0)
        blocks.append(block * contact_state[i])
    J_full = xp.concatenate(blocks, axis=1)

    J_new = xp.linalg.solve(H_B, J_full)
    coupling = xp.linalg.solve(H_B, -H_BL)
    bias = xp.linalg.solve(H_B, -qfrc_bias[:6].reshape(-1, 1))
    return J_new, coupling, bias
