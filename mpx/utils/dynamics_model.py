import numpy as np
import mujoco
from mpx.utils.kf_utils import skew

def estimate_acc_from_contact_force(mass, contact_states, contact_forces) ->  np.ndarray:
    """Estimate base acceleration from contact force by using Newton's second law of motion

    Args:
        m (float): body mass
        contact_states (np.ndarray): contact state
        contact_forces (np.ndarray): contact force

    Raises:
        ValueError: error if contact_state or contact_forces are None

    Returns:
        np.ndarray: estimated base acceleration
    """

    if contact_states is None or contact_forces is None:
        raise ValueError(f"contact_states and contact_forces are needed to estimate the base acc. We have contact_states: {contact_states} \t contact_forces: {contact_forces}")

    g = 9.81
    Fg = mass * np.array([0, 0, g])

    force = np.zeros((3,))

    # sum jacobians of all legs that are in contact
    for i in range(4):
        force += contact_states[i] * contact_forces[i]

    lin_acc = (force - Fg) / mass

    return lin_acc

def estimate_acc_from_contact_force_v2(env, joint_acc, contact_forces, contact_states) -> np.ndarray:
    force = np.zeros((3,))

    # sum jacobians of all legs that are in contact
    for i in range(4):
        c_i = contact_states[i]
        cf_i = contact_forces[i]
        force += c_i * cf_i

    qfrc_bias = env.mjData.qfrc_bias # contains coriolis and gravitational terms for each DOF, shape == (18,)

    M = np.zeros((env.mjModel.nv, env.mjModel.nv)) # shape == (18, 18)
    mujoco.mj_fullM(env.mjModel, M, env.mjData.qM)

    H_B = M[:6, :6]
    H_BL = M[:6, 6:18]

    force = np.concat([force, np.zeros((3,))])
    rhs = - H_BL @ joint_acc - qfrc_bias[:6] + force[:6]
    base_acc = np.linalg.solve(H_B, rhs)
    
    return base_acc[:3]

def estimate_acc_from_contact_force_v3(joint_acc, contact_forces, contact_states, contact_pos_b, orient, M, qfrc_bias) -> np.ndarray:
    force = np.zeros((6,))

    # sum jacobians of all legs that are in contact
    for i in range(4):
        c_i = contact_states[i]
        cf_i = contact_forces[i]
        cp_i = orient @ contact_pos_b[i] # contact position of foot in world frame relative to base, i.e. R @ contact_pos_b
        jacobian = np.concat([np.eye(3), skew(cp_i)]) # also considering angular acceleration
        force += c_i * (jacobian @ cf_i)

    # qfrc_bias = env.mjData.qfrc_bias # contains coriolis and gravitational terms for each DOF, shape == (18,)

    # M = np.zeros((env.mjModel.nv, env.mjModel.nv)) # shape == (18, 18)
    # mujoco.mj_fullM(env.mjModel, M, env.mjData.qM)

    H_B = M[:6, :6]
    H_BL = M[:6, 6:18]

    rhs = - H_BL @ joint_acc - qfrc_bias[:6] + force[:6]
    base_acc = np.linalg.solve(H_B, rhs)
    
    return base_acc[:6]

def estimate_contact_forces(joint_torque, contact_state, legs_order, J_w):
    """Estimate forces acting on the contact points (feet) using the dynamics model.

    Currently: tau = J * f -> f = (J^-1) * tau

    Args:
        joint_torque (np.ndarray): joint torque
        contact_state (np.ndarray): contact state
        legs_order (list): order of robot legs
        lin_jacobian_w (LegsAttr): linear Jacobian of legs in world frame

    Return:
        Estimation of contact force for each leg in contact
    """
    
    contact_forces = []

    # sum jacobians of all legs that are in contact
    for i in range(4):
        J_lin = J_w[legs_order[i]][:, 6 + 3*i : 6 + 3*(i+1)] # create 3x3 matrix of values corresponding to current leg
        tau_leg = joint_torque[3*i : 3*(i+1)] # same for torque
        
        c_force = contact_state[i] * (-np.linalg.pinv(J_lin.T) @ tau_leg)
        contact_forces.append(c_force)
    
    return np.array(contact_forces)

def estimate_contact_states(contact_force, threshold):
    return contact_force > threshold

class GMContactObserver:
    def __init__(self, dt, L1, L2, thresholds):
        self.p_hat = np.zeros(18) # 6 DOF base + 12 DOF legs
        self.f_hat = np.zeros(12) # 3x1 vector for 4 feet
        self.f_hat_prev = np.zeros(12)
        self.dt = dt
        self.L1 = L1
        self.L2 = L2
        self.thresholds = thresholds
        self.torque = np.zeros(18)

        self.alpha = 0.8
        self.initialized = False
    
    def _q(self, s):
        return np.sign(s) * np.sqrt(np.abs(s)) + s

    def _k1(self, s):
        return self._q(s)
    
    def _k2(self, s):
        return np.sign(s) + self._q(s)

    def step(self, vel, M, joint_torque, J, qfrc_bias):
        self.torque[6:] = joint_torque # [:6] = 0 since base is floating
        p_measured = M @ vel
        tau_bar = self.torque - qfrc_bias
        innovation = p_measured - self.p_hat

        p_hat_dot = - J.T @ self.f_hat + tau_bar + self.L1 * self._k1(innovation)
        f_hat_dot = self.L2 * self._k2(innovation[6:])

        self.p_hat += p_hat_dot * self.dt
        self.f_hat += f_hat_dot * self.dt

        # z-axis constraint and positive clipping
        self.f_hat[0::3] = 0.0
        self.f_hat[1::3] = 0.0
        self.f_hat[2::3] = np.maximum(self.f_hat[2::3], 0.0)

        if not self.initialized:
            self.f_hat_prev = self.f_hat.copy()
            self.initialized = True

        # filtering
        f_filtered = self.alpha * self.f_hat_prev + (1 - self.alpha) * self.f_hat
        self.f_hat_prev = f_filtered

        contact_state = f_filtered[2::3] > self.thresholds
        return contact_state, f_filtered
