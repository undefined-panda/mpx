import numpy as np

def estimate_acc_from_contact_force(m, contact_states, contact_forces) ->  np.ndarray:
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
    Fg = m * np.array([0, 0, g])

    force = np.zeros((3,))

    # sum jacobians of all legs that are in contact
    for i in range(4):
        c_i = contact_states[i]
        cf_i = contact_forces[i]

        force += c_i * cf_i

    acc = (force - Fg) / m

    return acc

def estimate_contact_forces(joint_torque, contact_state, legs_order, lin_jacobian_w) ->  np.ndarray:
    """Estimate forces acting on the contact points (feet) using the dynamics model.

    Currently: tau = J * f -> f = (J^-1) * tau

    Args:
        joint_torque (np.ndarray): joint torque
        contact_state (np.ndarray): contact state
        legs_order (list | tuple): order of robot legs
        lin_jacobian_w (LegsAttr): lineare Jacobian of legs in world frame

    Raises:
        ValueError: error if contact_state or joint_torque are None
    """

    if contact_state is None or joint_torque is None:
        raise ValueError(f"contact_state and joint_torque are needed to estimate contact_force. contact_state: {contact_state} \t joint_torque: {joint_torque}")

    contact_forces = []

    # sum jacobians of all legs that are in contact
    for i in range(len(legs_order)):
        leg_name = legs_order[i]

        J_lin = lin_jacobian_w[leg_name][:, 6 + 3*i : 6 + 3*(i+1)] # create 3x3 matrix of values corresponding to current leg
        tau_leg = joint_torque[3*i : 3*(i+1)] # same for torque
        
        f_leg = -np.linalg.pinv(J_lin.T) @ tau_leg
        c_force = contact_state[i] * f_leg
        contact_forces.append(c_force)
    
    return np.array(contact_forces)
