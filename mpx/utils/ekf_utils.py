import numpy as np
from utils.plot_data import *

def load_custom_dataset(dataset_path, sim_num):
    dataset = np.load(dataset_path)

    data = {"num_datasets": len(dataset["time"]),
            "num_datapoints": len(dataset["time"][0]),
            "dt": dataset["dt"][sim_num],
            "time": dataset["time"][sim_num],
            "base_pos": dataset["base_pos"][sim_num],
            "base_orient": dataset["base_orient"][sim_num],
            "base_vel" : dataset["base_vel"][sim_num],
            "base_ang_vel" : dataset["base_ang_vel"][sim_num],
            "base_acc" : dataset["base_acc"][sim_num],
            "joint_pos" : dataset["joint_pos"][sim_num],
            "joint_vel" : dataset["joint_vel"][sim_num],
            "joint_acc" : dataset["joint_acc"][sim_num],
            "joint_torque" : dataset["joint_torque"][sim_num],
            "contact_states" : dataset["contact_states"][sim_num],
            "contact_pos" : dataset["contact_pos"][sim_num],
            "contact_forces" : dataset["contact_forces"][sim_num]}
    
    print(f"Data loaded from {dataset_path}.")

    return data

def compute_pos_from_estimated_vel(velocities, dt):
    positions = []
    prev_pos = np.zeros((3,))
    for vel in velocities:
        next_pos = prev_pos + dt * vel
        positions.append(next_pos)
        prev_pos = next_pos
    
    return np.array(positions)

def estimation_error(ground_truth, estimation, title, subplot_title=None, plot=True, return_errors=False):
    errors = ground_truth - np.array(estimation)
    
    if plot:
        num_plots = ground_truth.shape[1]
        if subplot_title is None: titles = ["x", "y", "z"]
        fig, ax = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
        fig.suptitle(f"{title} Estimation Error over Time")

        for i in range(num_plots):
            ax[i].plot(errors[:, i])
            ax[i].set_title(titles[i]+" - Value")
            ax[i].set_xlabel("Time")
            ax[i].set_ylabel("Error")

        plt.show()
    
    if return_errors: return errors

def estimate_acc_from_contact_force(env, contact_states, contact_forces):
    m = float(np.sum(env.mjModel.body_mass))
    g = 9.81
    Fg = m * np.array([0, 0, g])

    force = np.zeros((3,))

    # sum jacobians of all legs that are in contact
    for i in range(4):
        c_i = contact_states[i]
        cf_i = contact_forces[i]

        force += c_i * cf_i

    acc = (force - Fg) / m

    return acc, force

def estimate_acc_from_estimated_contact_force(env, torque, contact_states):
    m = float(np.sum(env.mjModel.body_mass))
    g = 9.81
    Fg = m * np.array([0, 0, g])
    J_feet_linear = env.feet_jacobians(frame="world")

    force = np.zeros((3,))
    contact_forces = []

    # sum jacobians of all legs that are in contact
    for i in range(4):
        leg_name = env.legs_order[i]
        c_i = contact_states[i]

        J_lin = c_i * J_feet_linear[leg_name][:, 6:]
        force += J_lin @ torque
        contact_forces.append(force)

    acc = (force - Fg) / m

    return acc, force, contact_forces

def estimate_contact_state(contact_force, threshold):
    contact_force = np.array(contact_force)
    if contact_force.shape == (4,3): # x, y, z values for force
        mask = (np.sqrt(contact_force[:, 0]**2 + contact_force[:, 1]**2) <= contact_force[:, 2])
        contact_state = (contact_force != 0)[:, 0] & mask
    elif contact_force.shape == (4,): # single value for force
        contact_state = contact_force > threshold
    else:
        print(f"contact_force has invalid shape: {contact_force.shape}")
        return
    return contact_state

def quat_to_rot(orient) -> np.ndarray:
    """Convert quaternion to rotation matrix (source: https://cookierobotics.com/080/).

    Args:
        orient (np.ndarray | list): quaternion in [w, x, y, z] format

    Returns:
        np.ndarray: corresponding rotation matrix
    """

    w, x, y, z = orient
    R = np.array([
        [2*(w**2 + x**2) - 1, 2*(x*y - w*z)      , 2*(w*y + x*z)      ],
        [2*(x*y + w*z)      , 2*(w**2 + y**2) - 1, 2*(y*z - w*x)      ],
        [2*(x*z - w*y)      , 2*(y*z + w*x)      , 2*(w**2 + z**2) - 1]
    ])
    
    return R

def rot_to_quat(orient):
    # source: https://www.johndcook.com/blog/2025/05/07/quaternions-and-rotation-matrices/
    r11, r22, r33 = orient[0,0], orient[1,1], orient[2,2]
    w = 1/2 * (1 + r11 + r22 + r33)**0.5
    x = 1/2 * (1 + r11 - r22 - r33)**0.5 * np.sign(orient[2,1] - orient[1,2])
    y = 1/2 * (1 - r11 + r22 - r33)**0.5 * np.sign(orient[0,2] - orient[2,0])
    z = 1/2 * (1 - r11 - r22 + r33)**0.5 * np.sign(orient[1,0] - orient[0,1])

    return np.array([w, x, y, z])

def rpy_to_rot(orient) -> np.ndarray:
    """Convert rpy angles to rotation matrix.

    Args:
        orient (np.ndarray | list): rpy angles in roll, pitch, yaw format

    Returns:
        np.ndarray: corresponding rotation matrix
    """

    roll, pitch, yaw = orient
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    # Combined rotation matrix: ZYX sequence
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

def skew(w):
    w1, w2, w3 = w
    w_skew = np.array([[0, -w3, w2],
                       [w3, 0, -w1],
                       [-w2, w1, 0]])
    
    return w_skew

def matrix_exp(w, dt):
    # orientation estimation. source: https://cwzx.wordpress.com/2013/12/16/numerical-integration-for-rotational-dynamics/
    w_scaled = dt * w
    theta = np.linalg.norm(w_scaled) # source: http://mainline.brynmawr.edu/~dxu/206-2550-2.pdf
    w_skew = skew(w_scaled)

    # matrix exponential. source: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    return np.eye(3) + (np.sin(theta)/theta) * w_skew + ((1-np.cos(theta))/(theta**2)) * (w_skew @ w_skew)

def is_rotation_matrix(R):
    return (
        R.shape == (3, 3) and
        np.allclose(R.T @ R, np.eye(3)) and  # R^T R = I
        np.isclose(np.linalg.det(R), 1.0)    # det(R) = 1
    )

def skew_inverse(S):
    return np.array([S[2,1], S[0,2], S[1,0]])

def is_quaternion(q):
    return q.shape == (4,) and np.isclose(np.linalg.norm(q), 1.0)

def matrix_log(R):
    # Winkel der Rotation
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    
    if theta < 1e-6:  # kleine Winkel: Grenzwert
        return skew_inverse(R - R.T) * 0.5
    
    # schiefsymmetrischer Teil -> Rotationsvektor
    log_matrix = (theta / (2 * np.sin(theta))) * (R - R.T)
    return skew_inverse(log_matrix)
