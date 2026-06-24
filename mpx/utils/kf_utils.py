import numpy as np

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

def skew(w):
    w1, w2, w3 = w
    w_skew = np.array([[0, -w3, w2],
                       [w3, 0, -w1],
                       [-w2, w1, 0]])

    return w_skew

def skew_inverse(S):
    return np.array([S[2,1], S[0,2], S[1,0]])

def matrix_exp(w, dt):
    # orientation estimation. source: https://cwzx.wordpress.com/2013/12/16/numerical-integration-for-rotational-dynamics/
    w_scaled = dt * w
    theta = np.linalg.norm(w_scaled) # source: http://mainline.brynmawr.edu/~dxu/206-2550-2.pdf
    w_skew = skew(w_scaled)

    # matrix exponential. source: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    if theta < 1e-6:
        return np.eye(3) + w_skew
    else:
        return np.eye(3) + (np.sin(theta)/theta) * w_skew + ((1-np.cos(theta))/(theta**2)) * (w_skew @ w_skew)

def matrix_log(R):
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))

    if theta < 1e-6:
        return skew_inverse(R - R.T) * 0.5

    log_matrix = (theta / (2 * np.sin(theta))) * (R - R.T)
    return skew_inverse(log_matrix)

def quat_to_rot(orient):
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
    """Convert rotation matrix to quaternion (Shepherd's method, numerically robust).

    Args:
        orient (np.ndarray): 3x3 rotation matrix

    Returns:
        np.ndarray: corresponding quaternion in [w, x, y, z] format
    """

    R = orient
    trace = R[0,0] + R[1,1] + R[2,2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25 / s,
                         (R[2,1] - R[1,2]) * s,
                         (R[0,2] - R[2,0]) * s,
                         (R[1,0] - R[0,1]) * s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1] - R[1,2]) / s,
                         0.25 * s,
                         (R[0,1] + R[1,0]) / s,
                         (R[0,2] + R[2,0]) / s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2] - R[2,0]) / s,
                         (R[0,1] + R[1,0]) / s,
                         0.25 * s,
                         (R[1,2] + R[2,1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0] - R[0,1]) / s,
                         (R[0,2] + R[2,0]) / s,
                         (R[1,2] + R[2,1]) / s,
                         0.25 * s])
