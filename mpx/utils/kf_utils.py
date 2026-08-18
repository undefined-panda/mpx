import numpy as np
import mujoco
import jax.numpy as jnp

def _xp(enable_jax):
    return jnp if enable_jax else np

def load_custom_dataset(dataset_path, sim_num=None, print_keys=False):
    dataset = np.load(dataset_path)

    data = {
        "num_datasets": dataset["time"].shape[0],
        "num_datapoints": dataset["time"].shape[1],
    }

    if print_keys:
        print("Dataset keys:")
        for key in dataset.files:
            print(f"- {key}")

    for key in dataset.files:
        arr = dataset[key]
        data[key] = arr[sim_num] if sim_num is not None else arr

    print(f"Data loaded from {dataset_path}.")

    return data

def skew(w, enable_jax=False):
    xp = _xp(enable_jax)
    w1, w2, w3 = w
    w_skew = xp.array([[0, -w3, w2],
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

def quat_to_euler(orient, enable_jax=False):
    """Convert quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw] (ZYX / XYZ intrinsic).

    Args:
        orient (array-like): quaternion in [w, x, y, z] format
        enable_jax (bool): use jax.numpy if True, else numpy

    Returns:
        array of shape (3,): [roll, pitch, yaw] in radians
    """
    xp = _xp(enable_jax)
    w, x, y, z = orient

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = xp.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation); clip to handle numerical noise at the poles
    sinp = 2.0 * (w * y - z * x)
    sinp = xp.clip(sinp, -1.0, 1.0)
    pitch = xp.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = xp.arctan2(siny_cosp, cosy_cosp)

    return xp.array([roll, pitch, yaw])

def quat_to_rot(orient, enable_jax=False):
    """Convert quaternion to rotation matrix (source: https://cookierobotics.com/080/).

    Args:
        orient (np.ndarray | list): quaternion in [w, x, y, z] format

    Returns:
        np.ndarray: corresponding rotation matrix
    """
    xp = _xp(enable_jax)

    w, x, y, z = orient
    row0 = xp.stack([2*(w**2 + x**2) - 1, 2*(x*y - w*z)      , 2*(w*y + x*z)      ])
    row1 = xp.stack([2*(x*y + w*z)      , 2*(w**2 + y**2) - 1, 2*(y*z - w*x)      ])
    row2 = xp.stack([2*(x*z - w*y)      , 2*(y*z + w*x)      , 2*(w**2 + z**2) - 1])
    R = xp.stack([row0, row1, row2])

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

def rmse(gt_value, est_value):
    return np.sqrt(np.mean((gt_value - est_value)**2, axis=0))

def get_inertia_matrix(env):
    M = np.zeros((env.mjModel.nv, env.mjModel.nv)) # shape == (18, 18)
    mujoco.mj_fullM(env.mjModel, M, env.mjData.qM)

    return M

def get_jacobian(env, orient, joint_pos, joint_vel):
    env.mjData.qpos[:] = np.concatenate([np.zeros(shape=(3,)), orient, joint_pos])
    env.mjData.qvel[:] = np.concatenate([np.zeros(shape=(6,)), joint_vel])
    mujoco.mj_forward(env.mjModel, env.mjData)

    lin_jacobian_b = env.feet_jacobians(frame="base")
    lin_jacobian_w = env.feet_jacobians(frame="world")

    return lin_jacobian_b, lin_jacobian_w
