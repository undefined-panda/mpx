import numpy as np
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp

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

def build_kinematic_model(dt, include_ang_vel=True, include_contact_force=True, base_acc_source="external"):
    """Returns A, B and H matrix for chosen state layout.
    """
    idx, size, meas_dim = {}, 0, 0
    external_base_acc = base_acc_source == "external"
    control_dim = 0 if external_base_acc else 13

    def add(name, dim, in_control=True, in_measurement=True):
        nonlocal size, control_dim, meas_dim
        idx[name] = slice(size, size + dim)
        size += dim
        if in_control: control_dim += dim
        if in_measurement: meas_dim += dim

    add("pos", 3, in_control=False, in_measurement=False)
    add("lin_vel", 3, in_control=external_base_acc) # lin acc in control
    if include_ang_vel: add("ang_vel", 3, in_control=external_base_acc) # lin acc in control
    if include_contact_force: add("c_force", 12, False)

    A = np.eye(size)
    A[idx["pos"], idx["lin_vel"]] = dt * np.eye(3)
    B = np.zeros((size, control_dim))
    B[idx["lin_vel"], 0:3] = dt * np.eye(3)
    if include_ang_vel: B[idx["ang_vel"], 3:6] = dt * np.eye(3)

    H = np.zeros((meas_dim, size))
    H[0:3, idx["lin_vel"]] = np.eye(3)
    if include_ang_vel: H[3:6, idx["ang_vel"]] = np.eye(3)
    if include_contact_force: H[6:18, idx["c_force"]] = np.eye(12)

    return A, B, H

def _to_diag(val, size, name):
    arr = np.atleast_1d(val).astype(float)
    if arr.size == 1:
        arr = np.full(size, arr.item())
    elif arr.size != size:
        raise ValueError(f"{name} must be scalar or length {size}, got {arr.size}.")
    return np.diag(arr)

def build_noise_vector(noise, sizes):
    arr = []
    for i, val in enumerate(noise):
        if len(val) == 1:
            arr.extend(sizes[i] * val)
        elif len(val) != sizes[i]:
            raise ValueError(f"Noise values at position {i} dont match size value:\n{noise[i]} given, but {sizes[i]} values needed")
        else:
            arr.extend(val)

    return arr

def get_jacobian(env, orient, joint_pos, joint_vel, lin_vel, ang_vel):
    env.mjData.qpos[:] = np.concatenate([np.zeros(shape=(3,)), orient, joint_pos])
    env.mjData.qvel[:] = np.concatenate([lin_vel, ang_vel, joint_vel])
    mujoco.mj_forward(env.mjModel, env.mjData)

    lin_jacobian_b = env.feet_jacobians(frame="base")
    lin_jacobian_w = env.feet_jacobians(frame="world")

    J_b = np.stack([lin_jacobian_b[leg] for leg in env.legs_order])
    J_w = np.stack([lin_jacobian_w[leg] for leg in env.legs_order])
    return J_b, J_w

def get_jacobian_mjx(mjx_model, orient, joint_pos, joint_vel, 
                     lin_vel, ang_vel, foot_geom_ids, foot_body_ids):
    qpos = jnp.concatenate([jnp.zeros(3), orient, joint_pos])
    qvel = jnp.concatenate([lin_vel, ang_vel, joint_vel])
    mjx_data = mjx.make_data(mjx_model).replace(qpos=qpos, qvel=qvel)
    mjx_data = mjx.forward(mjx_model, mjx_data)

    # jacobian per feet
    def one_foot(geom_id, body_id):
        jacp, _ = mjx.jac(mjx_model, mjx_data, mjx_data.geom_xpos[geom_id], body_id)
        return jacp.T
    J_w = jax.vmap(one_foot)(foot_geom_ids, foot_body_ids)

    # in base frame: R_wb^T @ J_w
    R_wb = mjx_data.xmat[1].reshape(3, 3)
    J_b = R_wb.T @ J_w

    return J_b, J_w, mjx_data

def get_inertia_matrix(env):
    M = np.zeros((env.mjModel.nv, env.mjModel.nv)) # shape == (18, 18)
    mujoco.mj_fullM(env.mjModel, M, env.mjData.qM)

    return M

def get_inertia_matrix_mjx(mjx_model, data):
    return mjx.full_m(mjx_model, data)

def quat_to_rot(orient, xp=np):
    """Convert quaternion to rotation matrix (source: https://cookierobotics.com/080/).
    """

    w, x, y, z = orient
    row0 = xp.stack([2*(w**2 + x**2) - 1, 2*(x*y - w*z)      , 2*(w*y + x*z)      ])
    row1 = xp.stack([2*(x*y + w*z)      , 2*(w**2 + y**2) - 1, 2*(y*z - w*x)      ])
    row2 = xp.stack([2*(x*z - w*y)      , 2*(y*z + w*x)      , 2*(w**2 + z**2) - 1])
    R = xp.stack([row0, row1, row2])

    return R

def rot_to_quat(orient, xp=np):
    """Convert rotation matrix to quaternion (Shepherd's method, numerically robust).
    """

    R = orient
    trace = R[0,0] + R[1,1] + R[2,2]

    if trace > 0:
        s = 0.5 / xp.sqrt(trace + 1.0)
        return xp.array([0.25 / s,
                         (R[2,1] - R[1,2]) * s,
                         (R[0,2] - R[2,0]) * s,
                         (R[1,0] - R[0,1]) * s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * xp.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return xp.array([(R[2,1] - R[1,2]) / s,
                         0.25 * s,
                         (R[0,1] + R[1,0]) / s,
                         (R[0,2] + R[2,0]) / s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * xp.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return xp.array([(R[0,2] - R[2,0]) / s,
                         (R[0,1] + R[1,0]) / s,
                         0.25 * s,
                         (R[1,2] + R[2,1]) / s])
    else:
        s = 2.0 * xp.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return xp.array([(R[1,0] - R[0,1]) / s,
                         (R[0,2] + R[2,0]) / s,
                         (R[1,2] + R[2,1]) / s,
                         0.25 * s])

def convert_orient(orient, xp=np):
    """Return orient as quaternion ans rotation matrix.
    """
    if orient.shape == (4,):
        return orient, quat_to_rot(orient, xp)
    else:
        return rot_to_quat(orient, xp), orient

def skew(w, xp=np):
    w1, w2, w3 = w
    return xp.array([[0, -w3, w2],
                     [w3, 0, -w1],
                     [-w2, w1, 0]])
