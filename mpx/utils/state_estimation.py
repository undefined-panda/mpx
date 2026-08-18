import numpy as np
from utils.kf_utils import skew
import jax
import jax.numpy as jnp

class KF():
    """
    Kalman Filter using Leg Odometry as measurement.

    The linear state consists of:
    - position
    - linear velocity
    - angular velocity
    - contact force (flattened, one 3-vector per foot)

    Orientation itself is tracked separately as a 3x3 rotation matrix (`self.orient`)
    since it does not evolve linearly and cannot be part of a linear state vector.
    - removed in current state

    Depending on estimation mode, different A, B and H matrices are defined. The estimation 
    modes determine which values are given and which need to be estimated.

    The control input can be base accelaration or joint acceleration, used to estimate the 
    base acceleration inside the filter via the robot dynamics.

    The measurements are:
    - orientation coming from the IMU
    - linear velocity coming from leg odometry
    - angular velocity coming from the IMU
    - contact force
    """

    def __init__(self, dt, Q_diag, R_diag, est_mode=1):
        self.dt = dt
        self.est_mode = est_mode

        # state
        self.POS = slice(0, 3)
        self.LIN_VEL = slice(3, 6)
        self.ANG_VEL = slice(6, 9)
        self.C_FORCE = slice(9, 21)

        if est_mode == 1:
            self.x = np.zeros(9)
        if est_mode in [2,3,4]:
            self.x = np.zeros(21)

        # state transition matrix
        self.A = np.eye(self.x.shape[0])
        self.A[self.POS, self.LIN_VEL] = self.dt * np.eye(3)

        # control input matrix
        if est_mode == 1: # u = base acc
            self.B = np.eye(9)[:, 3:9] * self.dt
        if est_mode in [2,3]:
            self.B = np.eye(21)[:, 3:9] * self.dt
        if est_mode == 4: # base acc is inside KF
            self.B = np.zeros((21,13))

        # observation matrix (no measurement for position used)
        if est_mode == 1:
            self.H = np.eye(9)[3:9, :]
        if est_mode in [2,3,4]:
            self.H = np.eye(21)[3:21, :]

        # lowering a value means putting more trust into that part. Q = process noise, R = measurement noise
        if est_mode == 1: # pos, lin and ang vel have each 3 values
            self.Q = np.diag(np.repeat(Q_diag, 3))
            self.R = np.diag(np.repeat(R_diag[1:], 3))
        if est_mode in [2,3,4]: # contact force has 12
            self.Q = np.diag(np.concatenate([np.repeat(Q_diag[:3], 3), np.repeat(Q_diag[3], 12)]))
            self.R = np.diag(np.concatenate([np.repeat(R_diag[1:3], 3), np.repeat(R_diag[3], 12)]))

        self.P = self.Q.copy()  # error covariance

        self.filter_states = ["predict", "update"]

    def get_filter_state(self, filter_state, orient=False):
        if filter_state not in self.filter_states:
            raise ValueError(f"Invalid filter state: {filter_state}")

        if orient:
            state = self.orient_pred if filter_state == "predict" else self.orient
        else:
            state = self.x_pred if filter_state == "predict" else self.x

        return state

    def get_pos(self, filter_state="update"): return self.get_filter_state(filter_state)[self.POS]

    def get_lin_vel(self, filter_state="update"): return self.get_filter_state(filter_state)[self.LIN_VEL]

    def get_ang_vel(self, filter_state="update"): return self.get_filter_state(filter_state)[self.ANG_VEL]

    def get_contact_force(self, filter_state="update"): return self.get_filter_state(filter_state)[self.C_FORCE].reshape((4, 3))

    def get_orient(self, filter_state="update"): return self.get_filter_state(filter_state, orient=True)
    
    def update_A_B_contact_forces(self, orient, contact_pos_b, contact_state, M, qfrc_bias):
        # qfrc_bias = env.mjData.qfrc_bias  # (18,)
        # M = np.zeros((env.mjModel.nv, env.mjModel.nv))
        # mujoco.mj_fullM(env.mjModel, M, env.mjData.qM)
        
        H_B = M[:6, :6]
        H_BL = M[:6, 6:18]
        
        J_full = np.zeros((6, 12)) # 6 DoF base, 4x3=12 contact forces
        for i in range(4):
            if contact_state[i]:
                cp_i = orient @ contact_pos_b[i]
                J_full[:3, i*3:(i+1)*3] = np.eye(3)
                J_full[3:, i*3:(i+1)*3] = skew(cp_i)
        
        J_new = np.linalg.solve(H_B, J_full)
        temp1 = np.linalg.solve(H_B, -H_BL)
        temp2 = np.linalg.solve(H_B, -qfrc_bias[:6].reshape(-1, 1))
        
        self.A[3:9, self.C_FORCE] = self.dt * J_new
        self.B[3:9, :] = self.dt * np.hstack([temp1, temp2])

    def update_A_contact_force(self, contact_state):
        self.A[self.C_FORCE, self.C_FORCE] = np.diag(np.repeat(contact_state, 3))

    def predict(self, u):
        """Prediction step. Estimate robot state based on previous estimation.
        """

        self.x_pred = self.A @ self.x + self.B @ u
        self.P_pred = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        """Update step. Fuse the prediction with the measurement.
        """

        if self.est_mode in [1, 2, 3, 4]: # measurement for lin, ang vel and contact force
            z_tilde = z - self.H @ self.x_pred

        S = self.H @ self.P_pred @ self.H.T + self.R  # residual covariance
        K = self.P_pred @ self.H.T @ np.linalg.inv(S)  # kalman gain

        self.x = self.x_pred + K @ z_tilde
        self.P = (np.eye(K.shape[0]) - K @ self.H) @ self.P_pred

class KF_JAX():
    def __init__(self, dt, Q_diag, R_diag, est_mode=1):
        self.dt = dt
        self.est_mode = est_mode

        # state
        self.POS = slice(0, 3)
        self.LIN_VEL = slice(3, 6)
        self.ANG_VEL = slice(6, 9)
        self.C_FORCE = slice(9, 21)

        if est_mode == 1:
            self.x = jnp.zeros(9)
        if est_mode in [2,3,4]:
            self.x = jnp.zeros(21)

        # A
        A = jnp.eye(self.x.shape[0])
        A = A.at[self.POS, self.LIN_VEL].set(self.dt * jnp.eye(3))
        self.A = A

        # B
        if est_mode == 1: # u = base acc
            self.B = jnp.eye(9)[:, 3:9] * self.dt
        if est_mode in [2,3]:
            self.B = jnp.eye(21)[:, 3:9] * self.dt
        if est_mode == 4: # base acc is inside KF
            self.B = jnp.zeros((21,13))

        # H
        if est_mode == 1:
            self.H = jnp.eye(9)[3:9, :]
        if est_mode in [2,3,4]:
            self.H = jnp.eye(21)[3:21, :]

        # Q, R
        if est_mode == 1:
            self.Q = jnp.diag(jnp.repeat(jnp.asarray(Q_diag), 3))
            self.R = jnp.diag(jnp.repeat(jnp.asarray(R_diag)[1:], 3))
        if est_mode in [2,3,4]:
            Q_diag = jnp.asarray(Q_diag)
            R_diag = jnp.asarray(R_diag)
            self.Q = jnp.diag(jnp.concatenate([jnp.repeat(Q_diag[:3], 3), jnp.repeat(Q_diag[3], 12)]))
            self.R = jnp.diag(jnp.concatenate([jnp.repeat(R_diag[1:3], 3), jnp.repeat(R_diag[3], 12)]))

        self.P = self.Q
        self.filter_states = ["predict", "update"]
        self._predict_jit = jax.jit(self._predict_impl)
        self._update_jit = jax.jit(self._update_impl)
        self._update_AB_jit = jax.jit(self._update_AB_impl, static_argnames=("use_full_M",))

    def get_filter_state(self, filter_state, orient=False):
        if filter_state not in self.filter_states:
            raise ValueError(f"Invalid filter state: {filter_state}")

        if orient:
            return self.orient_pred if filter_state == "predict" else self.orient
        return self.x_pred if filter_state == "predict" else self.x

    def get_pos(self, filter_state="update"): return self.get_filter_state(filter_state)[self.POS]

    def get_lin_vel(self, filter_state="update"): return self.get_filter_state(filter_state)[self.LIN_VEL]

    def get_ang_vel(self, filter_state="update"): return self.get_filter_state(filter_state)[self.ANG_VEL]

    def get_contact_force(self, filter_state="update"): return self.get_filter_state(filter_state)[self.C_FORCE].reshape((4, 3))

    def get_orient(self, filter_state="update"): return self.get_filter_state(filter_state, orient=True)
    
    @staticmethod
    def _update_AB_impl(dt, A, B, orient, contact_pos_b, contact_state, M, qfrc_bias, use_full_M: bool):
        H_B = M[:6, :6]
        H_BL = M[:6, 6:18] if use_full_M else None

        cs = jnp.asarray(contact_state).astype(orient.dtype)
        cp_w = contact_pos_b @ orient.T
        z = jnp.zeros(4, dtype=cp_w.dtype)
        S = jnp.stack([
            jnp.stack([z,            -cp_w[:, 2],  cp_w[:, 1]], axis=1),
            jnp.stack([ cp_w[:, 2],   z,          -cp_w[:, 0]], axis=1),
            jnp.stack([-cp_w[:, 1],   cp_w[:, 0],  z          ], axis=1),
        ], axis=1)
        eye3 = jnp.broadcast_to(jnp.eye(3), (4, 3, 3))
        blocks = jnp.concatenate([eye3, S], axis=1) * cs[:, None, None]
        J_full = jnp.concatenate([blocks[i] for i in range(4)], axis=1)

        if use_full_M:
            rhs = jnp.concatenate([J_full, -H_BL, -qfrc_bias[:6].reshape(-1, 1)], axis=1)
            sol = jnp.linalg.solve(H_B, rhs)
            J_new, temp1, temp2 = sol[:, :12], sol[:, 12:24], sol[:, 24:25]
        else:
            rhs = jnp.concatenate([J_full, -qfrc_bias[:6].reshape(-1, 1)], axis=1)
            sol = jnp.linalg.solve(H_B, rhs)
            J_new, temp2 = sol[:, :12], sol[:, 12:13]
            temp1 = jnp.zeros((6, 12))

        A_new = A.at[3:9, 9:21].set(dt * J_new)
        B_new = B.at[3:9, :].set(dt * jnp.hstack([temp1, temp2]))
        return A_new, B_new

    def update_A_B_contact_forces(self, orient, contact_pos_b, contact_state, M, qfrc_bias):
        use_full_M = bool(M.shape[1] > 6)
        self.A, self.B = self._update_AB_jit(
            self.dt, self.A, self.B, orient, contact_pos_b, contact_state, M, qfrc_bias, use_full_M=use_full_M,
        )

    @staticmethod
    def _update_A_cf_impl(A, contact_state):
        return A.at[9:21, 9:21].set(jnp.diag(jnp.repeat(contact_state.astype(A.dtype), 3)))

    def update_A_contact_force(self, contact_state):
        if not hasattr(self, "_update_A_cf_jit"):
            self._update_A_cf_jit = jax.jit(self._update_A_cf_impl)
        self.A = self._update_A_cf_jit(self.A, jnp.asarray(contact_state))

    @staticmethod
    def _predict_impl(A, B, Q, P, x, u):
        x_pred = A @ x + B @ u
        P_pred = A @ P @ A.T + Q
        return x_pred, P_pred

    @staticmethod
    def _update_impl(H, R, P_pred, x_pred, z):
        z_tilde = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = jnp.linalg.solve(S.T, (P_pred @ H.T).T).T
        x = x_pred + K @ z_tilde
        P = (jnp.eye(K.shape[0]) - K @ H) @ P_pred
        return x, P

    def predict(self, u):
        self.x_pred, self.P_pred = self._predict_jit(self.A, self.B, self.Q, self.P, self.x, u)

    def update(self, z):
        self.x, self.P = self._update_jit(self.H, self.R, self.P_pred, self.x_pred, z)
