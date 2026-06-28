import numpy as np
import mujoco
import jax
import jax.numpy as jnp
from utils.kf_utils import matrix_exp, matrix_log, skew, matrix_exp_jax, matrix_log_jax


class KF():
    """
    Kalman Filter using Leg Odometry as measurement.

    The linear state consists of:
    - position
    - linear velocity
    - angular velocity
    - contact force (flattened, one 3-vector per foot)
    - attitude error: a 3-dim error-state placeholder used only to couple the
      orientation correction into the shared Kalman gain. It is reset to zero
      after every update (standard error-state/multiplicative EKF pattern).

    Orientation itself is tracked separately as a 3x3 rotation matrix (`self.orient`)
    since it does not evolve linearly and cannot be part of a linear state vector.

    The control input is the joint acceleration (+ a constant bias term), used to
    estimate the base acceleration inside the filter via the robot dynamics.

    The measurements are:
    - orientation coming from the IMU
    - linear velocity coming from leg odometry
    - angular velocity coming from the IMU
    - contact force
    """

    def __init__(self, dt, Q_diag, R_diag, contact_coupling="dynamics", contact_force_decay="constant"):
        """
        Args:
            contact_coupling (str): how contact force couples into lin_vel/ang_vel in A/B:
                - "dynamics": mass-matrix-weighted contact Jacobian (physically correct, default)
                - "direct": geometric contact Jacobian without the mass-matrix weighting
                  (contact force enters with coefficient 1 instead of pinv(H_B))
                - "identity": contact_state-gated identity on lin_vel only, no angular
                  coupling and no geometric leverage at all
            contact_force_decay (str): behaviour of the C_FORCE->C_FORCE block of A
                ("the bottom-right corner"):
                - "constant": contact force prediction persists unchanged (default)
                - "contact_gated": contact force prediction decays to zero for feet
                  that are currently not in contact
        """

        self.dt = dt
        self.contact_coupling = contact_coupling
        self.contact_force_decay = contact_force_decay

        # state
        self.POS = slice(0, 3)
        self.LIN_VEL = slice(3, 6)
        self.ANG_VEL = slice(6, 9)
        self.C_FORCE = slice(9, 21)
        self.ATT_ERR = slice(21, 24)
        self.DYN = slice(3, 9)  # lin_vel + ang_vel, coupled to contact forces

        self.x = np.zeros(24)
        self.orient = np.eye(3)  # nominal orientation, kept outside the linear state

        # state transition matrix
        self.A = np.eye(self.x.shape[0])
        self.A[self.POS, self.LIN_VEL] = self.dt * np.eye(3)

        # control input matrix: 12 joint accelerations + 1 bias term
        self.B = np.zeros((self.x.shape[0], 13))

        # observation matrix: z = [att_err(3), lin_vel(3), ang_vel(3), c_force(12)]
        self.H = np.zeros((21, self.x.shape[0]))
        self.H[0:3, self.ATT_ERR] = np.eye(3)
        self.H[3:6, self.LIN_VEL] = np.eye(3)
        self.H[6:9, self.ANG_VEL] = np.eye(3)
        self.H[9:21, self.C_FORCE] = np.eye(12)

        self.Q = np.diag(Q_diag * np.ones(self.x.shape[0]))  # process noise
        self.R = np.diag(R_diag * np.ones(self.H.shape[0]))  # measurement noise
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

    def update_A_B_contact_forces(self, env, orient, contact_pos_b, contact_state):
        """Update the contact-force coupling in A/B, since the mass distribution / contact
        Jacobian changes every step.

        Args:
            env: simulation environment providing the mujoco model/data
            orient (np.ndarray): current orientation estimate (3x3), used to rotate
                contact positions from base to world frame
            contact_pos_b (np.ndarray): foot positions in base frame, shape (4, 3)
            contact_state (np.ndarray): boolean/float contact state per foot, shape (4,)
        """

        need_mass_matrix = self.contact_coupling == "dynamics"
        if need_mass_matrix:
            qfrc_bias = env.mjData.qfrc_bias  # coriolis + gravitational terms per DOF, shape == (18,)

            M = np.zeros((env.mjModel.nv, env.mjModel.nv))  # shape == (18, 18)
            mujoco.mj_fullM(env.mjModel, M, env.mjData.qM)

            H_B = M[:6, :6]
            H_BL = M[:6, 6:18]
            H_B_inv = np.linalg.pinv(H_B)

        J_a = []  # upper part, only for linear acceleration
        J_b = []  # lower part, only for angular acceleration
        for i in range(4):
            cp_i = orient @ contact_pos_b[i]
            if self.contact_coupling == "dynamics":
                J_i = np.vstack([np.eye(3), skew(cp_i)])
                block = contact_state[i] * (H_B_inv @ J_i)
            elif self.contact_coupling == "direct":
                block = contact_state[i] * np.vstack([np.eye(3), skew(cp_i)])
            elif self.contact_coupling == "identity":
                block = contact_state[i] * np.vstack([np.eye(3), np.zeros((3, 3))])
            else:
                raise ValueError(f"Unknown contact_coupling mode: {self.contact_coupling}")
            J_a.append(block[:3, :])
            J_b.append(block[3:, :])

        J_new = np.vstack([np.hstack(J_a), np.hstack(J_b)])
        self.A[self.DYN, self.C_FORCE] = self.dt * J_new

        if need_mass_matrix:
            self.B[self.DYN, 0:12] = self.dt * (H_B_inv @ (-H_BL))
            self.B[self.DYN, 12] = self.dt * (H_B_inv @ (-qfrc_bias[:6]))
        else:
            # the joint-acc/bias coupling is derived from the mass matrix and is only
            # meaningful together with the "dynamics" contact_coupling mode
            self.B[self.DYN, :] = 0.0

        if self.contact_force_decay == "contact_gated":
            self.A[self.C_FORCE, self.C_FORCE] = np.diag(np.repeat(contact_state, 3))

    def predict(self, u):
        """Prediction step. Estimate robot state based on previous estimation.

        Args:
            u (np.ndarray): control input, [joint_acc(12), bias(1)]
        """

        self.x_pred = self.A @ self.x + self.B @ u
        self.P_pred = self.A @ self.P @ self.A.T + self.Q
        self.orient_pred = self.orient @ matrix_exp(self.x_pred[self.ANG_VEL], self.dt)

    def update(self, z):
        """Update step. Fuse the prediction with the measurement.

        Args:
            z (np.ndarray): measurement, [orient(9, flattened rotation matrix),
                lin_vel(3), ang_vel(3), c_force(12)]
        """

        z_orient = z[0:9].reshape((3, 3))
        z_tilde_orient = matrix_log(self.orient_pred.T @ z_orient)  # body-frame orientation error
        z_tilde_lin = z[9:27] - self.H[3:21, :] @ self.x_pred
        z_tilde = np.concatenate([z_tilde_orient, z_tilde_lin])

        S = self.H @ self.P_pred @ self.H.T + self.R  # residual covariance
        K = self.P_pred @ self.H.T @ np.linalg.inv(S)  # kalman gain

        self.x = self.x_pred + K @ z_tilde
        self.orient = self.orient_pred @ matrix_exp(self.x[self.ATT_ERR], 1)
        self.x[self.ATT_ERR] = 0.0  # reset error state, now folded into self.orient
        self.P = (np.eye(K.shape[0]) - K @ self.H) @ self.P_pred


# ---------------------------------------------------------------------------
# JAX equivalent of KF, for switching the estimation pipeline to a JAX backend. Same
# state layout/update equations as KF, but predict/update are jax.jit-compiled pure
# functions operating on jax arrays. update_A_B_contact_forces still relies on MuJoCo
# (NumPy-only, mj_fullM has no JAX equivalent), so it stays NumPy internally and only
# converts its results into jax arrays before writing them into self.A/self.B.
# ---------------------------------------------------------------------------

_ANG_VEL = slice(6, 9)
_ATT_ERR = slice(21, 24)

@jax.jit
def _kf_predict_jax(A, B, x, P, Q, u, orient, dt):
    x_pred = A @ x + B @ u
    P_pred = A @ P @ A.T + Q
    orient_pred = orient @ matrix_exp_jax(x_pred[_ANG_VEL], dt)
    return x_pred, P_pred, orient_pred

@jax.jit
def _kf_update_jax(H, R, x_pred, P_pred, orient_pred, z):
    z_orient = z[0:9].reshape((3, 3))
    z_tilde_orient = matrix_log_jax(orient_pred.T @ z_orient)  # body-frame orientation error
    z_tilde_lin = z[9:27] - H[3:21, :] @ x_pred
    z_tilde = jnp.concatenate([z_tilde_orient, z_tilde_lin])

    S = H @ P_pred @ H.T + R  # residual covariance
    K = P_pred @ H.T @ jnp.linalg.inv(S)  # kalman gain

    x_new = x_pred + K @ z_tilde
    orient_new = orient_pred @ matrix_exp_jax(x_new[_ATT_ERR], 1.0)
    x_new = x_new.at[_ATT_ERR].set(0.0)  # reset error state, now folded into orient_new
    P_new = (jnp.eye(K.shape[0]) - K @ H) @ P_pred

    return x_new, orient_new, P_new

class KFJax():
    """JAX equivalent of KF (see KF for the full documentation/algorithm description)."""

    def __init__(self, dt, Q_diag, R_diag, contact_coupling="dynamics", contact_force_decay="constant"):
        self.dt = dt
        self.contact_coupling = contact_coupling
        self.contact_force_decay = contact_force_decay

        # state
        self.POS = slice(0, 3)
        self.LIN_VEL = slice(3, 6)
        self.ANG_VEL = slice(6, 9)
        self.C_FORCE = slice(9, 21)
        self.ATT_ERR = slice(21, 24)
        self.DYN = slice(3, 9)  # lin_vel + ang_vel, coupled to contact forces

        self.x = jnp.zeros(24)
        self.orient = jnp.eye(3)  # nominal orientation, kept outside the linear state

        # state transition matrix
        self.A = jnp.eye(self.x.shape[0])
        self.A = self.A.at[self.POS, self.LIN_VEL].set(self.dt * jnp.eye(3))

        # control input matrix: 12 joint accelerations + 1 bias term
        self.B = jnp.zeros((self.x.shape[0], 13))

        # observation matrix: z = [att_err(3), lin_vel(3), ang_vel(3), c_force(12)]
        self.H = jnp.zeros((21, self.x.shape[0]))
        self.H = self.H.at[0:3, self.ATT_ERR].set(jnp.eye(3))
        self.H = self.H.at[3:6, self.LIN_VEL].set(jnp.eye(3))
        self.H = self.H.at[6:9, self.ANG_VEL].set(jnp.eye(3))
        self.H = self.H.at[9:21, self.C_FORCE].set(jnp.eye(12))

        self.Q = jnp.diag(Q_diag * jnp.ones(self.x.shape[0]))  # process noise
        self.R = jnp.diag(R_diag * jnp.ones(self.H.shape[0]))  # measurement noise
        self.P = self.Q  # error covariance

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

    def update_A_B_contact_forces(self, env, orient, contact_pos_b, contact_state):
        """See KF.update_A_B_contact_forces. MuJoCo's mass-matrix extraction is
        unavoidably NumPy (mj_fullM has no JAX equivalent), so the Jacobian/mass-matrix
        coupling is computed in NumPy here and only the final A/B updates are converted
        into jax arrays.
        """

        orient_np = np.asarray(orient)
        contact_pos_b_np = np.asarray(contact_pos_b)
        contact_state_np = np.asarray(contact_state)

        need_mass_matrix = self.contact_coupling == "dynamics"
        if need_mass_matrix:
            qfrc_bias = env.mjData.qfrc_bias  # coriolis + gravitational terms per DOF, shape == (18,)

            M = np.zeros((env.mjModel.nv, env.mjModel.nv))  # shape == (18, 18)
            mujoco.mj_fullM(env.mjModel, M, env.mjData.qM)

            H_B = M[:6, :6]
            H_BL = M[:6, 6:18]
            H_B_inv = np.linalg.pinv(H_B)

        J_a = []  # upper part, only for linear acceleration
        J_b = []  # lower part, only for angular acceleration
        for i in range(4):
            cp_i = orient_np @ contact_pos_b_np[i]
            if self.contact_coupling == "dynamics":
                J_i = np.vstack([np.eye(3), skew(cp_i)])
                block = contact_state_np[i] * (H_B_inv @ J_i)
            elif self.contact_coupling == "direct":
                block = contact_state_np[i] * np.vstack([np.eye(3), skew(cp_i)])
            elif self.contact_coupling == "identity":
                block = contact_state_np[i] * np.vstack([np.eye(3), np.zeros((3, 3))])
            else:
                raise ValueError(f"Unknown contact_coupling mode: {self.contact_coupling}")
            J_a.append(block[:3, :])
            J_b.append(block[3:, :])

        J_new = np.vstack([np.hstack(J_a), np.hstack(J_b)])
        self.A = self.A.at[self.DYN, self.C_FORCE].set(self.dt * jnp.asarray(J_new))

        if need_mass_matrix:
            self.B = self.B.at[self.DYN, 0:12].set(self.dt * jnp.asarray(H_B_inv @ (-H_BL)))
            self.B = self.B.at[self.DYN, 12].set(self.dt * jnp.asarray(H_B_inv @ (-qfrc_bias[:6])))
        else:
            # the joint-acc/bias coupling is derived from the mass matrix and is only
            # meaningful together with the "dynamics" contact_coupling mode
            self.B = self.B.at[self.DYN, :].set(0.0)

        if self.contact_force_decay == "contact_gated":
            self.A = self.A.at[self.C_FORCE, self.C_FORCE].set(jnp.diag(jnp.repeat(jnp.asarray(contact_state_np), 3)))

    def predict(self, u):
        """See KF.predict."""

        self.x_pred, self.P_pred, self.orient_pred = _kf_predict_jax(
            self.A, self.B, self.x, self.P, self.Q, jnp.asarray(u), self.orient, self.dt)

    def update(self, z):
        """See KF.update."""

        self.x, self.orient, self.P = _kf_update_jax(
            self.H, self.R, self.x_pred, self.P_pred, self.orient_pred, jnp.asarray(z))
