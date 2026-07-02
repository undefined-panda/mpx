import numpy as np
import mujoco
from utils.kf_utils import matrix_exp, matrix_log, skew


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

        self.Q = np.diag(Q_diag * np.ones(self.x.shape[0])) # process noise
        self.R = np.diag(R_diag * np.ones(self.H.shape[0])) # measurement noise
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
