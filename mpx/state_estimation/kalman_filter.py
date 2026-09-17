import numpy as np
from state_estimation.kf_utils import _to_diag
import jax
import jax.numpy as jnp
from functools import partial

class KF():
    """
    Kalman Filter using Leg Odometry as measurement.

    The state consists of:
    - position
    - linear velocity
    - angular velocity
    - contact force
    """

    def __init__(self, dt, A, B, H, Q, R, P0=None, init_state=None):
        self.dt = dt
        self.state_size = A.shape[0]
        meas_size = H.shape[0]

        self.x = np.zeros(self.state_size) if init_state is None else np.array(init_state) # state
        self.Q = _to_diag(Q, self.state_size, "Q") # process noise
        self.R = _to_diag(R, meas_size, "R") # measurement noise
        self.P = np.diag(P0) if P0 is not None else self.Q.copy()

        self.A = A # state transition matrix       
        self.B = B # control input matrix       
        self.H = H # observation matrix

    def update_process_model(self, J_new, coupling, bias):
        """Build A and B for the 'base_acc computed inside KF' mode.
        """
        self.A[3:9, 9:21] = self.dt * J_new
        self.B[3:9, :] = self.dt * np.hstack([coupling, bias])

    def predict(self, u):
        """Prediction step. Estimate robot state based on previous estimation.
        """
        self.x_pred = self.A @ self.x + self.B @ u
        self.P_pred = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        """Update step. Fuse the prediction with the measurement.
        """
        z_tilde = z - self.H @ self.x_pred # residual
        S = self.H @ self.P_pred @ self.H.T + self.R  # residual covariance
        K = self.P_pred @ self.H.T @ np.linalg.inv(S)  # kalman gain

        self.x = self.x_pred + K @ z_tilde
        self.P = (np.eye(self.state_size) - K @ self.H) @ self.P_pred

class KF_JAX():
    """JAX version of KF()
    """

    def __init__(self, dt, state_size, H, Q, R):
        self.dt = jnp.array(dt)
        self.state_size = state_size
        meas_size = H.shape[0]

        self.Q = jnp.asarray(_to_diag(Q, self.state_size, "Q"))
        self.R = jnp.asarray(_to_diag(R, meas_size, "R"))
        self.H = H

    @partial(jax.jit, static_argnums=0)
    def update_process_model(self, A, B, J_new, coupling, bias):
        new_A = A.at[3:9, 9:21].set(self.dt * J_new)
        new_B = B.at[3:9, :].set(self.dt * jnp.hstack([coupling, bias]))

        return new_A, new_B

    @partial(jax.jit, static_argnums=0)
    def predict(self, x, A, u, B, P):
        x_pred = A @ x + B @ u
        P_pred = A @ P @ A.T + self.Q

        return x_pred, P_pred

    @partial(jax.jit, static_argnums=0)
    def update(self, x_pred, P_pred, z):
        z_tilde = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ jnp.linalg.inv(S)

        x = x_pred + K @ z_tilde
        P = (jnp.eye(self.state_size) - K @ self.H) @ P_pred

        return x, P
