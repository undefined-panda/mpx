import numpy as np

class GMContactObserver:
    def __init__(self, dt, L1, L2, thresholds):
        self.p_hat = np.zeros(18) # 6 DOF base + 12 DOF legs
        self.f_hat = np.zeros(12) # 3x1 vector for 4 feet
        self.f_hat_prev = np.zeros(12)
        self.dt = dt
        self.L1 = L1
        self.L2 = L2
        self.thresholds = np.array(thresholds)
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

        contact_state = f_filtered[2::3] < self.thresholds
        return contact_state, f_filtered
