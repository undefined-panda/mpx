import numpy as np
from typing import NamedTuple

class GMState(NamedTuple):
    p_hat: any      # (18,)
    f_hat: any      # (12,)
    f_hat_prev: any # (12,)

class GMContactObserver:
    def __init__(self, dt, L1, L2, thresholds, xp=np):
        self.xp = xp
        self.dt = dt
        self.L1 = L1
        self.L2 = L2
        self.alpha = 0.8
        self.thresholds = xp.asarray(thresholds)
        self.state = GMState(p_hat=xp.zeros(18), # 6 DOF base + 12 DOF legs
                             f_hat=xp.zeros(12), # 3x1 vector for 4 feet
                             f_hat_prev=xp.zeros(12),)
    
    def _q(self, s):
        return self.xp.sign(s) * self.xp.sqrt(self.xp.abs(s)) + s

    def _k1(self, s):
        return self._q(s)
    
    def _k2(self, s):
        return self.xp.sign(s) + self._q(s)

    def step(self, state, vel, M, joint_torque, J, qfrc_bias):
        xp = self.xp

        torque = xp.concatenate([xp.zeros(6), joint_torque]) # [:6] = 0 since base is floating
        p_measured = M @ vel
        tau_bar = torque - qfrc_bias
        innovation = p_measured - state.p_hat 
        p_hat_dot = - J.T @ state.f_hat + tau_bar + self.L1 * self._k1(innovation) 
        f_hat_dot = self.L2 * self._k2(innovation[6:])

        p_hat = state.p_hat + p_hat_dot * self.dt
        f_hat = state.f_hat + f_hat_dot * self.dt

        # z-axis constraint and positive clipping
        f_hat = f_hat.reshape(4, 3)
        f_hat = xp.stack([xp.zeros(4), xp.zeros(4), xp.maximum(f_hat[:, 2], 0.0),], axis=1).reshape(12)

        f_filtered = self.alpha * state.f_hat_prev + (1 - self.alpha) * f_hat
        new_state = GMState(p_hat=p_hat, f_hat=f_hat, f_hat_prev=f_filtered)

        contact_state = f_filtered[2::3] < self.thresholds
        return new_state, contact_state, f_filtered
