import numpy as np
import mujoco
from utils.leg_odometry import LegOdom

class EKF():
    def __init__(self, init_pos, dt, Q_diag=1e-2, R_diag=1e-1):
        self.dt = dt

        # Kalman Filter matrices
        self.A = np.concatenate([np.concatenate([np.eye(3), dt * np.eye(3)], axis=1), np.concatenate([np.zeros((3,3)), np.eye(3)], axis=1)]) # state transition matrix
        self.B = np.concatenate([np.zeros(3), np.full(3, self.dt)]) # control matrix
        self.H = np.concatenate([np.zeros((3,3)), np.eye(3)], axis=1) # observation matrix

        # Kalman Filter noise
        self.Q = np.diag(Q_diag*np.ones(6)) # process noise
        self.R = np.diag(R_diag*np.ones(3)) # measurement noise

        self.x = np.concatenate([init_pos, np.zeros(3)]) # state values (in world coordinates)
        self.P = self.Q # error covariance
        self.leg_odom = LegOdom(init_state=self.x)
    
    def step(self, base_orient, base_acc, base_ang_vel, joint_pos, joint_vel, contact_states, contact_forces, contact_pos):
        
        self.predict(base_acc)

        z = self.calc_leg_odometry(base_orient, base_ang_vel, joint_pos, joint_vel, contact_states, contact_forces, contact_pos) # measurement, coming from leg odometry
        self.z = z

        self.update(z) # update step with leg odometry result as input

    def predict(self, acc):
        """
        Estimate robot movement based on information from the legs.

        :param u: Control input vector (acceleration of the base)
        """
        u = np.concatenate([np.zeros(3), acc])
        try:
            x_pred = self.A @ self.x + self.B * u
            P_pred = self.A @ self.P @ self.A.T + self.Q
        except ValueError as e:
            print("Error in predict", e)

        self.x_pred = x_pred
        self.P_pred = P_pred

    def calc_leg_odometry(self, base_orient, base_ang_vel, joint_pos, joint_vel, contact_states, contact_forces, contact_pos):
        """
        Create measurement from leg odometry, based on equations of SLAM Handbook Ch. 12
        
        :param base_pos: Description
        :param base_orient: Description
        :param base_ang_vel: Description
        :param joint_pos: Description
        :param joint_vel: Description
        :param contact_states: Description
        :param contact_forces: Description
        :param contact_pos: Description
        """

        # set mjData pos
        qpos = np.concatenate([np.zeros(shape=(3,)), base_orient, joint_pos])
        self.leg_odom.env.mjData.qpos[:] = qpos
        
        # calculate lineare jacobian
        mujoco.mj_forward(self.leg_odom.env.mjModel, self.leg_odom.env.mjData)
        self.leg_odom.lin_jacobian = self.leg_odom.env.feet_jacobians(frame="base") # use mj_jac from MuJoCo, defined in QuadrupedEnv

        # estimate state with leg odometry
        self.leg_odom.calc_leg_odometry(dt=self.dt,
                                        base_orient=base_orient,
                                        base_ang_vel=base_ang_vel,
                                        qdot=joint_vel,
                                        contact_state=contact_states,
                                        contact_force=contact_forces,
                                        contact_pos=contact_pos)
        
        return self.leg_odom.state.vel

    def update(self, z):
        """
        Update estimation with consideration of measurement. Measurement comes from leg odometry.
        
        :param z: measurement
        """

        z_tilde = z - self.H @ self.x_pred # measurement residual
        S = self.H @ self.P_pred @ self.H.T + self.R # residual covariance
        K = self.P_pred @ self.H.T @ np.linalg.inv(S) # kalman gain
        
        x_update = self.x_pred + K @ z_tilde
        P_update = (np.eye(6) - K @ self.H) @ self.P_pred

        self.x = x_update
        self.P = P_update
