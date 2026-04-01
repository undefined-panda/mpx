import numpy as np
from utils.leg_odometry import LegOdom
from utils.dynamics_model import estimate_acc_from_contact_force

class EKF():
    """
    This class imeplemnts a Kalman Filter using Leg Odometry as measurement.
    """

    def __init__(self, init_pos, dt, Q_diag=1e-2, R_diag=[1,1,1]):
        self.dt = dt

        # Kalman Filter matrices
        self.A = np.concatenate([np.concatenate([np.eye(3), self.dt * np.eye(3)], axis=1), np.concatenate([np.zeros((3,3)), np.eye(3)], axis=1)]) # state transition matrix
        self.B = np.concatenate([np.zeros((3,3)), self.dt*np.eye(3)],axis=0) # control matrix
        self.H = np.concatenate([np.zeros((3,3)), np.eye(3)], axis=1) # observation matrix

        # Kalman Filter noise
        self.Q = np.diag(Q_diag*np.ones(6)) # process noise. lower value means trusting the model more
        self.R = np.diag(np.array(R_diag)) # measurement noise. lower value means trusting the measurement more

        # State
        self.x = np.concatenate([init_pos, np.zeros(3)]) # state values (in world coordinates)
        self.P = self.Q # error covariance

        self.leg_odom = LegOdom(init_state=self.x)
    
    def step(self, base_orient, base_acc, base_ang_vel, joint_pos, joint_vel, joint_torque, contact_states, contact_forces, contact_pos, contact_state_threshold) -> None:
        """Kalman Filter algorithmic loop. Alternating between prediction step, gathering new measurement and update step.

        Args:
            base_orient (np.ndarray): base orientation as quaternion in [w, x, y, z] format
            base_acc (np.ndarray | None): base acceleration
            base_ang_vel (np.ndarray): base angular velocity
            joint_pos (np.ndarray): joint position
            joint_vel (np.ndarray): joint velocity
            joint_torque (np.ndarray): joint torque
            contact_states (np.ndarray): contact state of legs with ground
            contact_forces (np.ndarray): force acting on the contact point during stance phase
            contact_pos (np.ndarray): leg position during contact
            contact_state_threshold (int): contact force threshold indicating contact with the ground 
        """

        # measurement, coming from leg odometry
        self.leg_odom.compute_leg_odometry(dt=self.dt,
                                           base_orient=base_orient,
                                           base_ang_vel=base_ang_vel,
                                           qdot=joint_vel,
                                           joint_torque=joint_torque,
                                           joint_pos=joint_pos,
                                           contact_state=contact_states,
                                           contact_force=contact_forces,
                                           contact_state_threshold=contact_state_threshold)
        
        self.z = self.leg_odom.state.vel
        self.leg_odom_pos = self.leg_odom.state.pos
        self.c_force = self.leg_odom.contact_forces
        self.c_state = self.leg_odom.contact_states

        # estimate base acceleration with dynamics model
        if base_acc is None:
            body_mass = float(np.sum(self.leg_odom.env.mjModel.body_mass))
            base_acc = estimate_acc_from_contact_force(m=body_mass,
                                                       contact_states=self.c_state,
                                                       contact_forces=self.c_force)
        
        self.base_acc = base_acc

        # prediction step with base acceleration
        self.predict(u=base_acc)

        # update step with leg odometry result as input
        self.update(self.z)

    def predict(self, u):
        """Prediction step of Kalman Filter. Estimate robot state based on previous estimation.

        Args:
            u (np.ndarray): control input vector (here: base acceleration)
        """

        try:
            x_pred = self.A @ self.x + self.B @ u.T
            P_pred = self.A @ self.P @ self.A.T + self.Q
        except ValueError as e:
            print("Error in ekf.predict", e)

        self.x_pred = x_pred
        self.P_pred = P_pred
    
    def update(self, z):
        """Update step of Kalman Filter. Update estimation with consideration of measurement, comes from leg odometry.

        Args:
            z (np.ndarray): measurement (result of leg odometry)
        """

        z_tilde = z - self.H @ self.x_pred # measurement residual
        self.z_tilde = z_tilde
        S = self.H @ self.P_pred @ self.H.T + self.R # residual covariance
        K = self.P_pred @ self.H.T @ np.linalg.inv(S) # kalman gain
        self.K = K
        
        x_update = self.x_pred + K @ z_tilde
        P_update = (np.eye(K.shape[0]) - K @ self.H) @ self.P_pred

        self.x = x_update
        self.P = P_update
