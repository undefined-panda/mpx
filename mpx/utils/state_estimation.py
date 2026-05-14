import numpy as np
from utils.leg_odometry import LegOdom
from utils.dynamics_model import *
from utils.ekf_utils import rodrigues, rot_to_quat, is_rotation_matrix, is_quaternion

class KF():
    """
    This class imeplemnts a Kalman Filter using Leg Odometry as measurement.
    """

    def __init__(self, init_pos, dt, Q_diag=1e-2, R_diag=1e-2):
        self.dt = dt # currently using constant dt

        self.x = np.concatenate([init_pos, np.eye(3).flatten(), np.zeros(3), np.zeros(3)]) # state: [pos orient v_lin v_ang] - orientation as rotation matrix

        # Kalman Filter matrices
        self.A = np.eye(18) # state transition matrix
        self.A[:3, 12:15] = self.dt * np.eye(3) # velocity contributes to position via dt
        self.A[3:12, 3:12] = np.eye(9) # init for orientation estimation
        self.B = np.vstack([np.zeros((12,6)), self.dt*np.eye(6)]) # control input matrix
        self.H = np.hstack([np.zeros((6,12)), np.eye(6)]) # observation matrix
    
        # Kalman Filter noise
        self.Q = np.diag(Q_diag*np.ones(18)) # process noise. lower value means trusting the model more
        self.R = np.diag(R_diag*np.ones(self.H.shape[0])) # measurement noise. lower value means trusting the measurement more

        self.P = self.Q # error covariance

        self.leg_odom = LegOdom(init_state=self.x)

        self.verbose = True

        self.filter_states = ["predict", "update"]
    
    def get_pos(self, filter_state="update"):
        """Return state position. Choose after which filter step: 'predict' or 'update'

        Args:
            filter_state (str, optional): _description_. Defaults to "update".

        Returns:
            np.ndarray: Position of the state
        """
        if filter_state not in self.filter_states:
            return ValueError(f"Invalid filter state: {filter_state}")
        
        if filter_state == "predict": state = self.x_pred
        if filter_state == "update": state = self.x

        return state[0:3]
    
    def get_orient(self, filter_state="update", form="rotation-matrix"):
        """Return state orientation. 
        Choose a orientation form: 'rotation-matrix', 'quaternion' or 'flatten'

        Args:
            form (str, optional): Return orientation form. Defaults to "rotation-matrix".

        Returns:
            np.ndarray: Orientation of the state
        """
        if filter_state not in self.filter_states:
            return ValueError(f"Invalid filter state: {filter_state}")
        
        if filter_state == "predict": state = self.x_pred
        if filter_state == "update": state = self.x

        forms = ["flatten", "rotation-matrix", "quaternion"]
        if form not in forms:
            return ValueError(f"Chosen orientation form is invalid: {form}. Choose one from: {forms}")
        
        idx = forms.index(form)
        orient_forms = [state[3:12], state[3:12].reshape((3,3)), rot_to_quat(state[3:12].reshape((3,3)))]
        
        return orient_forms[idx]

    def get_lin_vel(self, filter_state="update"):
        if filter_state not in self.filter_states:
            return ValueError(f"Invalid filter state: {filter_state}")
        
        if filter_state == "predict": state = self.x_pred
        if filter_state == "update": state = self.x

        return state[12:15]

    def get_ang_vel(self, filter_state="update"):
        if filter_state not in self.filter_states:
            return ValueError(f"Invalid filter state: {filter_state}")
        
        if filter_state == "predict": state = self.x_pred
        if filter_state == "update": state = self.x

        return state[15:18]
    
    def step(self, 
             base_orient, 
             base_acc, 
             base_ang_vel, 
             joint_pos, 
             joint_vel, 
             joint_acc, 
             joint_torque, 
             contact_states, 
             contact_forces, 
             contact_pos, 
             contact_state_threshold) -> None:
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
        orient_rot = self.x[3:12].reshape((3,3))
        if not is_rotation_matrix(orient_rot) or not is_quaternion(rot_to_quat(orient_rot)):
            # print("Fehler")
            pass
        self.leg_odom.compute_leg_odometry(dt=self.dt,
                                           base_orient=orient_rot, # reshape flatten rotation matrix to 3x3 matrix
                                           base_ang_vel=base_ang_vel,
                                           qdot=joint_vel,
                                           joint_torque=joint_torque,
                                           joint_pos=joint_pos,
                                           contact_state=contact_states,
                                           contact_force=contact_forces,
                                           contact_state_threshold=contact_state_threshold)
        
        self.leg_odom_vel = self.leg_odom.state.vel
        self.leg_odom_pos = self.leg_odom.state.pos
        self.c_force = self.leg_odom.contact_forces
        self.c_state = self.leg_odom.contact_states

        # estimate base acceleration with dynamics model
        if base_acc is None:
            body_mass = float(np.sum(self.leg_odom.env.mjModel.body_mass))
            base_acc = estimate_acc_from_contact_force(mass=body_mass,
                                                       contact_states=self.c_state,
                                                       contact_forces=self.c_force)
            
            if self.verbose:
                base_acc2 = estimate_acc_from_contact_force_v2(env=self.leg_odom.env,
                                                               contact_forces=self.c_force,
                                                               contact_states=self.c_state,
                                                               joint_acc=joint_acc)
                self.base_acc2 = base_acc2

                base_acc3 = estimate_acc_from_contact_force_v3(env=self.leg_odom.env,
                                                               contact_forces=self.c_force,
                                                               contact_states=self.c_state,
                                                               joint_acc=joint_acc,
                                                               contact_pos_b=self.leg_odom.p_b,
                                                               R=self.leg_odom.orient_rot)
                
                self.base_acc3 = base_acc3
                base_acc = base_acc3
        
        self.base_acc = base_acc

        # check if angular velocity contains only zero entries, use previous orientation if thats the case
        # adjust A for orientation estimation since it changes for each step
        if self.x[-3:].any():
            exp_map = rodrigues(self.x[-3:], self.dt)
            exp_map = np.block([[a * np.eye(3) for a in row] for row in exp_map]) # adjust to fit in A matrix and make it compatible with flatten rotation matrix
            self.A[3:12, 3:12] = exp_map # equal to exp_map @ orient. source: https://cwzx.wordpress.com/2013/12/16/numerical-integration-for-rotational-dynamics/

        # prediction step with base acceleration
        self.predict(u=base_acc)

        # update step with leg odometry estimation of velocity and IMU angular velocity as measurement
        measurement = np.concatenate([self.leg_odom_vel, base_ang_vel]) # define ang vel
        self.update(z=measurement)

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

        # print(x_update)
