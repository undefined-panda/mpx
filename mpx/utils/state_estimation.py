import numpy as np
from utils.leg_odometry import LegOdom
from utils.dynamics_model import *
from utils.ekf_utils import *

class KF():
    """
    This class imeplemnts a Kalman Filter using Leg Odometry as measurement.

    The state consists of:
    - position
    - orientation (as a flattened orientation matrix)
    - linear velocity
    - angular velocity
    - contact force (as a flattened vector containing force values for each foot)

    The control input is the joint acceleration, used to estimate the base acceleration inside the kalman filter.

    The measurements are:
    - linear velocity coming from leg odometry
    - angular velocity coming from IMU
    - orientation coming from IMU
    """

    def __init__(self, init_pos, dt, Q_diag=1e-2, R_diag=1e-2, model_name="aliengo"):
        self.dt = dt # currently using constant dt

        # state        
        self.x = np.concatenate([init_pos, # pos
                                 np.eye(3).flatten(), # orient as rotation matrix
                                 np.zeros(3), # v_lin
                                 np.zeros(3), # v_ang
                                 np.zeros(12)]) # contact force

        self.POS = slice(0, 3)
        self.ORIENT = slice(3, 12)
        self.VPHI = slice(3, 6)
        self.V_LIN = slice(12, 15)
        self.V_ANG = slice(15, 18)
        self.C_FORCE = slice(18, 30)

        ##########################
        # Kalman Filter matrices #
        ##########################
        # state transition matrix
        self.A = np.eye(30) 
        self.A[self.POS, self.V_LIN] = self.dt * np.eye(3) # make velocity contribute to position via dt: pos + dt * v_lin
        self.A[self.ORIENT, self.ORIENT] = np.eye(9) # init for orientation estimation

        # control input matrix
        self.B = np.zeros((30,13))

        # observation matrix
        self.H = np.zeros((21, 30)) # z = [orient_error(3) v_lin(3) v_ang(3) contact_force(12)]
        self.H[0:3, self.VPHI] = np.eye(3)
        self.H[3:6, self.V_LIN] = np.eye(3)
        self.H[6:9, self.V_ANG] = np.eye(3)
        self.H[9:21, self.C_FORCE] = np.eye(12)
    
        # Kalman Filter noise
        self.Q = np.diag(Q_diag*np.ones(30)) # process noise. lower value means trusting the model more
        self.R = np.diag(R_diag*np.ones(self.H.shape[0])) # measurement noise. lower value means trusting the measurement more

        self.P = self.Q # error covariance

        self.leg_odom = LegOdom(init_state=self.x, model_name=model_name)

        self.verbose = True

        self.filter_states = ["predict", "update"]

    def get_filter_state(self, filter_state):
        if filter_state not in self.filter_states:
            return ValueError(f"Invalid filter state: {filter_state}")
        
        if filter_state == "predict": state = self.x_pred
        if filter_state == "update": state = self.x

        return state
    
    def get_pos(self, filter_state="update"):
        return self.get_filter_state(filter_state)[self.POS]
    
    def get_orient(self, filter_state="update", form="rotation-matrix"):
        state = self.get_filter_state(filter_state)

        forms = ["flatten", "rotation-matrix", "quaternion"]
        if form not in forms:
            return ValueError(f"Chosen orientation form is invalid: {form}. Choose one from: {forms}")
        
        idx = forms.index(form)
        orient_forms = [state[self.ORIENT], state[self.ORIENT].reshape((3,3)), rot_to_quat(state[self.ORIENT].reshape((3,3)))]
        
        return orient_forms[idx]

    def get_lin_vel(self, filter_state="update"):
        return self.get_filter_state(filter_state)[self.V_LIN]

    def get_ang_vel(self, filter_state="update"):
        return self.get_filter_state(filter_state)[self.V_ANG]
    
    def get_contact_force(self, filter_state="update"):
        return self.get_filter_state(filter_state)[self.C_FORCE].reshape((4,3))
    
    def update_A_orientation(self):
        exp_map = matrix_exp(self.get_ang_vel(), self.dt)
        exp_map = np.block([[a * np.eye(3) for a in row] for row in exp_map]) # adjust to fit in A matrix and make it compatible with flatten rotation matrix
        self.A[self.ORIENT, self.ORIENT] = exp_map 

    def update_A_B_contact_forces(self, env, orient, contact_pos_b, contact_state):
        qfrc_bias = env.mjData.qfrc_bias # contains coriolis and gravitational terms for each DOF, shape == (18,)

        M = np.zeros((env.mjModel.nv, env.mjModel.nv)) # shape == (18, 18)
        mujoco.mj_fullM(env.mjModel, M, env.mjData.qM)

        H_B = M[:6, :6]
        H_BL = M[:6, 6:18]

        J_a = [] # upper part, only for linear acceleration
        J_b = [] # lower part, only for angular acceleration
        for i in range(4):
            c_i = contact_state[i]
            cp_i = orient @ contact_pos_b[i]
            J_i = np.vstack([np.eye(3), skew(cp_i)])
            mass_weighted_jacobian = c_i * (np.linalg.pinv(H_B) @ J_i)
            J_a.append(mass_weighted_jacobian[:3, :])
            J_b.append(mass_weighted_jacobian[3:, :])

        J_new = np.vstack([np.hstack(J_a), np.hstack(J_b)])

        temp = np.linalg.pinv(H_B)@(-H_BL)

        self.A[12:18, self.C_FORCE] = self.dt * J_new
        self.B[12:18, :] = self.dt * np.hstack([temp, qfrc_bias[:6].reshape(-1, 1)])
    
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
             contact_state_threshold) -> None:
        """Kalman Filter algorithmic loop. Alternating between prediction step, gathering new measurement and update step.

        Args:
            base_orient (np.ndarray): base orientation as quaternion in [w, x, y, z] format
            base_acc (np.ndarray | None): base acceleration
            base_ang_vel (np.ndarray): base angular velocity (from IMU)
            joint_pos (np.ndarray): joint position
            joint_vel (np.ndarray): joint velocity
            joint_torque (np.ndarray): joint torque
            contact_states (np.ndarray): contact state of legs with ground
            contact_forces (np.ndarray): force acting on the contact point during stance phase
            contact_pos (np.ndarray): leg position during contact
            contact_state_threshold (int): contact force threshold indicating contact with the ground 
        """

        norms = np.linalg.norm(self.get_orient(form="quaternion"))
        if norms.round(4) != 1.0:
            print(self.get_orient(form="quaternion"))
            print(self.prev_base_orient)
        
        self.prev_base_orient = base_orient

        # using leg odometry as a measurement for linear velocity
        self.leg_odom.compute_leg_odometry(dt=self.dt,
                                           base_orient=self.get_orient(), # reshape flatten rotation matrix to 3x3 matrix
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
            # body_mass = float(np.sum(self.leg_odom.env.mjModel.body_mass))
            # base_acc = estimate_acc_from_contact_force(mass=body_mass,
            #                                         contact_states=self.c_state,
            #                                         contact_forces=self.c_force)
            
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
        
        # if angular velocity contains only zero entries, use previous orientation, else update A matrix
        if self.get_ang_vel().any(): self.update_A_orientation()

        # update B matrix since mass distribution changes for each step
        self.update_A_B_contact_forces(env=self.leg_odom.env, orient=self.leg_odom.orient_rot, contact_pos_b=self.leg_odom.p_b, contact_state=self.c_state)

        # prediction step with joint acceleration as control input
        self.predict(u=np.hstack([joint_acc, 1]))
        # update step. measurements:
        # - leg odom: linear velocity and contact_forces as J * tau
        # - IMU: angular velocity and orientation
        self.update(z=[quat_to_rot(base_orient), self.leg_odom_vel, base_ang_vel, self.c_force.flatten()])

    def predict(self, u):
        """Prediction step of Kalman Filter. Estimate robot state based on previous estimation.

        Args:
            u (np.ndarray): control input vector
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
            z (np.ndarray): measurement
        """
        
        pred_orient = self.get_orient(filter_state="predict")

        z_tilde_vel_force = np.concat([z[1], z[2], z[3]]) - self.H[3:21, :] @ self.x_pred # measurement residual of lin and ang velocity
        z_tilde_orient = matrix_log(pred_orient.T @ z[0]) # measurement residual of orientation
        z_tilde = np.concatenate([z_tilde_orient, z_tilde_vel_force])
        self.z_tilde = z_tilde

        S = self.H @ self.P_pred @ self.H.T + self.R # residual covariance
        K = self.P_pred @ self.H.T @ np.linalg.inv(S) # kalman gain
    
        correction = K @ z_tilde
        
        x_update = self.x_pred + correction # update for linear states
        x_update[self.ORIENT] = (pred_orient @ matrix_exp(correction[self.VPHI], 1)).flatten() # update for orient: R_pred with * exp(δθ) δθ = K @ Log(R_pred.T @ R_meas)

        P_update = (np.eye(K.shape[0]) - K @ self.H) @ self.P_pred

        self.x = x_update
        self.P = P_update
        self.K = K
