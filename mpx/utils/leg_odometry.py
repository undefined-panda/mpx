import numpy as np
import mujoco

from gym_quadruped.quadruped_env import QuadrupedEnv
from dataclasses import dataclass
from mpx.utils.kf_utils import quat_to_rot, rot_to_quat

@dataclass
class State:
    pos: np.ndarray
    vel: np.ndarray

class LegOdom():
    """
    This class implements leg odometry for quadrupeds based on chapter 12.4 from the SLAM Handbook (https://github.com/SLAM-Handbook-contributors/slam-handbook-public-release).
    The estimation consists of a motion estimation and contact estimation. The state of the robot is defined as the position and velocity of the base.
    
    Motion Estimation:
    By exploiting the fact that legged robots only move when their leg is in contact with the ground (stance phase), we get an (noisy) estimation of the base velocity in world frame.

    Contact Estimation:
    A foot is considered in contact when its stationary over time, i.e. it does not slip. For that, the ground reaction force is used.
    """

    def __init__(self, model_name="aliengo"):
        self.env = QuadrupedEnv(robot=model_name) # legs_order = ('FL', 'FR', 'RL', 'RR')

        self.dt = None
        self.state = State(pos=np.zeros(3), vel=np.zeros(3))

        self.info_log = False

    def estimate_contact_states(self, contact_force, threshold):
        """Estimate contact state of legs. Using equation (12.22) from SLAM Handbook.

        Args:
            contact_force (np.ndarray): force acting on the contact point during stance phase
            threshold (int): contact force threshold indicating contact with the ground
            joint_torque (np.ndarray): joint torque
        """

        # x, y, z values for force
        if contact_force.shape == (4,3):
            fx, fy, fz = contact_force[:, 0], contact_force[:, 1], contact_force[:, 2]
            contact_state = np.sqrt(fx**2 + fy**2) <= self.mu * fz

        # single value for force
        elif contact_force.shape == (4,):
            contact_state = contact_force > threshold

        else:
            raise ValueError(f"contact_force has invalid shape: {contact_force.shape}")
        
        return contact_state
        
    def compute_foot_positions_B(self, joint_pos):
        model = self.env.mjModel
        data = self.env.mjData

        data.qpos[:7] = np.array([0, 0, 0, 1.0, 0, 0, 0]) # pos = (0,0,0), orient = (1,0,0,0) -> in origin, no rotation
        data.qpos[7:] = joint_pos

        mujoco.mj_kinematics(model, data) # compute pos and orient based on current joint pos

        foot_positions = []
        for name in self.env.legs_order:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            pos = data.geom_xpos[geom_id].copy()
            foot_positions.append(pos)

        return np.array(foot_positions)

    def compute_leg_odometry(self, dt, base_orient, base_ang_vel, qdot, joint_pos, J_b, contact_state=None):
        """Estimate robot velocity based on informations from the legs. Using equation (12.21) from SLAM Handbook.

        Args:
            dt (float): sampling time
            base_orient (np.ndarray): base orientation as quaternion in [w, x, y, z] format or rotation matrix
            base_ang_vel (np.ndarray): base angular velocity in world frame
            qdot (np.ndarray): joint velocity
            joint_torque (np.ndarray): joint torque
            joint_pos (np.ndarray): joint position
            contact_state (np.ndarray | None): contact state
            contact_force (np.ndarray | None): contact force
            contact_state_threshold (int): contact state threshold
        """

        # provide orientation as quaternion and rotation matrix
        if base_orient.shape == (4,):
            self.orient_rot = quat_to_rot(orient=base_orient)
            self.orient_quat = base_orient
        else:
            self.orient_rot = base_orient
            self.orient_quat = rot_to_quat(orient=base_orient)

        # foot position in base frame
        p_b = self.compute_foot_positions_B(joint_pos=joint_pos)
        self.p_b = p_b

        # motion estimation
        estimated_lin_vels = []
        for i in range(4):
            omega_b = self.orient_rot.T @ base_ang_vel # transform angular velocity into base frame
            j_v = J_b[self.env.legs_order[i]][:, 6:] # velocity estimation
            v_b = np.cross(-omega_b, p_b[i]) - (j_v @ qdot) # equation taken from SLAM handbook (Eq (12.21)), everything is in base frame
            v_w = self.orient_rot @ v_b # transform to world frame
            if contact_state[i]:
                estimated_lin_vels.append(v_w)
        
        if not estimated_lin_vels:
            new_vel = np.zeros(3)
        else:
            new_vel = np.mean(np.array(estimated_lin_vels), axis=0)
            
        self.state.pos = self.state.pos + new_vel * dt # integrate velocity to get position <- estimation of previous step, not measurement
        self.state.vel = new_vel
