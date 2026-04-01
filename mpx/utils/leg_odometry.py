import numpy as np
import jax.numpy as jnp
import jax
import mujoco
# Update JAX configuration
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

from gym_quadruped.quadruped_env import QuadrupedEnv
from dataclasses import dataclass
from utils.dynamics_model import estimate_contact_forces

@dataclass
class State:
    pos: np.ndarray
    vel: np.ndarray

class LegOdom():
    """
    This class implements leg odometry for quadrupeds based on chapter 12.4 from the SLAM Handbook (https://github.com/SLAM-Handbook-contributors/slam-handbook-public-release).
    The estimation consists of a motion estimation and contact estimation. The state of the robot is defined as the position and velocity of the base: state = [pos vel]
    
    #### Motion Estimation ####
    By exploiting the fact that legged robots only move when their leg is in contact with the ground (stance phase), we get an (noisy) estimation of the base velocity in world frame.

    #### Contact Estimation ####
    A foot is considered in contact when its stationary over time, i.e. it does not slip. The Ground Reaction Force is used for that.
    """

    def __init__(self, init_state):
        self.env = QuadrupedEnv(robot="go1") # legs_order = ('FL', 'FR', 'RL', 'RR')

        self.dt = None
        self.state = State(pos=init_state[:3], vel=init_state[3:])

        self.info_log = False

    def get_state(self) -> State:
        """Returns state

        Returns:
            State: position and velocity of robot's base
        """

        return self.state
    
    def quat_to_rot(self, orient) -> np.ndarray:
        """Convert quaternion to rotation matrix (source: https://cookierobotics.com/080/).

        Args:
            orient (np.ndarray | list): quaternion in [w, x, y, z] format

        Returns:
            np.ndarray: corresponding rotation matrix
        """

        w, x, y, z = orient
        R = np.array([
            [2*(w**2 + x**2) - 1, 2*(x*y - w*z)      , 2*(w*y + x*z)      ],
            [2*(x*y + w*z)      , 2*(w**2 + y**2) - 1, 2*(y*z - w*x)      ],
            [2*(x*z - w*y)      , 2*(y*z + w*x)      , 2*(w**2 + z**2) - 1]
        ])
        
        return R
    
    def rpy_to_rot(self, orient) -> np.ndarray:
        """Convert rpy angles to rotation matrix.

        Args:
            orient (np.ndarray | list): rpy angles in roll, pitch, yaw format

        Returns:
            np.ndarray: corresponding rotation matrix
        """

        roll, pitch, yaw = orient
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
        
        # Combined rotation matrix: ZYX sequence
        R = np.dot(Rz, np.dot(Ry, Rx))
        return R

    def estimate_contact_states(self, contact_force, threshold, joint_torque) -> None:
        """Estimate contact state of legs. Additionally estimating contact force in case of contact_force having single value per feet.

        Args:
            contact_force (np.ndarray): force acting on the contact point during stance phase
            threshold (int): contact force threshold indicating contact with the ground
            joint_torque (np.ndarray): joint torque

        Raises:
            ValueError: error if contact_force or threshold are None
            ValueError: error if contact_force has invalid shape
        """

        if contact_force is None or threshold is None:
            raise ValueError(f"contact_force and threshold are needed to estimate the contact state. contact_force: {contact_force} \t threshold: {threshold}")
        
        contact_force = np.array(contact_force)

        # x, y, z values for force
        if contact_force.shape == (4,3):
            mask = (np.sqrt(contact_force[:, 0]**2 + contact_force[:, 1]**2) <= contact_force[:, 2])
            contact_state = (contact_force != 0)[:, 0] & mask

        # single value for force
        elif contact_force.shape == (4,):
            contact_state = contact_force > threshold
            
            if not self.info_log:
                print("Estimating contact_force from estimated contact_state and joint torque")
                self.info_log = True

            self.contact_forces = estimate_contact_forces(joint_torque=joint_torque, 
                                                          contact_state=contact_state, 
                                                          legs_order=self.env.legs_order, 
                                                          lin_jacobian_w=self.lin_jacobian_w)

        else:
            raise ValueError(f"contact_force has invalid shape: {contact_force.shape}")
        
        self.contact_states = contact_state
        
    def compute_foot_positions_B(self, joint_pos) -> np.ndarray:
        """Compute foot position in base frame

        Args:
            joint_pos (np.ndarray): joint position

        Returns:
            np.ndarray: foot position in base frame
        """

        model = self.env.mjModel
        data = self.env.mjData

        data.qpos[:7] = np.array([0, 0 , 0, 1.0, 0, 0, 0])
        data.qpos[7:] = joint_pos

        mujoco.mj_forward(model, data)

        foot_positions = []
        for name in self.env.legs_order:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            pos = data.geom_xpos[geom_id].copy()
            foot_positions.append(pos)

        return np.array(foot_positions)

    def compute_leg_odometry(self, dt, base_orient, base_ang_vel, qdot, joint_torque, joint_pos, contact_state, contact_force, contact_state_threshold) -> None:
        """Estimate robot velocity based on informations from the legs. Using equation (12.21) from SLAM Handbook.

        Args:
            dt (float): sampling time
            base_orient (np.ndarray): base orientation as quaternion in [w, x, y, z] format
            base_ang_vel (np.ndarray): base angular velocity in world frame
            qdot (np.ndarray): joint velocity
            joint_torque (np.ndarray): joint torque
            joint_pos (np.ndarray): joint position
            contact_state (np.ndarray | None): contact state
            contact_force (np.ndarray | None): contact force
            contact_state_threshold (int): contact state threshold
        """

        # create rotation matrix
        self.R = self.quat_to_rot(orient=base_orient) # turn quaternion orientation to rotation matrix: base -> world

        # set mjData pos
        self.env.mjData.qpos[:] = np.concatenate([np.zeros(shape=(3,)), base_orient, joint_pos])
        self.env.mjData.qvel[:] = np.concatenate([np.zeros(shape=(6,)), qdot])
        
        # calculate lineare jacobian
        mujoco.mj_forward(self.env.mjModel, self.env.mjData)
        self.lin_jacobian_b = self.env.feet_jacobians(frame="base") # use mj_jac from MuJoCo, defined in QuadrupedEnv
        self.lin_jacobian_w = self.env.feet_jacobians(frame="world")

        # contact estimation. if values for contact_force or contact_state are not provided, they will be estimated        
        if contact_state is None:
            self.estimate_contact_states(contact_force=contact_force, threshold=contact_state_threshold, joint_torque=joint_torque)
        else:
            self.contact_states = contact_state
            if contact_force is None:
                self.contact_forces = estimate_contact_forces(joint_torque=joint_torque, 
                                                              contact_state=self.contact_states, 
                                                              legs_order=self.env.legs_order, 
                                                              lin_jacobian_w=self.lin_jacobian_w)
            else:
                self.contact_forces = contact_force

        # motion estimation
        estimated_lin_vels = []

        # foot position in base frame
        p_b = self.compute_foot_positions_B(joint_pos=joint_pos)

        for i in range(len(self.env.legs_order)):
            if not self.contact_states[i]:
                continue

            # transform angular velocity into base frame
            omega_b = self.R.T @ base_ang_vel 

            # velocity estimation
            j_v = self.lin_jacobian_b[self.env.legs_order[i]][:, 6:]
            
            # equation taken from SLAM handbook (Eq (12.21))
            v_b = np.cross(-omega_b, p_b[i]) - (j_v @ qdot) # everything is in base frame
            v_w = self.R @ v_b # transform to world frame
            estimated_lin_vels.append(v_w)
        
        if not estimated_lin_vels:
            new_vel = np.zeros(3)
            new_pos = self.state.pos
        else:
            new_vel = np.mean(np.array(estimated_lin_vels), axis=0)
            new_pos = self.state.pos + new_vel * dt # integrate velocity to get position <- estimation of previous step, not measurement
        
        self.state.pos = new_pos
        self.state.vel = new_vel
