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

@dataclass
class State:
    pos: np.ndarray
    vel: np.ndarray

class LegOdom():
    """"
    This class implements the state estimation for quadrupeds, based on chapter 12.4 from the SLAM Handbook (https://github.com/SLAM-Handbook-contributors/slam-handbook-public-release).
    
    - gathering information of forces and contacts with MuJoCo
    - using JAX for numerical calculations
    """

    def __init__(self, init_state):
        self.env = QuadrupedEnv(robot="aliengo")

        self.dt = None
        self.state = State(pos=init_state[:3], vel=init_state[3:])

        # contact values
        self.robot_feet_geom_names = dict(FR='FR',FL='FL', RR='RR' , RL='RL')
        self.leg_names = ['FL','FR','RL','RR']

    def get_state(self):
        return self.state
    
    # source: https://cookierobotics.com/080/
    def quat_to_rot(self, orient):
        w, x, y, z = orient
        R = np.array([
            [2*(w**2 + x**2) - 1, 2*(x*y - w*z)      , 2*(w*y + x*z)      ],
            [2*(x*y + w*z)      , 2*(w**2 + y**2) - 1, 2*(y*z - w*x)      ],
            [2*(x*z - w*y)      , 2*(y*z + w*x)      , 2*(w**2 + z**2) - 1]
        ])
        
        return R

    def estimate_contact_forces(self, joint_torque, contact_state):
        """
        Estimate the forces acting on the contact points (feet) using the dynamics model.
        """

        if contact_state is None or joint_torque is None:
            raise ValueError(f"contact_state and joint_torque are needed to estimate contact_force. contact_state: {contact_state} \t joint_torque: {joint_torque}")

        force = np.zeros((3,))
        contact_forces = []

        # sum jacobians of all legs that are in contact
        for i in range(4):
            leg_name = self.env.legs_order[i]

            J_lin = self.lin_jacobian_w[leg_name][:, 6:]
            c_force = contact_state[i] * (J_lin @ joint_torque)
            contact_forces.append(c_force)
        
        self.contact_forces = np.array(contact_forces)

    def estimate_contact_states(self, contact_force, threshold, joint_torque):
        """
        Estimate the contact state of the feet (touching the ground? floating?).
        """

        if contact_force is None or threshold is None:
            raise ValueError(f"contact_force and threshold are needed to estimate the contact state. contact_force: {contact_force} \t threshold: {threshold}")
        
        contact_force = np.array(contact_force)
        if contact_force.shape == (4,3): # x, y, z values for force
            mask = (np.sqrt(contact_force[:, 0]**2 + contact_force[:, 1]**2) <= contact_force[:, 2])
            contact_state = (contact_force != 0)[:, 0] & mask
        elif contact_force.shape == (4,): # single value for force
            if threshold is None:
                raise ValueError(f"threshold is needed to estimate the force, when its shape is (4,)")
            contact_state = contact_force > threshold
            print("Estimating contact_force from estimated contact_state and joint torque")
            self.estimate_contact_forces(contact_state, joint_torque)
        else:
            print(f"contact_force has invalid shape: {contact_force.shape}")
            return
        
        self.contact_states = contact_state

    def calc_rel_foot_pos(self, feet_pos, frame="world"):
        """
        Calculates the position of the feet relatively to the base: 
        f_p(q) = R^T * (c_i - t)
        - R = robot orientation (rotation from world frame to base frame)
        - c_i = position of i-th feet when in contact (world frames)
        - t = robot position (world frame)
        """

        if frame == 'world':
            return (feet_pos - self.state.pos)
        elif frame == 'base':
            return self.R.T @ (feet_pos - self.state.pos)
        else:
            raise ValueError(f"Invalid frame: {frame} != 'world' or 'base'")
        
    def compute_foot_positions_B(self, joint_pos):
        model = self.env.mjModel
        data = self.env.mjData

        data.qpos[:7] = np.array([0, 0 , 0, 1.0, 0, 0, 0])
        data.qpos[7:] = joint_pos

        mujoco.mj_forward(model, data)

        foot_positions = []
        for name in self.leg_names:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            pos = data.geom_xpos[geom_id].copy()
            foot_positions.append(pos)

        return foot_positions

    def calc_leg_odometry(self, dt, base_orient, base_ang_vel, qdot, joint_torque, joint_pos, contact_state, contact_force, contact_pos, contact_state_threshold):
        """
        Estimate robot movement based on informations from the legs (joint angle, joint velocity, contact) by using the forward kinematics to get the position of the foot based on the base.
        """

        # set mjData pos
        self.env.mjData.qpos[:] = np.concatenate([np.zeros(shape=(3,)), base_orient, joint_pos])
        self.env.mjData.qvel[:] = np.concatenate([np.zeros(shape=(6,)), qdot])
        
        # calculate lineare jacobian
        mujoco.mj_forward(self.env.mjModel, self.env.mjData)
        self.lin_jacobian_b = self.env.feet_jacobians(frame="base") # use mj_jac from MuJoCo, defined in QuadrupedEnv
        self.lin_jacobian_w = self.env.feet_jacobians(frame="world")

        # turn quaternion orientation to rotation matrix
        self.R = self.quat_to_rot(orient=base_orient) 

        # contact estimation. if values for contact_force or contact_state are not provided, they will be estimated
        if contact_force is None:
            self.estimate_contact_forces(joint_torque=joint_torque, contact_state=contact_state)
        else:
            self.contact_forces = contact_force
        
        if contact_state is None:
            self.estimate_contact_states(contact_force=contact_force, threshold=contact_state_threshold, joint_torque=joint_torque)
        else:
            self.contact_states = contact_state

        # motion estimation
        estimated_lin_vels = []

        p_b = self.compute_foot_positions_B(joint_pos=joint_pos)
        for i in range(len(self.leg_names)):
            if not self.contact_states[i]:
                continue

            omega_b = self.R.T @ base_ang_vel # transform into base frame

            # relative pose estimation
            # p_b = self.calc_rel_foot_pos(contact_pos[i], frame="base")

            # velocity estimation
            j_v = self.lin_jacobian_b[self.leg_names[i]][:, 6:]
            
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
