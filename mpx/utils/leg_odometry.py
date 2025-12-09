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

        # robot base's twist in base coordinates
        self.lin_vel = None
        self.ang_vel = None

        self.estimated_ang_vel = np.zeros(3)
        self.R = None # rotation matrix, world -> base

        # joint values
        self.q = None
        self.qdot = None
        self.lin_jacobian = None

        # contact values
        self.robot_feet_geom_names = dict(FR='FR',FL='FL', RR='RR' , RL='RL')
        self.leg_names = ['FL','FR','RL','RR']
        self.contact_states = None
        self.contact_forces = None

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

    def estimate_contact_forces(self, contact_force):
        """
        Estimate the forces acting on the contact points (feet) using the dynamical model.
        (Temporary using values from simulation directly.)
        """
        
        self.contact_forces = contact_force

    def estimate_contact_states(self, contact_state):
        """
        Estimate the contact state of the feet (touching the ground? floating?).
        (Temporary using values from simulation directly.)
        """
        
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

    def calc_leg_odometry(self, dt, base_orient, base_ang_vel, qdot, contact_state, contact_force, contact_pos):
        """
        Estimate robot movement based on informations from the legs (joint angle, joint velocity, contact) by using the forward kinematics to get the position of the foot based on the base.
        """

        self.R = self.quat_to_rot(base_orient) # turn quaternion orientation to rotation matrix
        #self.R = self.env.base_configuration[0:3, 0:3]

        # contact estimation
        self.estimate_contact_states(contact_state)
        self.estimate_contact_forces(contact_force)

        # motion estimation
        estimated_lin_vels = []
        for i in range(len(self.leg_names)):
            if not self.contact_states[i]:
                continue

            omega_b = self.R.T @ base_ang_vel # transform into base frame

            # relative pose estimation
            p_b = self.calc_rel_foot_pos(contact_pos[i], frame="base")

            # velocity estimation
            j_v = self.lin_jacobian[self.leg_names[i]][:, 6:]
            
            # equation taken from SLAM handbook (Eq (12.21))
            v_b = -np.cross(omega_b, p_b) - (j_v @ qdot) # everything is in base frame
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

    # def update(self, dt, base_pos, base_orient, base_ang_vel, joint_pos, joint_vel, contact_states, contact_force, contact_pos):
    #     """
    #     Update the estimated state.
    #     """

    #     # self.dt = dt

    #     # self.qpos = np.concatenate([base_pos, base_orient, joint_pos])
    #     # self.env.mjData.qpos[:] = self.qpos
    #     # mujoco.mj_forward(self.env.mjModel, self.env.mjData)
    #     # self.lin_jacobians, self.ang_jacobians = self.env.feet_jacobians(frame="base", return_rot_jac=True) # use mj_jac from MuJoCo, defined in QuadrupedEnv

    #     # self.R = self.quat_to_rot(base_orient) # turn quaternion orientation to rotation matrix
    #     # #self.R = self.env.base_configuration[0:3, 0:3]
    #     # self.ang_vel = base_ang_vel

    #     # self.contact_pos = contact_pos

    #     # self.q = joint_pos
    #     # self.qdot = joint_vel

    #     self.calc_leg_odometry(contact_states, contact_force)
