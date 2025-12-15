import numpy as np
import jax.numpy as jnp
import jax
import mujoco
# Update JAX configuration
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

from gym_quadruped.quadruped_env import QuadrupedEnv

class StateEstimator():
    """"
    This class implements the state estimation for quadrupeds, based on chapter 12.4 from the SLAM Handbook (https://github.com/SLAM-Handbook-contributors/slam-handbook-public-release).
    
    - gathering information of forces and contacts with MuJoCo
    - using JAX for numerical calculations
    """

    def __init__(self):
        """
        Docstring for __init__
        
        :param self: Description
        :param env: object of QuadrupedEnv
        """

        # state values
        # pose of robot's base in world coordinates
        self.pos = None
        self.orient = None

        self.R = None
        self.dt = None

        # robot base's twist in base coordinates
        self.lin_vel = None
        self.ang_vel = None

        self.state = {
            "pos": jnp.zeros(3),
            "orient": jnp.eye(3),
            "lin_vel": jnp.zeros(3),
            "ang_vel": jnp.zeros(3),
        }

        # joint values
        self.q = None
        self.qdot = None

        # contact values
        self.robot_feet_geom_names = dict(FR='FR',FL='FL', RR='RR' , RL='RL')
        self.leg_names = ['FL','FR','RL','RR']
        self.contact_states = None
        self.contact_forces = None
        self.contact_pos = None

    def get_state(self):
        return self.state
    
    # source: https://cookierobotics.com/080/
    def quaternion_to_rotation(self, orient):
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

    def forward_kinematics(self, feet_pos):
        """
        Calculates the position of the feet relatively to the base: 
        f_p(q) = R^T * (c_i - t)
        - R = robot orientation (rotation from world frame to base frame)
        - c_i = position of i-th feet when in contact (world frames)
        - t = robot position (world frame)
        """

        return self.R.T @ (feet_pos - self.pos)

    def calc_leg_odometry(self, contact_state, contact_force):
        """
        Estimate robot movement based on informations from the legs (joint angle, joint velocity, contact) by using the forward kinematics to get the position of the foot based on the base.
        """

        # contact estimation
        self.estimate_contact_states(contact_state)
        self.estimate_contact_forces(contact_force)

        # motion estimation
        foot_vels = []
        for i in range(len(self.leg_names)):
            if not self.contact_states[i]:
                continue

            # relative pose estimation
            foot_in_base = self.forward_kinematics(self.contact_pos[i])

            # velocity estimation
            lin_jacobian = jax.jacobian(self.forward_kinematics)(self.pos)
            rel_vals = -np.cross(self.ang_vel, foot_in_base) - (lin_jacobian @ self.qdot)
            foot_vels.append(self.R @ rel_vals) # transformation body frame -> world frame
        
        new_vel = np.mean(np.array(foot_vels))
        new_pos = self.pos + new_vel * self.dt # integrate velocity to get position
        
        return new_pos, new_vel

    def update(self, dt, base_pos, base_orient, base_ang_val, joint_pos, joint_vel, contact_states, contact_force, contact_pos):
        """
        Updating the estimated state.
        """

        self.dt = dt

        self.pos = base_pos
        self.orient = base_orient
        self.R = self.quaternion_to_rotation(base_orient) # turn quaternion orientation to rotation matrix
        self.ang_vel = base_ang_val

        self.contact_states = contact_states
        self.contact_pos = contact_pos
        self.contact_forces = contact_force

        self.q = joint_pos
        self.qdot = joint_vel

        self.contact_states = contact_states
        self.contact_pos = contact_pos

        self.calc_leg_odometry(contact_states, contact_force)