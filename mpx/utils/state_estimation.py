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

    def __init__(self, env):
        """
        Docstring for __init__
        
        :param self: Description
        :param env: object of QuadrupedEnv
        """

        self.env = env

        # state values
        # pose of robot's base in world coordinates
        self.pos = None
        self.orient = None

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

    def estimate_contact_forces(self):
        """
        Estimate the forces acting on the contact points (feet) using the dynamical model.
        
        :param self: Description
        """
        _, contact_temp, contact_forces_temp = self.env.feet_contact_state(ground_reaction_forces=True)
            
        self.contact_forces = np.array([contact_forces_temp[self.robot_feet_geom_names[leg]] for leg in self.leg_names])
        self.contact_pos = np.array([contact_temp[self.robot_feet_geom_names[leg]][0].pos for leg in self.leg_names])

    def estimate_contact_states(self):
        """
        Estimate the contact state of the feet (touching the ground? floating?).
        """
        contact_states_temp, contact_temp = self.env.feet_contact_state(ground_reaction_forces=False)
            
        self.contact_states = np.array([contact_states_temp[self.robot_feet_geom_names[leg]] for leg in self.leg_names])
        self.contact_pos = np.array([contact_temp[self.robot_feet_geom_names[leg]][0].pos for leg in self.leg_names])

    def forward_kinematics(self, feet_pos):
        """
        Calculates the position of the feet relatively to the base: 
        f_p(q) = R^T * (c_i - t)
        - R = robot orientation (rotation from world frame to base frame)
        - c_i = position of i-th feet when in contact (world frames)
        - t = robot position (world frame)
        """

        return self.orient.T @ (feet_pos - self.pos)

    def calc_leg_odometry(self):
        """
        Estimate robot movement based on informations from the legs (joint angle, joint velocity, contact) by using the forward kinematics to get the position of the foot based on the base.
        """

        # contact estimation
        self.estimate_contact_states()
        self.estimate_contact_forces()

        # motion estimation
        foot_vels = []
        for i in range(len(self.leg_names)):
            if not self.contact_states[i]:
                continue

            foot_in_base = self.forward_kinematics(self.contact_pos[i])
            lin_jacobian = jax.jacobian(self.forward_kinematics)(self.pos)
            rel_vals = -np.cross(self.ang_vel, foot_in_base) - (lin_jacobian @ self.qdot)
            foot_vels.append(rel_vals)

    def update(self):
        """
        Updating the estimated state
        """

        qpos = self.env.mjData.qpos.copy()
        self.pos = qpos[0:3]
        self.orient = qpos[3:7]
        self.q = qpos[7:]

        qvel = self.env.mjData.qvel.copy()
        self.lin_vel = qvel[0:3]
        self.ang_vel = qvel[3:6]
        self.qdot = qvel[6:]