import numpy as np
import mujoco
import jax
import jax.numpy as jnp

from gym_quadruped.quadruped_env import QuadrupedEnv
from dataclasses import dataclass
from utils.dynamics_model import estimate_contact_forces, estimate_contact_forces_jax
from mpx.utils.kf_utils import quat_to_rot, rot_to_quat, quat_to_rot_jax, rot_to_quat_jax

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

    def __init__(self, init_state, model_name="aliengo"):
        self.env = QuadrupedEnv(robot=model_name) # legs_order = ('FL', 'FR', 'RL', 'RR')

        self.dt = None
        self.state = State(pos=init_state[:3], vel=init_state[3:])

        self.info_log = False

    def get_state(self) -> State:
        """Returns state

        Returns:
            State: position and velocity of robot's base
        """

        return self.state

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
        
        contact_force = np.array(contact_force)

        # x, y, z values for force
        if contact_force.shape == (4,3):
            mask = (np.sqrt(contact_force[:, 0]**2 + contact_force[:, 1]**2) <= contact_force[:, 2])
            contact_state = (contact_force != 0)[:, 0] & mask
            self.contact_forces = contact_force

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
        
    def compute_foot_positions_B(self, joint_pos):
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

        mujoco.mj_kinematics(model, data)  # propagate qpos -> geom_xpos (mj_forward is not needed, cheaper)

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
            base_orient (np.ndarray): base orientation as quaternion in [w, x, y, z] format or rotation matrix
            base_ang_vel (np.ndarray): base angular velocity in world frame
            qdot (np.ndarray): joint velocity
            joint_torque (np.ndarray): joint torque
            joint_pos (np.ndarray): joint position
            contact_state (np.ndarray | None): contact state
            contact_force (np.ndarray | None): contact force
            contact_state_threshold (int): contact state threshold
        """

        # create rotation matrix
        if base_orient.shape == (4,): # if given as quaternion
            self.orient_rot = quat_to_rot(orient=base_orient) # turn quaternion orientation to rotation matrix: base -> world
            self.orient_quat = base_orient
        else: # if given as rotation matrix
            self.orient_rot = base_orient
            self.orient_quat = rot_to_quat(orient=base_orient) # turn rotation matrix to quaternion

        # set mjData pos
        self.env.mjData.qpos[:] = np.concatenate([np.zeros(shape=(3,)), self.orient_quat, joint_pos])
        self.env.mjData.qvel[:] = np.concatenate([np.zeros(shape=(6,)), qdot])
        
        # calculate lineare jacobian
        mujoco.mj_forward(self.env.mjModel, self.env.mjData) # for jacobian

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

        # foot position in base frame
        p_b = self.compute_foot_positions_B(joint_pos=joint_pos)
        self.p_b = p_b

        # motion estimation
        estimated_lin_vels = []
        for i in range(4):
            omega_b = self.orient_rot.T @ base_ang_vel # transform angular velocity into base frame
            j_v = self.lin_jacobian_b[self.env.legs_order[i]][:, 6:] # velocity estimation
            v_b = np.cross(-omega_b, p_b[i]) - (j_v @ qdot) # equation taken from SLAM handbook (Eq (12.21)), everything is in base frame
            v_w = self.orient_rot @ v_b # transform to world frame
            estimated_lin_vels.append(np.where(self.contact_states[i], v_w, np.zeros(3)))
        
        if not estimated_lin_vels:
            new_vel = np.zeros(3)
            new_pos = self.state.pos
        else:
            new_vel = np.mean(np.array(estimated_lin_vels), axis=0)
            new_pos = self.state.pos + new_vel * dt # integrate velocity to get position <- estimation of previous step, not measurement
        
        self.state.pos = new_pos
        self.state.vel = new_vel


# ---------------------------------------------------------------------------
# JAX equivalent of LegOdom, for switching the estimation pipeline to a JAX backend.
# MuJoCo itself (mj_forward, mj_kinematics, feet jacobians) stays NumPy-based, since
# MuJoCo's classic Python bindings are not JAX-jittable; only the pure motion- and
# contact-force-estimation math uses jax.numpy/jax.jit/jax.vmap.
# ---------------------------------------------------------------------------

@jax.jit
def _compute_motion_estimate_jax(orient_rot, base_ang_vel, p_b, lin_jacobian_b_stack, qdot, contact_states):
    """jax.jit + jax.vmap equivalent of the per-leg motion-estimation loop in
    LegOdom.compute_leg_odometry (SLAM Handbook eq. 12.21), vectorized across legs
    instead of using a Python for loop.

    Args:
        orient_rot (jax.Array): base orientation, shape (3, 3)
        base_ang_vel (jax.Array): angular velocity in world frame, shape (3,)
        p_b (jax.Array): foot positions in base frame, shape (4, 3)
        lin_jacobian_b_stack (jax.Array): per-leg linear Jacobian (base frame), joint
            columns only, shape (4, 3, 12)
        qdot (jax.Array): joint velocities, shape (12,)
        contact_states (jax.Array): contact state per leg, shape (4,)

    Returns:
        jax.Array: estimated base velocity in world frame, shape (3,)
    """

    omega_b = orient_rot.T @ base_ang_vel  # transform angular velocity into base frame

    def per_leg(p_i, J_i, c_i):
        v_b = jnp.cross(-omega_b, p_i) - (J_i @ qdot)  # SLAM handbook eq. (12.21), base frame
        v_w = orient_rot @ v_b  # transform to world frame
        return jnp.where(c_i, v_w, jnp.zeros(3))

    estimated_lin_vels = jax.vmap(per_leg)(p_b, lin_jacobian_b_stack, contact_states)
    return jnp.mean(estimated_lin_vels, axis=0)

class LegOdomJax():
    """JAX equivalent of LegOdom (see LegOdom for the algorithm description/references)."""

    def __init__(self, init_state, model_name="aliengo"):
        self.env = QuadrupedEnv(robot=model_name)  # legs_order = ('FL', 'FR', 'RL', 'RR')

        self.dt = None
        self.state = State(pos=jnp.asarray(init_state[:3]), vel=jnp.asarray(init_state[3:]))

        self.info_log = False

    def get_state(self) -> State:
        return self.state

    def _stack_lin_jacobian_w(self):
        return jnp.stack([
            jnp.asarray(self.lin_jacobian_w[name][:, 6 + 3*i : 6 + 3*(i+1)])
            for i, name in enumerate(self.env.legs_order)
        ])

    def estimate_contact_states(self, contact_force, threshold, joint_torque) -> None:
        """See LegOdom.estimate_contact_states."""

        contact_force = jnp.asarray(contact_force)

        # x, y, z values for force
        if contact_force.shape == (4,3):
            mask = (jnp.sqrt(contact_force[:, 0]**2 + contact_force[:, 1]**2) <= contact_force[:, 2])
            contact_state = (contact_force != 0)[:, 0] & mask
            self.contact_forces = contact_force

        # single value for force
        elif contact_force.shape == (4,):
            contact_state = contact_force > threshold
            if not self.info_log:
                print("Estimating contact_force from estimated contact_state and joint torque")
                self.info_log = True

            self.contact_forces = estimate_contact_forces_jax(joint_torque=jnp.asarray(joint_torque),
                                                               contact_state=contact_state,
                                                               lin_jacobian_w_stack=self._stack_lin_jacobian_w())

        else:
            raise ValueError(f"contact_force has invalid shape: {contact_force.shape}")

        self.contact_states = contact_state

    def compute_foot_positions_B(self, joint_pos):
        """See LegOdom.compute_foot_positions_B."""

        model = self.env.mjModel
        data = self.env.mjData

        data.qpos[:7] = np.array([0, 0 , 0, 1.0, 0, 0, 0])
        data.qpos[7:] = np.asarray(joint_pos)

        mujoco.mj_kinematics(model, data)  # propagate qpos -> geom_xpos (mj_forward is not needed, cheaper)

        foot_positions = []
        for name in self.env.legs_order:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            foot_positions.append(data.geom_xpos[geom_id].copy())

        return jnp.asarray(np.array(foot_positions))

    def compute_leg_odometry(self, dt, base_orient, base_ang_vel, qdot, joint_torque, joint_pos, contact_state, contact_force, contact_state_threshold) -> None:
        """See LegOdom.compute_leg_odometry."""

        base_orient = jnp.asarray(base_orient)

        # create rotation matrix
        if base_orient.shape == (4,):  # if given as quaternion
            self.orient_rot = quat_to_rot_jax(orient=base_orient)  # base -> world
            self.orient_quat = base_orient
        else:  # if given as rotation matrix
            self.orient_rot = base_orient
            self.orient_quat = rot_to_quat_jax(orient=base_orient)

        # set mjData pos (MuJoCo needs plain numpy arrays)
        self.env.mjData.qpos[:] = np.concatenate([np.zeros(3), np.asarray(self.orient_quat), np.asarray(joint_pos)])
        self.env.mjData.qvel[:] = np.concatenate([np.zeros(6), np.asarray(qdot)])

        # calculate lineare jacobian
        mujoco.mj_forward(self.env.mjModel, self.env.mjData)  # for jacobian

        self.lin_jacobian_b = self.env.feet_jacobians(frame="base")
        self.lin_jacobian_w = self.env.feet_jacobians(frame="world")

        # contact estimation. if values for contact_force or contact_state are not provided, they will be estimated
        if contact_state is None:
            self.estimate_contact_states(contact_force=contact_force, threshold=contact_state_threshold, joint_torque=joint_torque)
        else:
            self.contact_states = jnp.asarray(contact_state)
            if contact_force is None:
                self.contact_forces = estimate_contact_forces_jax(joint_torque=jnp.asarray(joint_torque),
                                                                   contact_state=self.contact_states,
                                                                   lin_jacobian_w_stack=self._stack_lin_jacobian_w())
            else:
                self.contact_forces = jnp.asarray(contact_force)

        # foot position in base frame
        p_b = self.compute_foot_positions_B(joint_pos=joint_pos)
        self.p_b = p_b

        # motion estimation, vectorized across legs instead of a Python for loop
        lin_jacobian_b_stack = jnp.stack([jnp.asarray(self.lin_jacobian_b[name][:, 6:]) for name in self.env.legs_order])
        new_vel = _compute_motion_estimate_jax(self.orient_rot, jnp.asarray(base_ang_vel), p_b,
                                               lin_jacobian_b_stack, jnp.asarray(qdot), self.contact_states)
        new_pos = self.state.pos + new_vel * dt  # integrate velocity to get position <- estimation of previous step, not measurement

        self.state.pos = new_pos
        self.state.vel = new_vel
