import numpy as np
import mujoco
from mujoco import mjx
from gym_quadruped.quadruped_env import QuadrupedEnv
import jax.numpy as jnp

class LegOdom():
    """
    This class implements leg odometry for quadrupeds based on chapter 12.4 from the SLAM Handbook (https://github.com/SLAM-Handbook-contributors/slam-handbook-public-release).
    The estimation consists of a motion estimation and contact estimation. The state of the robot is defined as the position and velocity of the base.
    
    Motion Estimation:
    By exploiting the fact that legged robots only move when their leg is in contact with the ground (stance phase), we get an (noisy) estimation of the base velocity in world frame.

    Contact Estimation:
    A foot is considered in contact when its stationary over time, i.e. it does not slip. For that, the ground reaction force is used.
    """

    def __init__(self, model_name="aliengo", xp=np):
        self.env = QuadrupedEnv(robot=model_name) # legs_order = ('FL', 'FR', 'RL', 'RR')
        self.xp = xp
        self.init_pos = xp.zeros(3)
        self.init_vel = xp.zeros(3)

        model = self.env.mjModel
        self.foot_geom_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in self.env.legs_order
        ])
        self.foot_body_ids = np.asarray(model.geom_bodyid[self.foot_geom_ids])
        self._base_qpos = np.array([0., 0., 0., 1., 0., 0., 0.])

        if xp is jnp:
            self.mjx_model = mjx.put_model(model)
            self._base_qpos_j = jnp.asarray(self._base_qpos)
            self._foot_geom_ids_j = jnp.asarray(self.foot_geom_ids)
            self._foot_body_ids_j = jnp.asarray(self.foot_body_ids)
        
    def compute_foot_positions_B(self, joint_pos):
        return (self._foot_positions_jax(joint_pos)
                if self.xp is jnp
                else self._foot_positions_np(joint_pos))

    def _foot_positions_np(self, joint_pos):
        data = self.env.mjData
        data.qpos[:7] = self._base_qpos
        data.qpos[7:] = joint_pos
        mujoco.mj_kinematics(self.env.mjModel, data)
        return data.geom_xpos[self.foot_geom_ids].copy()

    def _foot_positions_jax(self, joint_pos):
        qpos = jnp.concatenate([self._base_qpos_j, joint_pos])
        data = mjx.make_data(self.mjx_model).replace(qpos=qpos)
        data = mjx.kinematics(self.mjx_model, data)
        return data.geom_xpos[self._foot_geom_ids_j]

    def motion_estimation(self, dt, base_orient, base_ang_vel, joint_pos, joint_vel, 
                          J_b, contact_state, prev_pos, prev_vel):
        """Estimate robot velocity based on informations from the legs, using equation (12.21) from SLAM Handbook.
        """

        p_b = self.compute_foot_positions_B(joint_pos=joint_pos)
        J_stack = self.xp.stack([J_b[i][:, 6 + 3*i : 6 + 3*(i+1)] for i in range(len(self.env.legs_order))]) # (4, 3, 3)
        q_dot = joint_vel.reshape(4, 3) # (4, 3)

        # Eq. 12.21 from SLAM handbook
        omega_cross_p = self.xp.cross(base_ang_vel, p_b)
        J_qdot = self.xp.stack([J_stack[i] @ q_dot[i] for i in range(4)]) # (4, 3)
        vels_b = -omega_cross_p - J_qdot
        mask = self.xp.asarray(contact_state, dtype=bool)

        count = mask.sum()
        safe_count = self.xp.maximum(count, 1)
        v_b = (vels_b * mask[:, None]).sum(axis=0) / safe_count

        new_vel = self.xp.where(count > 0, base_orient @ v_b, prev_vel)
        new_pos = prev_pos + new_vel * dt
        return new_pos, new_vel, p_b
