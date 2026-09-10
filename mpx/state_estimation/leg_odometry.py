import numpy as np
import mujoco
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

    def motion_estimation(self, dt, base_orient, base_ang_vel, joint_pos, joint_vel, 
                          J_b, contact_state, prev_pos, prev_vel):
        """Estimate robot velocity based on informations from the legs, using equation (12.21) from SLAM Handbook.
        """

        p_b = self.compute_foot_positions_B(joint_pos=joint_pos)
        self.p_b = p_b

        J_stack = self.xp.stack([J_b[leg][:, 6 + 3*i : 6 + 3*(i+1)] for i, leg in enumerate(self.env.legs_order)]) # (4, 3, 3)
        q_dot = joint_vel.reshape(4, 3) # (4, 3)

        # Eq. 12.21 from SLAM handbook
        omega_cross_p = self.xp.cross(base_ang_vel, p_b)
        J_qdot = self.xp.stack([J_stack[i] @ q_dot[i] for i in range(4)]) # (4, 3)
        vels_b = -omega_cross_p - J_qdot
        mask = self.xp.asarray(contact_state, dtype=bool)

        count = mask.sum()
        safe_count = jnp.maximum(count, 1)
        v_b = (vels_b * mask[:, None]).sum(axis=0) / safe_count

        new_vel = jnp.where(count > 0, base_orient @ v_b, prev_vel)
        new_pos = prev_pos + new_vel * dt
        return new_pos, new_vel
