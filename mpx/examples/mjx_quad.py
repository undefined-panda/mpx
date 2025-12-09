import jax.numpy as jnp
import jax
import mujoco
# Update JAX configuration
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
 
import numpy as np
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.mujoco.visual import render_sphere, render_vector
 
import mpx.utils.mpc_wrapper as mpc_wrapper
import mpx.config.config_aliengo as config

from timeit import default_timer as timer

# ADDED BY ME
from pathlib import Path 
from tqdm import tqdm

# Set GPU device for JAX
# gpu_device = jax.devices('gpu')[0]
# jax.default_device(gpu_device)
 
# Define robot and scene parameters
robot_name = "aliengo"   # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
scene_name = "random_boxes"
robot_feet_geom_names = dict(FR='FR',FL='FL', RR='RR' , RL='RL')
robot_leg_joints = dict(FR=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', ],
                        FL=['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', ],
                        RR=['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', ],
                        RL=['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'])
mpc_frequency = config.mpc_frequency
state_observables_names = tuple(QuadrupedEnv.ALL_OBS)  # return all available state observables
 
# Initialize simulation environment
sim_frequency = 200.0
env = QuadrupedEnv(robot=robot_name,
                       scene=scene_name,
                       sim_dt = 1/sim_frequency,  # Simulation time step [s]
                       ref_base_lin_vel=0.0, # Constant magnitude of reference base linear velocity [m/s]
                       ground_friction_coeff=0.7,  # pass a float for a fixed value
                       base_vel_command_type="human",  # "forward", "random", "forward+rotate", "human"
                       state_obs_names=state_observables_names,  # Desired quantities in the 'state'
                       )
obs = env.reset(random=False)

# Define the MPC wrapper
mpc = mpc_wrapper.MPCControllerWrapper(config)
env.mjData.qpos = jnp.concatenate([config.p0, config.quat0,config.q0])
env.render()
counter = 0

# Main simulation loop
tau = jnp.zeros(config.n_joints)
tau_old = jnp.zeros(config.n_joints)
delay = 0 #int(0.007*sim_frequency)
print('Delay: ',delay)

q = config.q0.copy()
dq = jnp.zeros(config.n_joints)
mpc_time = 0
mpc.robot_height = config.robot_height
mpc.reset(env.mjData.qpos.copy(),env.mjData.qvel.copy())

# BEGIN ADDED BY ME: dataset, log_values, save_dataset, for-loop
dataset_path = Path.cwd() / "custom_datasets"
dataset_path.mkdir(exist_ok=True)

# add values to dataset
def log_values(dataset, sim_num, t, qpos, qvel, qacc, tau_total, contact, contact_forces, q):
    """
    Docstring for log_values
    
    :param dataset: dictionary to store values
    :param qpos: linear, angular and joint position
    :param qvel: linear, angular and joint velocity
    :param qacc: linear, angular and joint accelaration
    :param tau_total: tau + tau_fb
    :param contact: boolean array with elem for each contact (here: 4)
    :param contact_forces: 2d-array with array for each contact (here: 4)
    :param q: desired joint position/angle
    """

    dataset["time"][sim_num].append(t)

    # --- Base ---
    dataset["base_pos"][sim_num].append(qpos[0:3].copy())
    dataset["base_orient"][sim_num].append(qpos[3:7].copy())
    dataset["base_vel"][sim_num].append(qvel[0:3].copy())
    dataset["base_acc"][sim_num].append(qacc[0:3].copy())

    # --- Joint ---
    dataset["joint_pos"][sim_num].append(qpos[7:].copy())
    dataset["joint_vel"][sim_num].append(qvel[6:].copy())
    dataset["joint_acc"][sim_num].append(qacc[6:].copy())
    dataset["joint_torque"][sim_num].append(tau_total.copy())
    
    # --- Contact ---
    dataset["contact_state"][sim_num].append(contact.copy())
    dataset["contact_forces"][sim_num].append(contact_forces.copy())
    dataset["contact_pos_des"][sim_num].append(q.copy())

def save_dataset(dataset):
    # convert data to numpy array
    for i in dataset:
        dataset[i] = np.stack(dataset[i], axis=0)

    next_run = 1
    for file in Path(dataset_path).glob("*"):
        last_run = int(file.stem.split("run", 1)[1])
        if last_run >= next_run:
            next_run = last_run + 1

    np.savez(f"{dataset_path}/quad_dataset_run{next_run}.npz", **dataset)
    print("Data saved.")

# store values in lists, convert them later to numpy arrays
custom_dataset = {"time":[],
                  "base_pos":[],
                  "base_orient":[],
                  "base_vel":[],
                  "base_acc":[],
                  "joint_pos":[],
                  "joint_vel":[],
                  "joint_acc":[],
                  "joint_torque":[],
                  "contact_state":[],
                  "contact_forces":[],
                  "contact_pos_des":[]
                  }

num_simulations = 5
max_steps = 5000
q_init = env.mjData.qpos.copy()
dq_init = env.mjData.qvel.copy()

custom_dataset = {k : [[] for _ in range(num_simulations)] for k in custom_dataset} # add one list per simulation: "base_pos" : [[], [], ...]
log_and_save = True

for sim_num in range(num_simulations):

    # ADDED BY ME: reset environment after each simulation
    if not env.viewer.is_running():
        break

    env.reset(qpos=q_init, qvel=dq_init, random=False)
    mpc.reset(q_init, dq_init)
    tau = jnp.zeros(config.n_joints)
    env.render()

    for counter in tqdm(range(max_steps), desc=f"Running simulation {sim_num+1}"):
        if not env.viewer.is_running():
            break

        qpos = env.mjData.qpos.copy()
        qvel = env.mjData.qvel.copy()
        qacc = env.mjData.qacc.copy() # ADDED BY ME: get base and joint accelaration
        if (counter % (sim_frequency / mpc_frequency) == 0 or counter == 0):
            
            ref_base_lin_vel = env._ref_base_lin_vel_H
            ref_base_ang_vel = np.array([0., 0., env._ref_base_ang_yaw_dot])

            # ADDED BY ME: make the robot move on its own
            # sample a velocity in the direction the robot is facing
            if counter != 0:
                vx = np.random.uniform(0, 0.5) 
                ref_base_lin_vel = np.array([vx, 0., 0.])
            
                # sample a angular rotation to turn the robot to left/right
                if (counter % 1000) == 0:
                    az = np.random.uniform(-2, 2)
                    ref_base_ang_vel = np.array([0, 0, az])

            #print(f"base vel: {ref_base_lin_vel} \t base ang: {ref_base_ang_vel}")

            input = np.array([ref_base_lin_vel[0],ref_base_lin_vel[1],ref_base_lin_vel[2],
                            ref_base_ang_vel[0],ref_base_ang_vel[1],ref_base_ang_vel[2],
                            config.robot_height])
            
            contact_temp, _, contact_forces_temp = env.feet_contact_state(ground_reaction_forces=True) # ADDED BY ME: set parameter to get contact forces
            
            contact = np.array([contact_temp[robot_feet_geom_names[leg]] for leg in ['FL','FR','RL','RR']])
            contact_forces = np.array([contact_forces_temp[robot_feet_geom_names[leg]] for leg in ['FL','FR','RL','RR']]) # ADDED BY ME: get values for contact forces

            if counter != 0:
                for i in range(delay):
                    qpos = env.mjData.qpos.copy()
                    qvel = env.mjData.qvel.copy()
                    qacc = env.mjData.qacc.copy() # ADDED BY ME: get base and joint accelaration
                    # tau_fb = K@(x-np.concatenate([qpos,qvel]))

                    tau_fb = 10*(q-qpos[7:7+config.n_joints]) -2*(qvel[6:6+config.n_joints])
                    state, reward, is_terminated, is_truncated, info = env.step(action=tau + tau_fb)

                    t = env.mjData.time 
                    log_values(custom_dataset, sim_num, t, qpos, qvel, qacc, tau+tau_fb, contact, contact_forces, q) # ADDED BY ME: add values to dataset

                    counter += 1

            start = timer()
            tau, q, dq = mpc.run(qpos,qvel,input,contact)   
            stop = timer()
            #print("Time taken for MPC: ", stop-start)

            stop = timer()
            # for i in range(4):
            #     render_sphere(env.viewer,
            #                   collision_point[3*i:3*i+3],
            #                   0.2,
            #                   np.array([1, 0, 0, 0.5]),
            #                   ids[i])

        tau_fb = 10*(q-qpos[7:7+config.n_joints])-2*(qvel[6:6+config.n_joints])
        state, reward, is_terminated, is_truncated, info = env.step(action= tau + tau_fb)

        t = env.mjData.time
        log_values(custom_dataset, sim_num, t, qpos, qvel, qacc, tau+tau_fb, contact, contact_forces, q) # ADDED BY ME: add values to dataset

        # time.sleep(0.1)
        counter += 1
        env.render()

    print(f"\n----- Simulation {sim_num+1} finished -----\n")

env.close()

if log_and_save: save_dataset(custom_dataset) # ADDED BY ME: save dataset of simulations