import copy
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
import mpx.config.config_aliengo as config_aliengo
import mpx.config.config_go2 as config_go2

from timeit import default_timer as timer

# ADDED BY ME
from pathlib import Path
from tqdm import tqdm

# Set GPU device for JAX
# gpu_device = jax.devices('gpu')[0]
# jax.default_device(gpu_device)

# Define robot and scene parameters
robot_name = "aliengo"   # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
config = config_aliengo
scene_name = "flat" # "random_boxes"
robot_feet_geom_names = dict(FR='FR',FL='FL', RR='RR' , RL='RL')
robot_leg_joints = dict(FR=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', ],
                        FL=['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', ],
                        RR=['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', ],
                        RL=['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'])
mpc_frequency = config.mpc_frequency
state_observables_names = tuple(QuadrupedEnv.ALL_OBS)  # return all available state observables

# REPRODUCIBILITY: dedicated RNG for base-mass and command-velocity sampling, so a
# run can be replayed exactly by reusing the same SEED (independent of any other
# library touching the global numpy random state).
SEED = 0
rng = np.random.default_rng(SEED)

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

# BASE MASS/INERTIA RANDOMIZATION: the floating base is always the first body
# after the world in these MJCF models (id 1, e.g. "trunk" for aliengo, "base"
# for go2).
BASE_BODY_ID = 1
nominal_base_mass = env.mjModel.body_mass[BASE_BODY_ID].copy()
nominal_base_inertia = env.mjModel.body_inertia[BASE_BODY_ID].copy()  # principal moments (3,)
nominal_base_iquat = env.mjModel.body_iquat[BASE_BODY_ID].copy()     # principal-axes orientation
nominal_base_ipos = env.mjModel.body_ipos[BASE_BODY_ID].copy()       # CoM offset in the body frame
base_mass_offset_range = (0.0, 5.0)             # sampled base mass = nominal_mass + U(lo, hi) [kg]
inertia_density_offset_range = (0.0, 0.02)     # offset for the 3 inertia "density" terms [kg m^2]
rotation_offset_range = (-0.1, 0.1)              # small rotation-vector offset for principal axes [rad]
ipos_offset_range = (-0.02, 0.02)                # offset for the base CoM position [m]

# ADDED BY ME: scratch model/data holding the *nominal* (unrandomized) spatial
# inertia, captured once here before any per-run randomization is ever applied.
# Reused for compute_tau_components across every run/step (never the live
# env.mjModel/env.mjData), so we can log the nominal-model torque decomposition
# alongside the one for the randomized model actually driving the simulation.
nominal_dyn_model = copy.deepcopy(env.mjModel)
nominal_dyn_data = copy.deepcopy(env.mjData)

# Physically-consistent-by-construction inertia parametrization, ported to numpy
# from the "PrincipalTriangular" reparametrization in get_tri_dyn_params()
# (felan/jax/floating_base/mjx_dnea.py). Instead of perturbing principal moments
# directly and clipping to satisfy the triangle inequality after the fact, it
# reparametrizes them as J0=d1+d2, J1=d0+d2, J2=d0+d1 for non-negative "density"
# terms d0,d1,d2 -- any non-negative d's yield a J that is automatically positive
# and satisfies the triangle inequality, so no post-hoc correction is needed.

def expmap_to_quat(rotvec):
    """Convert a 3D rotation vector (exponential map) to a unit quaternion (w,x,y,z)."""
    theta = np.linalg.norm(rotvec)
    if theta < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rotvec / theta
    half = 0.5 * theta
    return np.concatenate([[np.cos(half)], axis * np.sin(half)])

def quat_mul(q1, q2):
    """Hamilton product of two (w, x, y, z) quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def principal_inertia_to_densities(inertia_diag):
    """Invert J0=d1+d2, J1=d0+d2, J2=d0+d1 to recover the nominal densities."""
    J0, J1, J2 = inertia_diag
    s = 0.5 * (J0 + J1 + J2)
    return np.array([s - J0, s - J1, s - J2])  # d0, d1, d2

def densities_to_principal_inertia(d):
    d0, d1, d2 = d
    return np.array([d1 + d2, d0 + d2, d0 + d1])  # J0, J1, J2

def sample_tri_dyn_params(nominal_mass, nominal_inertia_diag, nominal_iquat, nominal_ipos, mass_offset_range,
                           inertia_density_offset_range, rotation_offset_range, ipos_offset_range, rng,
                           density_epsilon=1e-6):
    """
    Physically-consistent-by-construction sampling of the base's spatial inertia:
    - mass: additive offset around the nominal mass.
    - rotational inertia: perturbed in "density" space; any non-negative densities
      give principal moments that automatically satisfy positivity and the
      triangle inequality, so no eigendecomposition/clipping is needed.
    - principal-axes orientation: nominal orientation composed with a small
      random rotation (as a unit quaternion via the exponential map), which is a
      valid quaternion by construction.
    - CoM offset: additive offset around the nominal position (any 3-vector is a
      valid CoM location, so no extra constraint is needed).
    Returns (base_mass [kg], base_inertia_diag (3,) [kg m^2], base_iquat (4,),
    base_ipos (3,) [m]).
    """
    new_mass = nominal_mass + rng.uniform(*mass_offset_range)

    d_nominal = principal_inertia_to_densities(nominal_inertia_diag)
    d_offset = rng.uniform(*inertia_density_offset_range, size=3)
    d_new = np.maximum(d_nominal + d_offset, density_epsilon)
    new_inertia_diag = densities_to_principal_inertia(d_new)

    rotvec = rng.uniform(*rotation_offset_range, size=3)
    new_iquat = quat_mul(nominal_iquat, expmap_to_quat(rotvec))
    new_iquat /= np.linalg.norm(new_iquat)

    new_ipos = nominal_ipos + rng.uniform(*ipos_offset_range, size=3)

    return new_mass, new_inertia_diag, new_iquat, new_ipos

def sample_base_mass(env, nominal_mass, nominal_inertia, mass_offset_range, rng):
    """
    Randomize the floating-base mass by adding an offset on top of the nominal
    mass, scaling its inertia tensor by the resulting mass ratio so the mass
    distribution stays physically consistent.
    Returns the sampled base mass [kg].
    """
    offset = rng.uniform(*mass_offset_range)
    new_mass = nominal_mass + offset
    scale = new_mass / nominal_mass
    env.mjModel.body_mass[BASE_BODY_ID] = new_mass
    env.mjModel.body_inertia[BASE_BODY_ID] = nominal_inertia * scale
    return float(env.mjModel.body_mass[BASE_BODY_ID])

def sample_base_spatial_inertia(env, nominal_mass, nominal_inertia, nominal_iquat, nominal_ipos, mass_offset_range,
                                 inertia_density_offset_range, rotation_offset_range, ipos_offset_range, rng):
    """
    Randomize the floating base's whole spatial inertia (mass + rotational
    inertia tensor + principal-axes orientation + CoM offset) using the
    physically-consistent-by-construction `sample_tri_dyn_params` parametrization,
    and applies it to the true simulated model.
    Returns (base_mass [kg], base_inertia_diag (3,) [kg m^2], base_iquat (4,),
    base_ipos (3,) [m]) so the sampled spatial inertia can be handed to the MPC
    as well, not just used to mutate the true simulated physics.
    """
    new_mass, new_inertia_diag, new_iquat, new_ipos = sample_tri_dyn_params(
        nominal_mass, nominal_inertia, nominal_iquat, nominal_ipos, mass_offset_range,
        inertia_density_offset_range, rotation_offset_range, ipos_offset_range, rng)

    env.mjModel.body_mass[BASE_BODY_ID] = new_mass
    env.mjModel.body_inertia[BASE_BODY_ID] = new_inertia_diag
    env.mjModel.body_iquat[BASE_BODY_ID] = new_iquat
    env.mjModel.body_ipos[BASE_BODY_ID] = new_ipos

    return float(env.mjModel.body_mass[BASE_BODY_ID]), new_inertia_diag, new_iquat, new_ipos

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

# ADDED BY ME: decompose the equations of motion M(q)qdd + C(q,qd)qd + g(q) = tau
def compute_tau_components(model, data, qpos, qvel, qacc):
    """
    Decompose the generalized-coordinate equations of motion at the given
    qpos/qvel/qacc into:
      tau_m = M(q) @ qacc          (inertial term)
      tau_g = g(q)                 (gravity term, isolated with qvel = 0)
      tau_c = qfrc_bias - tau_g    (Coriolis/centrifugal term)
    mj_rne(model, data, flg_acc=0, result) computes M(q)*qacc + C(q,qd) with the
    inertial term removed, i.e. exactly qfrc_bias == C(q,qd)*qd + g(q); zeroing
    qvel before calling it isolates gravity alone (the velocity-product term
    vanishes), so no separate gravity-only model/solver is needed.

    `model`/`data` must be a scratch copy created once per simulation run at
    reset (see dyn_model/dyn_data below) -- not the live env.mjModel/env.mjData
    -- so dataset logging never perturbs the live simulation. Each call only
    overwrites qpos/qvel on that scratch `data` and re-runs mj_forward, which is
    far cheaper than a full MjData deepcopy every timestep.
    Returns (tau_m, tau_c, tau_g), each shape (nv,).
    """
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)  # refresh qM/qfrc_bias/kinematics for this state

    nv = model.nv
    M = np.zeros((nv, nv))
    mujoco.mj_fullM(model, M, data.qM)
    tau_m = M @ qacc

    tau_bias = data.qfrc_bias.copy()  # C(q,qd)*qd + g(q)

    data.qvel[:] = 0
    tau_g = np.zeros(nv)
    mujoco.mj_rne(model, data, 0, tau_g)

    tau_c = tau_bias - tau_g
    return tau_m, tau_c, tau_g

# add values to dataset
def log_values(target_dict, dt, t, qpos, qvel, qacc, tau_total, contact_states, contact_pos, contact_forces, q, base_mass, model, data):
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
    :param base_mass: sampled floating-base mass for this simulation run [kg]
    :param model: scratch MjModel (randomized spatial inertia, matches the
        simulation/control model) for compute_tau_components -- not the live
        env.mjModel
    :param data: scratch MjData paired with `model` -- not the live env.mjData
    """
    tau_m, tau_c, tau_g = compute_tau_components(model, data, qpos, qvel, qacc)
    # ADDED BY ME: same decomposition but with the nominal (unrandomized) spatial
    # inertia, so the dataset also has the dynamics as the nominal robot model
    # would predict them (for comparison against the true/simulated model above).
    tau_m_nom, tau_c_nom, tau_g_nom = compute_tau_components(nominal_dyn_model, nominal_dyn_data, qpos, qvel, qacc)
    # ADDED BY ME: real (randomized/simulated) minus nominal-model prediction --
    # the mismatch the nominal-model MPC/controller is implicitly compensating for.
    diff_tau_m_nom = tau_m - tau_m_nom
    diff_tau_c_nom = tau_c - tau_c_nom
    diff_tau_g_nom = tau_g - tau_g_nom

    target_dict["dt"].append(dt)
    target_dict["time"].append(t)

    # --- Base ---
    target_dict["base_pos"].append(qpos[0:3].copy())
    target_dict["base_orient"].append(qpos[3:7].copy()) # stored as quaternion
    target_dict["base_vel"].append(qvel[0:3].copy())
    target_dict["base_ang_vel"].append(qvel[3:6].copy())
    target_dict["base_acc"].append(qacc[0:6].copy())
    target_dict["base_mass"].append(base_mass)

    # --- Joint ---
    target_dict["joint_pos"].append(qpos[7:].copy())
    target_dict["joint_vel"].append(qvel[6:].copy())
    target_dict["joint_acc"].append(qacc[6:].copy())
    target_dict["joint_torque"].append(tau_total.copy())

    # --- Dynamics decomposition (full generalized coords: base 6 DOF + joints) ---
    target_dict["tau_m"].append(tau_m.copy())
    target_dict["tau_c"].append(tau_c.copy())
    target_dict["tau_g"].append(tau_g.copy())
    target_dict["tau_m_nom"].append(tau_m_nom.copy())
    target_dict["tau_c_nom"].append(tau_c_nom.copy())
    target_dict["tau_g_nom"].append(tau_g_nom.copy())
    target_dict["diff_tau_m_nom"].append(diff_tau_m_nom.copy())
    target_dict["diff_tau_c_nom"].append(diff_tau_c_nom.copy())
    target_dict["diff_tau_g_nom"].append(diff_tau_g_nom.copy())

    # --- Contact ---
    target_dict["contact_states"].append(contact_states.copy())
    target_dict["contact_pos"].append(contact_pos.copy())
    target_dict["contact_forces"].append(contact_forces.copy())
    target_dict["contact_pos_des"].append(q.copy())

def save_dataset(dataset, dataset_path):
    # convert data to numpy array
    for i in dataset:
        dataset[i] = np.stack(dataset[i], axis=0)

    next_run = 1
    for file in Path(dataset_path).glob("quad_mass_dataset_run*.npz"):
        last_run = int(file.stem.split("run", 1)[1])
        if last_run >= next_run:
            next_run = last_run + 1

    np.savez(f"{dataset_path}/quad_mass_dataset_run{next_run}.npz", **dataset)
    print(f"Data saved in: {dataset_path}/quad_mass_dataset_run{next_run}.npz")

# store values in lists, convert them later to numpy arrays
custom_dataset = {"dt":[],
                  "time":[],
                  "base_pos":[],
                  "base_orient":[],
                  "base_vel":[],
                  "base_ang_vel":[],
                  "base_acc":[],
                  "base_mass":[],
                  "joint_pos":[],
                  "joint_vel":[],
                  "joint_acc":[],
                  "joint_torque":[],
                  "tau_m":[],
                  "tau_c":[],
                  "tau_g":[],
                  "tau_m_nom":[],
                  "tau_c_nom":[],
                  "tau_g_nom":[],
                  "diff_tau_m_nom":[],
                  "diff_tau_c_nom":[],
                  "diff_tau_g_nom":[],
                  "contact_states":[],
                  "contact_pos":[],
                  "contact_forces":[],
                  "contact_pos_des":[]
                  }

num_simulations = 25
max_steps = 1000
q_init = env.mjData.qpos.copy()
dq_init = env.mjData.qvel.copy()

custom_dataset = {k : [[] for _ in range(num_simulations)] for k in custom_dataset} # add one list per simulation: "base_pos" : [[], [], ...]
dt = env.simulation_dt # constant for each simulation
log_and_save = True
old_dt = 0

height_limit = 0.25 # re-do run when height is below this value (indicating robot fell)

sim_num = 0
MAX_ATTEMPTS_PER_RUN = 5
while sim_num < num_simulations:
    # ADDED BY ME: reset environment after each simulation
    if not env.viewer.is_running():
        break

    for attempt in range(MAX_ATTEMPTS_PER_RUN):
        env.reset(qpos=q_init, qvel=dq_init, random=False)

        # ADDED BY ME: sample a new base mass + spatial inertia for this simulation run
        base_mass, base_inertia_diag, base_iquat, base_ipos = sample_base_spatial_inertia(
            env, nominal_base_mass, nominal_base_inertia, nominal_base_iquat, nominal_base_ipos,
            base_mass_offset_range, inertia_density_offset_range,
            rotation_offset_range, ipos_offset_range, rng)
        print(f"Sampled base mass: {base_mass:.3f} kg base_inertia_diag: {base_inertia_diag} base_iquat: {base_iquat} base_ipos: {base_ipos}")

        # ADDED BY ME: scratch model/data for compute_tau_components, created once per
        # simulation run (after the spatial-inertia randomization above, so they
        # reflect the sampled mass/inertia) -- never the live env.mjModel/env.mjData.
        dyn_model = copy.deepcopy(env.mjModel)
        dyn_data = copy.deepcopy(env.mjData)

        mpc.reset(q_init, dq_init)
        tau = jnp.zeros(config.n_joints)
        env.render()

        # ADDED BY ME: make the robot move on its own by sampling linear velocity and angular velocity for each simulation
        vx = rng.uniform(-0.5, 1)
        vy = rng.uniform(-0.1, 0.1)
        az = rng.uniform(-0.5, 0.5)

        ref_base_lin_vel = np.array([vx, vy, 0.])
        ref_base_ang_vel = np.array([0, 0, az])

        print(f"ref_base_lin_vel = {ref_base_lin_vel}")
        print(f"ref_base_ang_vel = {ref_base_ang_vel}")

        fell = False
        run_buffer = {k: [] for k in custom_dataset.keys()}
        for counter in tqdm(range(max_steps), desc=f"Running simulation {sim_num+1}"):
            if not env.viewer.is_running():
                break

            qpos = env.mjData.qpos.copy()
            qvel = env.mjData.qvel.copy()
            qacc = env.mjData.qacc.copy() # ADDED BY ME: get base and joint accelaration

            if qpos[2] < height_limit:
                print(f"\nWARNING: Robot base height: {qpos[2]} (robot fell down). Re-running simulation no. {sim_num+1}\n")
                fell = True
                break
            if (counter % (sim_frequency / mpc_frequency) == 0 or counter == 0):

                input = np.array([ref_base_lin_vel[0],ref_base_lin_vel[1],ref_base_lin_vel[2],
                                ref_base_ang_vel[0],ref_base_ang_vel[1],ref_base_ang_vel[2],
                                config.robot_height])

                contact_temp, contact_pos_temp, contact_forces_temp = env.feet_contact_state(ground_reaction_forces=True) # ADDED BY ME: set parameter to get contact forces

                contact_states = np.array([contact_temp[robot_feet_geom_names[leg]] for leg in ['FL','FR','RL','RR']])
                contact_forces = np.array([contact_forces_temp[robot_feet_geom_names[leg]] for leg in ['FL','FR','RL','RR']]) # ADDED BY ME: get values for contact forces

                contact_pos = np.full((4, 3), np.nan, dtype=np.float32) # ADDED BY ME: store contact position in a matrix with each leg
                for i, leg in enumerate(['FL','FR','RL','RR']):
                    if contact_states[i]:
                        contacts = contact_pos_temp[robot_feet_geom_names[leg]]
                        contact_pos[i] = contacts[0].pos

                # DEBUG: check the vertical force balance against the sampled base mass.
                # contact_forces is (4, 3) -- one [Fx, Fy, Fz] row per leg -- so column 2 is Fz.
                # force_z = contact_forces[:, 2]
                # print(f"Fz per leg: {np.round(force_z, 2)}  total Fz: {np.sum(force_z):.2f} N  mass*g: {base_mass*9.81:.2f} N", flush=True)

                if counter != 0:
                    for i in range(delay):
                        qpos = env.mjData.qpos.copy()
                        qvel = env.mjData.qvel.copy()
                        qacc = env.mjData.qacc.copy() # ADDED BY ME: get base and joint accelaration
                        # tau_fb = K@(x-np.concatenate([qpos,qvel]))

                        tau_fb = 10*(q-qpos[7:7+config.n_joints]) -2*(qvel[6:6+config.n_joints])
                        state, reward, is_terminated, is_truncated, info = env.step(action=tau + tau_fb)

                        t = env.simulation_time
                        log_values(run_buffer, dt, t, qpos, qvel, qacc, tau+tau_fb, contact_states, contact_pos, contact_forces, q, base_mass, dyn_model, dyn_data) # ADDED BY ME: add values to dataset

                        counter += 1

                start = timer()
                tau, q, dq = mpc.run(qpos,qvel,input,contact_states,base_mass=base_mass,
                                    base_inertia_diag=base_inertia_diag,base_iquat=base_iquat,
                                    base_ipos=base_ipos)
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

            t = env.simulation_time
            log_values(run_buffer, dt, t, qpos, qvel, qacc, tau+tau_fb, contact_states, contact_pos, contact_forces, q, base_mass, dyn_model, dyn_data) # ADDED BY ME: add values to dataset

            # time.sleep(0.1)
            counter += 1
            env.render()

        if not fell:
            for k in custom_dataset.keys():
                custom_dataset[k][sim_num] = run_buffer[k]
            sim_num += 1
            break
    else:
        print(f"WARNING: Sim {sim_num+1} failed after {MAX_ATTEMPTS_PER_RUN} attempts")
        sim_num += 1

    print(f"\n----- Simulation {sim_num} finished -----\n")

env.close()

if log_and_save: save_dataset(custom_dataset, dataset_path) # ADDED BY ME: save dataset of simulations
