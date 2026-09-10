# Legged Robot Localization under Uncertain Dynamics

This repo implements a state estimation algorithm for the _**Robot Learning: Integrated Project**_ at TU Darmstadt. The goal is to estimate the state of a quadruped when the dynamics are uncertain, e.g. due to an additional payload.

This repo was forked from [mpx](https://github.com/iit-DLSLab/mpx), a framework for legged robot MPC written in JAX. The [second part](#ip-2) of the project uses [felan](https://github.com/undefined-panda/felan) for a physics-encoded neural network for learned inertia estimation. For more details, refer to the original repo.

## Setup
Clone the repo and run
```
cd mpx && git submodule update --init --recursive
```
Create and activate the conda environment:
```
conda create -n mpx_env python=3.13 -y
conda activate mpx_env
```
Install with pip:
```
pip install -e .
```

### Dataset creation
Run a simulation script. The dataset is stored [here](custom_datasets/).

For this project, Aliengo was used. [This](mpx/examples/mjx_quad.py) simulation script was used for IP1, where commands for linear and angular velocity were sampled for automated movement:
```
python mpx/examples/mjx_quad.py
```

For IP2, [this](mpx/examples/mjx_quad_mass_random.py) was used. An additional payload is simulated by sampling a base mass offset, inertia density offset and rotation offset which are added to the nominal values:
```
python mpx/examples/mjx_quad_mass_random.py
```

## IP 1
Implementation of a Kalman Filter for estimating the state of a quadruped, using dynamics equation for base acceleration estimation.

### Kalman Filter

The state of the robot is defined as the position $\mathbf{p}$ and velocity $\mathbf{v}$ at time step $k$:
$
\begin{equation}
    \mathbf{x}_k = \begin{bmatrix}\mathbf{p}_k \\\mathbf{v}_k \end{bmatrix}, 
    \qquad
    \mathbf{p}_k \in \mathbb{R}^3,\; \mathbf{v}_k \in \mathbb{R}^3
\end{equation}
$
calculated with Euler discretization with sampling time $\Delta t$ as $\mathbf{T}_s = \Delta t \mathbf{I}_3$ where $\mathbf{I}$ stands for the identity matrix:
$
\begin{align}
    \mathbf{p}_{k} &= \mathbf{p}_{k-1} + \mathbf{T}_s \mathbf{v}_{k-1}, \\
    \mathbf{v}_{k} &= \mathbf{v}_{k-1} + \mathbf{T}_s \mathbf{a}_{k-1} .
\end{align}
$
with base acceleration $\mathbf{a}$. This is used as the control input vector $\mathbf{u}$ of the Kalman Filter. The prediction step is therefore defined as:
$
\begin{equation}
    \begin{bmatrix}
        \mathbf{p}_k\\
        \mathbf{v}_k
    \end{bmatrix} = 
    \begin{bmatrix}
        \mathbf{I}_3 & \mathbf{T}_s \\ \mathbf{0} & \mathbf{I}_3
    \end{bmatrix}
    \begin{bmatrix}
    \mathbf{p}_{k-1} \\ \mathbf{v}_{k-1}    
    \end{bmatrix} + 
    \begin{bmatrix}
        \mathbf{0} \\ \mathbf{T}_s
    \end{bmatrix} [\mathbf{a}_{k-1}]
\end{equation}
$

**Leg odometry** is used as a measurement for velocity, where the base velocity in world frame is expressed by
$
\begin{equation}
    \mathbf{v}_b^w = -\boldsymbol{\omega}_b^w \times \mathbf{f}_p(\mathbf{q}) - \mathbf{J}(\mathbf{q})\dot{\mathbf{q}}
\end{equation}
$
with angular velocity of the base in world frame $\boldsymbol{\omega}_b^w$, foot position in base frame $\mathbf{f}_p$, linear Jacobian of the leg $\mathbf{J}$ and joint velocity $\dot{\mathbf{q}}$ (ref. SLAM Handbook Ch. 12). 

The measurement $\mathbf{z}_k$ is made up of $\mathbf{v}_b^w$.

### Dynamics Model
The **dynamics model** is estimated by applying Newton's second law of motion $F = ma + mg$:
$
\begin{equation}
    \mathbf{a}_{k-1} = \frac{\sum_{i=1}^{N} \mathbf{c}_i \cdot (\mathbf{F}_i - m\mathbf{g})}{m}.
\end{equation}
$
where $\mathbf{F}$ is calculated as
$
\begin{equation}
    \mathbf{F}_i = (\mathbf{J}_i^{\top})^{-1} \boldsymbol{\tau}_i.
\end{equation}
$
coming from the relationship between joint torque $\boldsymbol{\tau}$ and contact force $\mathbf{F}$ for each foot, with the assumption of $\mathbf{J}$ being invertible.

### Experiments
This approach is tested with the following methods:
- Kalman filter estimation using the ground truth $a_k$ from the simulation
- Leg Odometry alone
- Kalman filter estimation using leg odometry, without considering the dynamics model ($\mathbf{a}_k = 0$)
- Kalman Filter estimation using Leg odometry with considering the dynamics model with ground truth contact force from the simulation
- Kalman Filter estimation using Leg odometry with considering the dynamics model with estimated contact force

## IP 2
### State Extension
Angular velocity $\boldsymbol{\omega}$ and the flattened contact forces $\mathbf{F}$ are added to the state:
$
\begin{equation}
    \mathbf{x}_k = \begin{bmatrix}\mathbf{p}_k \\ \mathbf{v}_k \\ \boldsymbol{\omega}_k \\ \mathbf{F}_k \end{bmatrix},
    \quad
    \mathbf{p}_k, \mathbf{v}_k, \boldsymbol{\omega}_k \in \mathbb{R}^3, \mathbf{F}_k \in \mathbb{R}^{12}
\end{equation}
$
where $\boldsymbol{\omega}$ is estimated in the same way as $\mathbf{p}$ and $\mathbf{v}$:
$
\begin{align}
    \boldsymbol{\omega}_{k} &= \boldsymbol{\omega}_{k-1} + \mathbf{T}_s \boldsymbol{\alpha}_{k-1}, \\
\end{align}
$
with angular base acceleration $\boldsymbol{\alpha}$.
In addition to that, a **contact state estimation** based on the generalized momentum observer is included. The observer is defined as:
$
\begin{equation}
    \begin{bmatrix} \dot{\hat{\mathbf{p}}} \\ \dot{\hat{\mathbf{f}}} \end{bmatrix}
    =
    \begin{bmatrix} \mathbf{0} & -\mathbf{J}^{T} \\ \mathbf{0} & \mathbf{0} \end{bmatrix}
    \begin{bmatrix} \hat{\mathbf{p}} \\ \hat{\mathbf{f}} \end{bmatrix}
    +
    \begin{bmatrix} \bar{\boldsymbol{\tau}} \\ \mathbf{0} \end{bmatrix}
    +
    \begin{bmatrix} \mathbf{L}k_1(\mathbf{p} - \hat{\mathbf{p}}) \\ \mathbf{L}^2k_2(\mathbf{p} - \hat{\mathbf{p}}) \end{bmatrix}
\end{equation}
$
with the measured generalized momentum $\mathbf{p} = \mathbf{M}(\mathbf{x})\mathbf{v}$, the joint-space mass matrix $\mathbf{M}$, and the estimated generalized momentum $\hat{\mathbf{p}}$ and estimated contact forces at the four feet $\hat{\mathbf{f}}$. 
The compensated torque is given by $\bar{\boldsymbol{\tau}} = \boldsymbol{\tau}_m + \mathbf{C}^{T}\mathbf{v} - \mathbf{g}$, with motor torques $\boldsymbol{\tau}_m$, Coriolis matrix $\mathbf{C}$, and gravity vector $\mathbf{g}$. The matrix $\mathbf{L}$ is the observer gain, and the correction terms $k_1$ and $k_2$ are defined element-wise as
$
\begin{align}
    k_1(s) := q(s) \quad k_2(s) := \operatorname{sign}(s) + q(s)
\end{align}
$
with $q(s) := \operatorname{sign}(s)\, |s|^{1/2} + s$ and $\operatorname{sign}(s) = 1$ for $s > 0$ and $\operatorname{sign}(s) = -1$ for $s < 0$.

### Dynamics Model Extension
The **dynamics model** is now the Rigid-Body-Dynamics equation:
$
\begin{equation}
    \mathbf{M(q)\ddot{q}} + \mathbf{c(q,\dot{q})} + \mathbf{g(q)} = \mathbf{S}^T\boldsymbol{\tau} + \mathbf{J}^T \mathbf{F},
\end{equation}
$
with the generalized coordinates $\mathbf{q} = [\mathbf{q}_b^T, \mathbf{q}_j^T]^T \in \mathbb{R}^{6+n}$ consisting of the base pose $\mathbf{q}_b$ (6 DoF) and the joint positions $\mathbf{q}_j$ ($n$ DoF), the joint-space inertia matrix $\mathbf{M(q)}$, the Coriolis and gravitational terms $\mathbf{c(q,\dot{q})}$ and $\mathbf{g(q)}$ and a selection matrix $\mathbf{S}$ that maps the joint torques onto the actuated coordinates.

The inertia matrix can be partitioned according to the base and joint coordinates as
$
\begin{equation}
    \mathbf{M(q)} = 
    \begin{bmatrix} \mathbf{H}_B & \mathbf{H}_{BL} \\ \mathbf{H}_{LB} & \mathbf{H}_L \end{bmatrix},
\end{equation}
$
where $\mathbf{H}_B \in \mathbb{R}^{6 \times 6}$ is the base inertia, $\mathbf{H}_L \in \mathbb{R}^{n \times n}$ the joint inertia, and $\mathbf{H}_{BL} = \mathbf{H}_{LB}^T \in \mathbb{R}^{6 \times n}$ the coupling block between the two. Since only the base acceleration is required for the Kalman filter, it is sufficient to consider the first six rows, which describe the dynamics of the unactuated base. The equation for the base acceleration can be reduced to:
$
\begin{equation}
    \mathbf{\ddot{q}}_B = \mathbf{H}_B^{-1} \left( \mathbf{J}_B^T \mathbf{F} - \mathbf{H}_{BL} \mathbf{\ddot{q}}_L - \mathbf{c}_B - \mathbf{g}_B \right),
\end{equation}
$

Two approaches are implemented: computing the base acceleration in a separate function or include it into the prediction model of the Kalman Filter. For the latter, the prediction step changes to:
$
\begin{equation}
    \mathbf{A}\cdot \mathbf{x} + \mathbf{B}\cdot \mathbf{u} = 
    \begin{bmatrix}
        \mathbf{I}_3 & \mathbf{T}_s & 0 & 0 \\
        0 & \mathbf{I}_3 & 0 & (H_B^{-1} \cdot J^TF)[:3] \\
        0 & 0 & \mathbf{I}_3 & (H_B^{-1} \cdot J^TF)[3:] \\
        0 & 0 & 0 & \mathbf{I}_{12}
    \end{bmatrix}
    \cdot
    \begin{bmatrix}
        \mathbf{p}_k \\ \mathbf{v}_k \\ \boldsymbol{\omega}_k \\ \mathbf{F}_k
    \end{bmatrix}
    +
    \begin{bmatrix}
        0 & 0 \\ H_B^{-1} \cdot (-H_{BL}) & H_B^{-1} \cdot (-\mathbf{c-g}) \\ 0 & 0
    \end{bmatrix}
    \cdot
    \begin{bmatrix}
        \mathbf{\ddot{q}}_L \\ 1
    \end{bmatrix}
\end{equation}
$
