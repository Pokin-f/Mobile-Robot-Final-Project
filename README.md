# Mobile-Robot-Final-Project

# 🤖 Yaskawa Robotic Arm — Cup Grasping from Conveyor Belt

A 3-script pipeline for autonomous cup grasping using **Geometric Jacobian IK** with **Damped Least Squares (DLS)** in CoppeliaSim.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Mathematics](#mathematics)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

---

## Overview

This project implements a full **pick-and-place** pipeline for a 6-DOF Yaskawa robot arm grasping a moving cup on a conveyor belt inside CoppeliaSim simulation. The system uses:

- **Quintic Polynomial** trajectory planning for smooth, jerk-free motion
- **SLERP** (Spherical Linear Interpolation) for smooth orientation interpolation
- **Geometric Jacobian** for task-space to joint-space velocity mapping
- **Adaptive DLS** (Damped Least Squares) for singularity-robust IK solving

---

## Project Structure

```
yaskawa-cup-grasp/
├── getCup.py              # Step 1: Track cup position on conveyor belt
├── ef_trajectory.py       # Step 2: Plan end-effector trajectory
├── joint_trajectory.py    # Step 3: Execute IK control loop
├── cup_trajectory.npy     # (generated) Cup path data
├── ef_trajectory.npy      # (generated) EE trajectory data
├── cup_path_check.png     # (generated) Cup path visualization
├── trajectory_3d_plot.png # (generated) EE trajectory visualization
├── ik_performance_plot.png# (generated) IK tracking performance
├── requirements.txt
└── README.md
```

---

## Pipeline

```
getCup.py  ──►  cup_trajectory.npy  ──►  ef_trajectory.py  ──►  ef_trajectory.npy  ──►  joint_trajectory.py
  (Step 1)                                    (Step 2)                                        (Step 3)
Track cup                               Plan EE path                                    Run IK loop
on conveyor                             + orientation                                   in simulation
```

### Step 1 — `getCup.py`
Runs the simulation and records the cup's 6D pose `[x, y, z, roll, pitch, yaw]` at every timestep (`dt = 0.05s`). Automatically stops when the cup completes one full loop on the conveyor belt by detecting when the XY distance returns within 5 cm of the starting position.

### Step 2 — `ef_trajectory.py`
Loads the cup data and plans the robot's end-effector (EE) trajectory in 5 segments:

| Segment | Description |
|---------|-------------|
| Home → Via | Arc over the conveyor |
| Via → Wait | Approach wait point beside cup |
| Wait → Wait | Hold position, wait for cup |
| Wait → Grab | Move in to grasp |
| Grab → Lift | Lift cup vertically |

### Step 3 — `joint_trajectory.py`
Runs the full **closed-loop IK controller** at every simulation step, tracking the planned trajectory while respecting joint limits.

---

## Mathematics

### 1. Quintic Polynomial Trajectory

To ensure **zero velocity and zero acceleration** at waypoints (jerk-free motion):

$$s(t) = 10t^3 - 15t^4 + 6t^5, \quad t \in [0, 1]$$

**Boundary conditions verified:**
- $s(0) = 0$, $s(1) = 1$
- $s'(0) = 0$, $s'(1) = 0$ (zero velocity at endpoints)
- $s''(0) = 0$, $s''(1) = 0$ (zero acceleration at endpoints)

The position at each step is then:

$$\mathbf{p}(t) = \mathbf{p}_{start} + (\mathbf{p}_{end} - \mathbf{p}_{start}) \cdot s(t)$$

### 2. SLERP — Orientation Interpolation

Euler angle interpolation suffers from **Gimbal Lock** and non-uniform angular velocity. SLERP operates on **unit quaternions** on the 4D unit sphere:

$$\text{SLERP}(q_0, q_1, t) = q_0 \cdot \frac{\sin((1-t)\Omega)}{\sin(\Omega)} + q_1 \cdot \frac{\sin(t\Omega)}{\sin(\Omega)}$$

where $\Omega = \arccos(q_0 \cdot q_1)$ is the angle between quaternions.

In this project, the quintic $s(t)$ is used as the interpolation parameter instead of raw $t$, combining smooth position and orientation in one framework:

```python
s = 10*(t_norm**3) - 15*(t_norm**4) + 6*(t_norm**5)
rot = slerp_func([s])[0].as_euler('XYZ')
```

### 3. Geometric Jacobian

The **Geometric Jacobian** $J \in \mathbb{R}^{6 \times 6}$ maps joint velocities to end-effector velocity:

$$\dot{\mathbf{x}} = J(\mathbf{q}) \cdot \dot{\mathbf{q}}$$

For each revolute joint $i$:

$$J_{:3,\, i} = \mathbf{z}_i \times (\mathbf{p}_e - \mathbf{p}_i)  \quad \text{(linear velocity)}$$
$$J_{3:,\, i} = \mathbf{z}_i  \quad \text{(angular velocity)}$$

where $\mathbf{z}_i$ is the joint's rotation axis in world frame and $\mathbf{p}_e - \mathbf{p}_i$ is the vector from joint $i$ to the end-effector.

### 4. Manipulability & Singularity Detection

The **Yoshikawa Manipulability Measure** detects proximity to singularity:

$$w = \sqrt{|\det(J J^T)|}$$

- $w \rightarrow 0$: robot is at or near a **singular configuration** (loses a degree of freedom)
- $w$ large: robot can move freely in all directions

### 5. Adaptive Damped Least Squares (DLS)

Standard pseudoinverse $J^+ = J^T(JJ^T)^{-1}$ becomes numerically unstable near singularities. DLS adds a damping term $\lambda$:

$$J^+_{DLS} = J^T \left( J J^T + \lambda^2 I \right)^{-1}$$

The damping coefficient $\lambda$ adapts based on manipulability:

$$\lambda = \begin{cases} \lambda_{max} \left(1 - \dfrac{w}{w_0}\right)^2 + 0.05 & \text{if } w < w_0 \\ 0.05 & \text{if } w \geq w_0 \end{cases}$$

Parameters used: $\lambda_{max} = 0.4$, $w_0 = 0.04$

This gives **high damping near singularities** (safe, slow) and **low damping in normal regions** (accurate, fast).

### 6. Control Law — Feedforward + P-Controller

The desired task-space velocity combines a **feedforward term** (anticipate trajectory motion) with a **proportional feedback term** (correct tracking error):

$$\dot{\mathbf{x}}_{desired} = \dot{\mathbf{x}}_{ff} + K_p \cdot \mathbf{e}$$

**Position:**
$$\dot{\mathbf{x}}_{ff} = \frac{\mathbf{x}_{ref}[k] - \mathbf{x}_{ref}[k-1]}{\Delta t}$$
$$\mathbf{e}_{pos} = \mathbf{x}_{ref}[k] - \mathbf{x}_{actual}[k]$$

**Orientation error** (using rotation vector / axis-angle):
$$\mathbf{e}_{ori} = \left( R_{target} \cdot R_{current}^{-1} \right)^{\vee}$$

where $(\cdot)^\vee$ extracts the rotation vector (rotvec) from the rotation matrix, implemented as:
```python
w_err = (R_tar * R_cur.inv()).as_rotvec()
```

**Full control law:**
$$\dot{\mathbf{x}}_{desired} = \begin{bmatrix} \dot{\mathbf{x}}_{ff} + K_{p,pos} \cdot \mathbf{e}_{pos} \\ K_{p,ori} \cdot \mathbf{e}_{ori} \end{bmatrix}$$

**Joint update (Euler integration):**
$$\mathbf{q}[k+1] = \mathbf{q}[k] + \dot{\mathbf{q}} \cdot \Delta t = \mathbf{q}[k] + J^+_{DLS} \cdot \dot{\mathbf{x}}_{desired} \cdot \Delta t$$

| Parameter | Value |
|-----------|-------|
| $K_{p,pos}$ | 2.5 |
| $K_{p,ori}$ | 1.5 |
| $\Delta t$ | 0.05 s |
| $v_{max}$ | 0.8 m/s |
| $\omega_{max}$ | 1.5 rad/s |

---

## Installation

### Requirements
- CoppeliaSim (tested with EDU version)
- Python 3.8+

```bash
pip install -r requirements.txt
```

### `requirements.txt`
```
coppeliasim-zmqremoteapi-client
numpy
scipy
matplotlib
```

---

## Usage

Run the three scripts **in order**:

```bash
# Step 1: Record cup trajectory (run while simulation is open)
python getCup.py

# Step 2: Plan end-effector trajectory
python ef_trajectory.py

# Step 3: Execute IK control in simulation
python joint_trajectory.py
```

> **Note:** Each script connects to CoppeliaSim via ZMQ Remote API. Make sure CoppeliaSim is running before executing any script.

---

## Results

The IK controller generates three plots saved automatically:

| Plot | File | Description |
|------|------|-------------|
| Cup path | `cup_path_check.png` | Full conveyor loop recorded in Step 1 |
| EE trajectory | `trajectory_3d_plot.png` | Planned 3D path with key waypoints |
| IK performance | `ik_performance_plot.png` | Position error (mm), orientation error (rad), and 3D path tracking |

### Typical Performance
- Position tracking error: **< 5 mm** during free motion
- Orientation error: **< 0.05 rad** steady state
- Gripper closes at `step_B` — the moment the cup is within reach

---

## License

MIT License
