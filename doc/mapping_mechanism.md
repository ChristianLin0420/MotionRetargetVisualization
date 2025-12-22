# Motion Retargeting: Mapping Mechanism Deep Dive

This document provides a comprehensive, step-by-step explanation of the motion retargeting pipeline, including the mathematical foundations, key functions, and implementation details.

---

## Table of Contents

1. [Overview](#overview)
2. [Stage 1: SMPL Human Model](#stage-1-smpl-human-model)
3. [Stage 2: Shape Fitting](#stage-2-shape-fitting)
4. [Stage 3: Keypoint Extraction](#stage-3-keypoint-extraction)
5. [Stage 4: Inverse Kinematics](#stage-4-inverse-kinematics)
6. [Mathematical Foundations](#mathematical-foundations)
7. [Key Functions Reference](#key-functions-reference)
8. [Data Flow Diagram](#data-flow-diagram)

---

## Overview

Motion retargeting transfers human motion capture data to a humanoid robot while preserving the essential characteristics of the movement. The process involves several transformation stages:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     MOTION RETARGETING PIPELINE                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌───────────┐  │
│  │   AMASS     │───>│    SMPL     │───>│   Target    │───>│    G1     │  │
│  │  MoCap Data │    │  Skeleton   │    │  Keypoints  │    │   Robot   │  │
│  │   (poses)   │    │ (24 joints) │    │ (13 joints) │    │ (29 DOFs) │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └───────────┘  │
│                                                                          │
│    Input: θ, β        FK: θ → J         Select 13        IK: J → q       │
│                                         key joints                       │
└──────────────────────────────────────────────────────────────────────────┘
```

### Key Challenges Addressed

| Challenge | Solution |
|-----------|----------|
| Different body proportions | Shape fitting with scaling parameters |
| Different joint structures | 13-joint correspondence mapping |
| Different degrees of freedom | Levenberg-Marquardt IK optimization |
| Temporal smoothness | Regularization in optimization |
| Joint limits | Clamping to valid ranges |

---

## Stage 1: SMPL Human Model

### What is SMPL?

**SMPL (Skinned Multi-Person Linear Model)** is a parametric human body model that represents body shape and pose using a compact set of parameters.

### Mathematical Model

The SMPL model generates a 3D mesh $\mathbf{M}$ and joint positions $\mathbf{J}$ from two parameter sets:

$$
\mathbf{M}(\boldsymbol{\beta}, \boldsymbol{\theta}) = W(T_P(\boldsymbol{\beta}, \boldsymbol{\theta}), J(\boldsymbol{\beta}), \boldsymbol{\theta}, \mathcal{W})
$$

Where:
- $\boldsymbol{\beta} \in \mathbb{R}^{10}$: Shape parameters (body proportions)
- $\boldsymbol{\theta} \in \mathbb{R}^{72}$: Pose parameters (24 joints × 3 axis-angle)
- $T_P$: Template mesh deformation function
- $J(\boldsymbol{\beta})$: Joint locations as function of shape
- $\mathcal{W}$: Skinning weights

### SMPL Joint Hierarchy (24 joints)

```
                    [0] Pelvis
                        │
         ┌──────────────┼──────────────┐
         │              │              │
     [1] L_Hip      [3] Spine      [2] R_Hip
         │              │              │
     [4] L_Knee     [6] Spine1     [5] R_Knee
         │              │              │
     [7] L_Ankle    [9] Spine2     [8] R_Ankle
         │              │              │
    [10] L_Foot    [12] Neck      [11] R_Foot
                        │
                   [15] Head
                        │
         ┌──────────────┼──────────────┐
         │                             │
    [13] L_Collar                 [14] R_Collar
         │                             │
    [16] L_Shoulder               [17] R_Shoulder
         │                             │
    [18] L_Elbow                  [19] R_Elbow
         │                             │
    [20] L_Wrist                  [21] R_Wrist
         │                             │
    [22] L_Hand                   [23] R_Hand
```

### Implementation: `setup_smpl_parsers()`

```python
# File: scripts/g1/process_humanml3d_g1.py

def setup_smpl_parsers():
    """Initialize SMPL parsers for all genders."""
    smpl_parser_n = SMPL_Parser(model_path="../../data/smpl", gender="neutral")
    smpl_parser_m = SMPL_Parser(model_path="../../data/smpl", gender="male")
    smpl_parser_f = SMPL_Parser(model_path="../../data/smpl", gender="female")
    return smpl_parser_n, smpl_parser_m, smpl_parser_f
```

### Forward Kinematics

Given pose parameters $\boldsymbol{\theta}$, SMPL computes joint positions through forward kinematics:

```python
verts, joints = smpl_parser.get_joints_verts(pose_aa, beta, trans)
# joints: (N_frames, 24, 3) - 3D positions for all 24 joints
```

**Mathematical formulation:**

For each joint $i$ in the kinematic chain:

$$
\mathbf{T}_i^{\text{global}} = \mathbf{T}_{\text{parent}(i)}^{\text{global}} \cdot \mathbf{T}_i^{\text{local}}(\theta_i)
$$

$$
\mathbf{J}_i = \mathbf{T}_i^{\text{global}} \cdot \begin{bmatrix} 0 \\ 0 \\ 0 \\ 1 \end{bmatrix}
$$

Where $\mathbf{T}_i^{\text{local}}$ is computed from the axis-angle rotation $\theta_i$.

---

## Stage 2: Shape Fitting

### Purpose

The human body and robot have different proportions. Shape fitting optimizes SMPL's $\boldsymbol{\beta}$ parameters and a global scale to match the robot's link lengths.

### Mathematical Formulation

**Objective:**

$$
\min_{\boldsymbol{\beta}, s} \sum_{i=1}^{N} \left\| \mathbf{p}_i^{\text{robot}} - s \cdot \mathbf{p}_i^{\text{smpl}}(\boldsymbol{\beta}) \right\|^2 + \lambda \cdot \mathcal{L}_{\text{symmetric}}
$$

Where:
- $\mathbf{p}_i^{\text{robot}}$: Robot link position in T-pose
- $\mathbf{p}_i^{\text{smpl}}(\boldsymbol{\beta})$: SMPL joint position with shape $\boldsymbol{\beta}$
- $s$: Global scale factor
- $\mathcal{L}_{\text{symmetric}}$: Symmetry regularization

### Symmetry Regularization

Ensures left-right symmetry is preserved:

$$
\mathcal{L}_{\text{symmetric}} = \sum_{(l,r) \in \mathcal{P}} \left\| \mathbf{p}_l - \text{mirror}(\mathbf{p}_r) \right\|^2
$$

Where $\text{mirror}(\mathbf{p}) = [p_x, -p_y, p_z]^T$ (reflection across the sagittal plane) and $\mathcal{P}$ is the set of symmetric joint pairs.

### Implementation: `fit_robot_shape_g1.py`

```python
# File: scripts/g1/fit_robot_shape_g1.py

# Optimization variables
shape_new = Variable(torch.zeros([1, 10]), requires_grad=True)  # β
scale = Variable(torch.ones([1]), requires_grad=True)           # s

optimizer = torch.optim.Adam([shape_new, scale], lr=0.05)

for iteration in range(2001):
    # Get SMPL joints with current shape
    verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, shape_new, trans)
    
    # Apply scaling
    root_pos = joints[:, 0]
    joints = (joints - joints[:, 0]) * scale + root_pos
    
    # Position loss: match robot links
    diff = fk_return.global_translation[:, :, link_pick_idx] - joints[:, smpl_link_pick_idx]
    loss_g = diff.norm(dim=-1).mean()
    
    # Symmetry loss
    symmetric_loss = compute_symmetric_loss(joints)
    
    # Total loss
    loss = loss_g + symmetric_loss * 10
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Output

```python
optimized_shape_scale = {
    "shape": β*,   # Optimized shape parameters (1, 10)
    "scale": s*    # Optimized scale factor (scalar)
}
# Saved to: data/g1/optimized_shape_scale_g1.pkl
```

---

## Stage 3: Keypoint Extraction

### Purpose

Select 13 key joints from SMPL that have direct correspondence to robot links. These serve as IK targets.

### Joint Correspondence Table

| Index | SMPL Joint | SMPL Index | Robot Link | Purpose |
|-------|------------|------------|------------|---------|
| 0 | Pelvis | 0 | `pelvis` | Root/base |
| 1 | L_Hip | 1 | `left_hip_pitch_link` | Left leg root |
| 2 | L_Knee | 4 | `left_knee_link` | Left knee |
| 3 | L_Ankle | 7 | `left_ankle_roll_link` | Left foot |
| 4 | R_Hip | 2 | `right_hip_pitch_link` | Right leg root |
| 5 | R_Knee | 5 | `right_knee_link` | Right knee |
| 6 | R_Ankle | 8 | `right_ankle_roll_link` | Right foot |
| 7 | L_Shoulder | 16 | `left_shoulder_roll_link` | Left arm root |
| 8 | L_Elbow | 18 | `left_elbow_link` | Left elbow |
| 9 | L_Wrist | 20 | `left_wrist_yaw_link` | Left hand |
| 10 | R_Shoulder | 17 | `right_shoulder_roll_link` | Right arm root |
| 11 | R_Elbow | 19 | `right_elbow_link` | Right elbow |
| 12 | R_Wrist | 21 | `right_wrist_yaw_link` | Right hand |

### Visual Mapping

```
        SMPL (24 joints)                    Robot (13 targets)
        ================                    ==================
        
              ○ Head                              
              │                                   
           ○──┼──○ Shoulders              ●───────●
              │                            │       │
              ○ Spine                     ●│       │●
              │                            │       │
           ○──┼──○ Hips              ●────┼───────┼────●
              │                           │       │
           ○──┼──○ Knees                  ●       ●
              │                           │       │
           ○──┼──○ Ankles                 ●       ●
              │
           ○──┼──○ Feet
           
     (24 joints total)              (13 keypoints selected)
```

### Implementation

```python
# File: scripts/g1/process_humanml3d_g1.py

smpl_link_pick = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle",
    "R_Hip", "R_Knee", "R_Ankle",
    "L_Shoulder", "L_Elbow", "L_Wrist",
    "R_Shoulder", "R_Elbow", "R_Wrist",
]

robot_link_pick = [
    'pelvis',
    'left_hip_pitch_link', 'left_knee_link', 'left_ankle_roll_link',
    'right_hip_pitch_link', 'right_knee_link', 'right_ankle_roll_link',
    'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_yaw_link',
    'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_yaw_link',
]

# Get indices
smpl_link_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_link_pick]
robot_link_pick_idx = [robot_link_names.index(j) for j in robot_link_pick]

# Extract keypoints from SMPL joints
verts, joints = smpl_parser_n.get_joints_verts(pose_aa, shape_new, trans)
target_keypoints = joints[:, smpl_link_pick_idx]  # Shape: (N_frames, 13, 3)
```

### Scaling Application

Before using as IK targets, keypoints are scaled to match robot proportions:

$$
\mathbf{J}_{\text{scaled}} = (\mathbf{J} - \mathbf{o}) \cdot s^* + \mathbf{o}
$$

Where $\mathbf{o}$ is the root offset and $s^*$ is the optimized scale factor.

```python
# Apply optimized scale from shape fitting
offset = joints[:, 0] - trans
joints = (joints - offset.unsqueeze(1)) * optimized_scale + offset.unsqueeze(1)
```

---

## Stage 4: Inverse Kinematics

### Purpose

Find robot joint angles $\mathbf{q}$ that position robot links as close as possible to target keypoints.

### Mathematical Formulation

**Primary Objective (Position Matching):**

$$
E_{\text{pos}}(\mathbf{q}) = \sum_{i=1}^{N} w_i \left\| \mathbf{p}_i^{\text{target}} - \text{FK}(\mathbf{q})_i \right\|^2
$$

Where:
- $\mathbf{p}_i^{\text{target}}$: Target keypoint position (from SMPL)
- $\text{FK}(\mathbf{q})_i$: Robot link position given joint angles $\mathbf{q}$
- $w_i$: Per-joint weight

**Secondary Objective (Orientation Matching for Hands):**

$$
E_{\text{ori}}(\mathbf{q}) = \left\| \mathbf{R}_{\text{hand}}^{\text{target}} - \mathbf{R}_{\text{hand}}^{\text{robot}} \right\|_F^2
$$

**Temporal Smoothness:**

$$
E_{\text{smooth}}(\mathbf{q}) = \lambda \sum_{t=1}^{T} \left\| \mathbf{q}_t - \mathbf{q}_{t-1} \right\|^2
$$

**Complete Objective:**

$$
\min_{\mathbf{q}} \quad E_{\text{pos}}(\mathbf{q}) + \alpha \cdot E_{\text{ori}}(\mathbf{q}) + \lambda \cdot E_{\text{smooth}}(\mathbf{q})
$$

$$
\text{subject to:} \quad \mathbf{q}_{\min} \leq \mathbf{q} \leq \mathbf{q}_{\max} \quad \text{(joint limits)}
$$

### Levenberg-Marquardt Algorithm

The optimization uses the **Levenberg-Marquardt (LM)** algorithm, which combines gradient descent and Gauss-Newton methods.

#### Update Rule

$$
\mathbf{q}_{k+1} = \mathbf{q}_k - (\mathbf{J}^T\mathbf{J} + \mu \mathbf{I})^{-1} \mathbf{J}^T \mathbf{r}
$$

Where:
- $\mathbf{J}$: Jacobian matrix $\frac{\partial \mathbf{r}}{\partial \mathbf{q}}$
- $\mathbf{r}$: Residual vector (position differences)
- $\mu$: Damping factor (adaptive)
- $\mathbf{I}$: Identity matrix

#### Jacobian Computation

The Jacobian relates changes in joint angles to changes in end-effector positions:

$$
J_{ij} = \frac{\partial p_i}{\partial q_j}
$$

Computed numerically via finite differences:

$$
J_{ij} \approx \frac{p_i(\mathbf{q} + \epsilon \mathbf{e}_j) - p_i(\mathbf{q})}{\epsilon}
$$

```python
def compute_batch_jacobian(var_dof_pose, ..., epsilon=1e-4):
    """Compute Jacobian via finite differences."""
    for j in range(num_joints):
        # Perturb joint j
        var_dof_pose_perturbed = var_dof_pose.clone()
        var_dof_pose_perturbed[0, :, j, 0] += epsilon
        
        # Compute FK with perturbed pose
        fk_perturbed = robot_fk.fk_batch(...)
        
        # Jacobian column j
        jacobian[:, :, j] = (fk_perturbed - fk_base) / epsilon
```

#### Adaptive Damping

The damping factor $\mu$ adapts based on optimization progress:

$$
\mu_{k+1} = \begin{cases}
\mu_k \cdot \rho_{\uparrow} & \text{if } \mathcal{L}_{k+1} > \mathcal{L}_k \quad \text{(increase damping)} \\
\mu_k \cdot \rho_{\downarrow} & \text{if } \mathcal{L}_{k+1} \leq \mathcal{L}_k \quad \text{(decrease damping)}
\end{cases}
$$

Where $\rho_{\uparrow} = 10$ and $\rho_{\downarrow} = 0.5$.

```python
if loss_updated > last_loss:
    # Increase damping (more gradient descent-like)
    lambda_val *= lambda_increase_factor  # e.g., 10
else:
    # Decrease damping (more Gauss-Newton-like)
    lambda_val *= lambda_decrease_factor  # e.g., 0.5
```

### Implementation: `compute_batch_LM_step()`

```python
# File: scripts/g1/process_humanml3d_g1.py

def compute_batch_LM_step(jacobian_mat, diff, const_last_dof_pos, identity_matrix,
                          lambda_val=1e-3, smooth_weight=1e-2):
    """Compute the batch Levenberg-Marquardt step."""
    
    # JᵀJ: Approximate Hessian
    JTJ = torch.bmm(jacobian_mat.transpose(1, 2), jacobian_mat)
    
    # Damping: μI
    damping = lambda_val * identity_matrix
    
    # Smoothness: penalize deviation from previous frame
    smooth_mat = smooth_weight * identity_matrix
    
    # Build block-diagonal system for temporal coupling
    H = torch.block_diag(*JTJ_damped)
    
    # Add off-diagonal blocks for smoothness
    for i in range(1, num_frames):
        H[i*n:(i+1)*n, (i-1)*n:i*n] = -smooth_mat
        H[(i-1)*n:i*n, i*n:(i+1)*n] = -smooth_mat
    
    # Gradient: Jᵀr
    grad = torch.bmm(jacobian_mat.transpose(1, 2), diff.unsqueeze(-1))
    
    # Solve: (JᵀJ + μI + S)⁻¹ Jᵀr
    step = torch.linalg.solve(H, grad_flat)
    
    return step.reshape(num_frames, num_joints, 1)
```

### Optimization Loop

```python
# File: scripts/g1/process_humanml3d_g1.py (in retarget_data function)

for iteration in range(5):
    # 1. Compute Jacobian
    jacobian_pos, jacobian_ori = compute_batch_jacobian(
        dof_pos, root_trans_offset, gt_root_rot, robot_fk, ...
    )
    
    # 2. Compute position/orientation differences
    pos_diff, ori_diff = compute_batch_diff(
        dof_pos, root_trans_offset, gt_root_rot, 
        joints[:, smpl_link_pick_idx], ...
    )
    
    # 3. Compute LM step
    step = compute_batch_LM_step(jacobian, diff, dof_pos, identity_matrix)
    
    # 4. Propose new joint angles
    propose_dof_pos = (dof_pos - step).clamp_(
        robot_fk.joints_range[:, 0],  # min limits
        robot_fk.joints_range[:, 1]   # max limits
    )
    
    # 5. Accept/reject based on loss
    if loss_updated < last_loss:
        dof_pos = propose_dof_pos
```

---

## Mathematical Foundations

### 1. Axis-Angle Representation

Joint rotations are represented as axis-angle vectors:

$$
\boldsymbol{\theta} = \theta \cdot \hat{\mathbf{n}}
$$

Where:
- $\theta$: Rotation angle (scalar, in radians)
- $\hat{\mathbf{n}}$: Unit rotation axis (3D vector, $\|\hat{\mathbf{n}}\| = 1$)

**Conversion to rotation matrix (Rodrigues' formula):**

$$
\mathbf{R} = \mathbf{I} + \sin(\theta) [\hat{\mathbf{n}}]_\times + (1 - \cos(\theta)) [\hat{\mathbf{n}}]_\times^2
$$

Where $[\hat{\mathbf{n}}]_\times$ is the skew-symmetric matrix of $\hat{\mathbf{n}}$:

$$
[\hat{\mathbf{n}}]_\times = \begin{bmatrix}
0 & -n_z & n_y \\
n_z & 0 & -n_x \\
-n_y & n_x & 0
\end{bmatrix}
$$

### 2. Forward Kinematics (FK)

Robot FK computes link positions from joint angles:

$$
\mathbf{p}_i = \text{FK}(\mathbf{q})_i = \mathbf{T}_0^i(\mathbf{q}) \cdot \mathbf{o}_i
$$

Where $\mathbf{T}_0^i$ is the transformation from base to link $i$:

$$
\mathbf{T}_0^i = \prod_{j=0}^{i} \mathbf{T}_{j-1}^j(q_j)
$$

```python
# Robot uses axis-aligned rotations
pose_aa = torch.cat([
    root_pose[None, :, None],           # Root orientation
    robot_rotation_axis * dof_pos       # Joint angles × rotation axes
], axis=2)

fk_return = robot_fk.fk_batch(pose_aa, root_trans)
link_positions = fk_return['global_translation']  # (1, N_frames, 30, 3)
```

### 3. Rotation Error (SO(3))

For hand orientation matching, the error is computed in the tangent space of SO(3):

$$
\boldsymbol{\omega} = \text{vee}\left( \frac{1}{2} (\mathbf{R}_{\text{target}}^T \mathbf{R}_{\text{current}} - \mathbf{R}_{\text{current}}^T \mathbf{R}_{\text{target}}) \right)
$$

The $\text{vee}$ operator extracts the rotation vector from a skew-symmetric matrix:

$$
\text{vee}\left( \begin{bmatrix}
0 & -\omega_z & \omega_y \\
\omega_z & 0 & -\omega_x \\
-\omega_y & \omega_x & 0
\end{bmatrix} \right) = \begin{bmatrix} \omega_x \\ \omega_y \\ \omega_z \end{bmatrix}
$$

```python
def vee(skew_tensor):
    """Extract rotation vector from skew-symmetric matrix."""
    return torch.stack([
        skew_tensor[:, :, 2, 1],  # ω_x
        skew_tensor[:, :, 0, 2],  # ω_y  
        skew_tensor[:, :, 1, 0]   # ω_z
    ], dim=2)

# Rotation error: log(R_target^T · R_current)
delta_rot = 0.5 * (R_target.T @ R_current - R_current.T @ R_target)
error_vec = vee(delta_rot)  # 3D rotation error vector
```

### 4. Block-Tridiagonal System

For temporal smoothness, frames are coupled through a block-tridiagonal linear system:

$$
\begin{bmatrix}
\mathbf{H}_0 + \mathbf{S} & -\mathbf{S} & 0 & \cdots & 0 \\
-\mathbf{S} & \mathbf{H}_1 + 2\mathbf{S} & -\mathbf{S} & \cdots & 0 \\
0 & -\mathbf{S} & \mathbf{H}_2 + 2\mathbf{S} & \cdots & 0 \\
\vdots & \vdots & \ddots & \ddots & \vdots \\
0 & 0 & \cdots & -\mathbf{S} & \mathbf{H}_{T-1} + \mathbf{S}
\end{bmatrix}
\begin{bmatrix}
\delta\mathbf{q}_0 \\
\delta\mathbf{q}_1 \\
\delta\mathbf{q}_2 \\
\vdots \\
\delta\mathbf{q}_{T-1}
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{g}_0 \\
\mathbf{g}_1 \\
\mathbf{g}_2 \\
\vdots \\
\mathbf{g}_{T-1}
\end{bmatrix}
$$

Where:
- $\mathbf{H}_t = \mathbf{J}_t^T \mathbf{J}_t + \mu \mathbf{I}$: Per-frame approximate Hessian
- $\mathbf{S} = \lambda \mathbf{I}$: Smoothness coupling matrix
- $\mathbf{g}_t = \mathbf{J}_t^T \mathbf{r}_t$: Per-frame gradient

---

## Key Functions Reference

### Data Loading

| Function | File | Purpose |
|----------|------|---------|
| `load_humanml3d_data()` | `process_humanml3d_g1.py` | Load AMASS motion data |
| `setup_smpl_parsers()` | `process_humanml3d_g1.py` | Initialize SMPL models |
| `read_captions_from_file()` | `process_humanml3d_g1.py` | Load text descriptions |

### SMPL Processing

| Function | File | Purpose |
|----------|------|---------|
| `smpl_parser.get_joints_verts()` | `smpl_parser.py` | SMPL forward kinematics |
| `fix_height_smpl_vanilla()` | `process_humanml3d_g1.py` | Ground contact correction |
| `compute_hand_global_orientations()` | `process_humanml3d_g1.py` | Extract hand rotations |

### Inverse Kinematics

| Function | File | Purpose |
|----------|------|---------|
| `compute_batch_jacobian()` | `process_humanml3d_g1.py` | Numerical Jacobian computation |
| `compute_batch_diff()` | `process_humanml3d_g1.py` | Position/orientation error |
| `compute_batch_LM_step()` | `process_humanml3d_g1.py` | LM optimization step |
| `vee()` | `process_humanml3d_g1.py` | Skew-symmetric to vector |

### Robot FK

| Function | File | Purpose |
|----------|------|---------|
| `Humanoid_Batch.fk_batch()` | `torch_g1_humanoid_batch.py` | Batch robot FK |

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          COMPLETE DATA FLOW                             │
└─────────────────────────────────────────────────────────────────────────┘

      ┌─────────────┐
      │   AMASS     │
      │  .npz files │
      │ (poses, β)  │
      └──────┬──────┘
             │
             ▼
    ┌────────────────────┐
    │ load_humanml3d_data│ ←── index.csv (frame ranges, captions)
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │  process_data_dict │ ←── occlusion_data (skip bad sequences)
    │  • Downsample 120→30 Hz
    │  • Segment by frame range
    │  • Fix ground height
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐      ┌────────────────────┐
    │    retarget_data   │◄─────│ optimized_shape_   │
    │                    │      │ scale_g1.pkl       │
    │  ┌──────────────┐  │      │  • β* (shape)      │
    │  │ SMPL FK      │  │      │  • s* (scale)      │
    │  │ pose→joints  │  │      └────────────────────┘
    │  └──────┬───────┘  │
    │         │          │
    │         ▼          │
    │  ┌──────────────┐  │
    │  │ Scale joints │  │
    │  │ by s*, shape │  │
    │  │ by β*        │  │
    │  └──────┬───────┘  │
    │         │          │
    │         ▼          │
    │  ┌──────────────┐  │
    │  │ Select 13    │  │
    │  │ keypoints    │  │
    │  └──────┬───────┘  │
    │         │          │
    │         ▼          │
    │  ┌──────────────┐  │
    │  │ LM IK Loop   │──┼──┐
    │  │ (5 iters)    │  │  │
    │  │              │  │  │  ┌─────────────────┐
    │  │ • Jacobian   │  │  │  │ Robot FK        │
    │  │ • Diff       │  │  ├─►│ q → link pos    │
    │  │ • LM step    │  │  │  └─────────────────┘
    │  │ • Clamp      │  │  │
    │  └──────┬───────┘  │  │
    │         │          │  │
    │         ▼          │◄─┘
    │  ┌──────────────┐  │
    │  │ Height fix   │  │
    │  │ (min z=0.015)│  │
    │  └──────────────┘  │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │    OUTPUT PKL      │
    │                    │
    │ • root_trans_offset│ (T, 3)
    │ • root_rot         │ (T, 4) quaternion
    │ • dof              │ (T, 29) joint angles
    │ • global_translation│ (T, 30, 3) robot links
    │ • mocap_global_trans│ (T, 24, 3) SMPL joints
    │ • target_keypoints │ (T, 13, 3) IK targets
    │ • smpl_pose_aa     │ (T, 72) original pose
    │ • smpl_betas       │ (10,) shape params
    │ • captions         │ text descriptions
    └────────────────────┘
```

---

## Appendix: G1 Robot Joint Configuration

### Joint Limits and Axes

Each robot joint rotates around a specific axis. The rotation is computed as:

$$
\mathbf{R}_j = \exp(q_j \cdot [\mathbf{a}_j]_\times)
$$

Where $\mathbf{a}_j$ is the rotation axis for joint $j$.

```python
G1_ROTATION_AXIS = torch.tensor([
    # Left leg (6 DOF)
    [1, 0, 0],  # left_hip_pitch
    [0, 1, 0],  # left_hip_roll
    [0, 0, 1],  # left_hip_yaw
    [1, 0, 0],  # left_knee
    [1, 0, 0],  # left_ankle_pitch
    [0, 1, 0],  # left_ankle_roll
    
    # Right leg (6 DOF)
    [1, 0, 0],  # right_hip_pitch
    [0, 1, 0],  # right_hip_roll
    [0, 0, 1],  # right_hip_yaw
    [1, 0, 0],  # right_knee
    [1, 0, 0],  # right_ankle_pitch
    [0, 1, 0],  # right_ankle_roll
    
    # Torso (3 DOF)
    [0, 0, 1],  # waist_yaw
    [0, 1, 0],  # waist_roll
    [1, 0, 0],  # waist_pitch
    
    # Left arm (7 DOF)
    [1, 0, 0],  # left_shoulder_pitch
    [0, 1, 0],  # left_shoulder_roll
    [0, 0, 1],  # left_shoulder_yaw
    [1, 0, 0],  # left_elbow
    [0, 1, 0],  # left_wrist_roll
    [1, 0, 0],  # left_wrist_pitch
    [0, 0, 1],  # left_wrist_yaw
    
    # Right arm (7 DOF)
    [1, 0, 0],  # right_shoulder_pitch
    [0, 1, 0],  # right_shoulder_roll
    [0, 0, 1],  # right_shoulder_yaw
    [1, 0, 0],  # right_elbow
    [0, 1, 0],  # right_wrist_roll
    [1, 0, 0],  # right_wrist_pitch
    [0, 0, 1],  # right_wrist_yaw
])
```

### Total: 29 Degrees of Freedom

| Body Part | DOFs | Joints |
|-----------|------|--------|
| Left Leg | 6 | hip (3) + knee (1) + ankle (2) |
| Right Leg | 6 | hip (3) + knee (1) + ankle (2) |
| Torso | 3 | waist (3) |
| Left Arm | 7 | shoulder (3) + elbow (1) + wrist (3) |
| Right Arm | 7 | shoulder (3) + elbow (1) + wrist (3) |
| **Total** | **29** | |

---

## References

1. **SMPL**: Loper, M., et al. "SMPL: A Skinned Multi-Person Linear Model." *ACM Trans. Graphics*, 2015.
2. **AMASS**: Mahmood, N., et al. "AMASS: Archive of Motion Capture as Surface Shapes." *ICCV*, 2019.
3. **Levenberg-Marquardt**: Levenberg, K. "A Method for the Solution of Certain Non-Linear Problems in Least Squares." *Quarterly of Applied Mathematics*, 1944.
4. **Rodrigues' Formula**: Rodrigues, O. "Des lois géométriques qui régissent les déplacements d'un système solide." *Journal de Mathématiques Pures et Appliquées*, 1840.
