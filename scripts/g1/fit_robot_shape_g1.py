import os
import sys
sys.path.append(os.getcwd())

from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from phc.smpllib.smpl_parser import (
    SMPL_Parser,
)

import joblib
import torch
from torch.autograd import Variable
from phc.smpllib.smpl_parser import SMPL_BONE_ORDER_NAMES
from phc.utils.torch_g1_humanoid_batch import Humanoid_Batch, G1_ROTATION_AXIS

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def visualize_matching_single(g1_positions, smpl_positions, iteration):
    """
    Visualize the matching between G1 robot and SMPL model joints from different angles
    Args:
        g1_positions: (n_links, 3) array of G1 joint positions
        smpl_positions: (n_links, 3) array of SMPL joint positions
        iteration: current iteration number
    """
    fig = plt.figure(figsize=(20, 15))
    
    # Define different viewing angles (elevation, azimuth)
    views = [
        (30, 45),   # perspective view
        (0, 0),     # front view
        (0, 90),    # side view
        (90, 0),    # top view
    ]
    
    for idx, (elev, azim) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, idx, projection='3d')
        
        # Plot G1 joints
        ax.scatter(g1_positions[:, 0], g1_positions[:, 1], g1_positions[:, 2], 
                  color='blue', label='G1 joints', s=100)
        
        # Plot SMPL joints
        ax.scatter(smpl_positions[:, 0], smpl_positions[:, 1], smpl_positions[:, 2], 
                  color='red', label='SMPL joints', s=100)
        
        # Draw lines between corresponding joints
        for i in range(len(g1_positions)):
            ax.plot([g1_positions[i, 0], smpl_positions[i, 0]],
                    [g1_positions[i, 1], smpl_positions[i, 1]],
                    [g1_positions[i, 2], smpl_positions[i, 2]], 
                    color='green', alpha=0.3)
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set view angle
        ax.view_init(elev=elev, azim=azim)
        
        # Set title for each subplot
        view_names = ['Perspective View', 'Front View', 'Side View', 'Top View']
        ax.set_title(f'{view_names[idx-1]} - Iteration {iteration}')
        
        # Set consistent axis limits
        max_range = max([
            g1_positions.max() - g1_positions.min(),
            smpl_positions.max() - smpl_positions.min()
        ])
        mid_x = (g1_positions[:, 0].mean() + smpl_positions[:, 0].mean()) / 2
        mid_y = (g1_positions[:, 1].mean() + smpl_positions[:, 1].mean()) / 2
        mid_z = (g1_positions[:, 2].mean() + smpl_positions[:, 2].mean()) / 2
        
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        if idx == 1:  # Only show legend on the first subplot
            ax.legend()
    
    plt.tight_layout()
    plt.show()

bh2_fk = Humanoid_Batch() # load forward kinematics model
#### Define corresonpdances between h1 and smpl joints
links_names = [
    'pelvis',
    'left_hip_pitch_link',
    'left_hip_roll_link',
    'left_hip_yaw_link',
    'left_knee_link',
    'left_ankle_pitch_link',
    'left_ankle_roll_link',
    'right_hip_pitch_link',
    'right_hip_roll_link',
    'right_hip_yaw_link',
    'right_knee_link',
    'right_ankle_pitch_link',
    'right_ankle_roll_link',
    'waist_yaw_link',
    'waist_roll_link',
    'torso_link',
    'left_shoulder_pitch_link',
    'left_shoulder_roll_link',
    'left_shoulder_yaw_link',
    'left_elbow_link',
    'left_wrist_roll_link',
    'left_wrist_pitch_link',
    'left_wrist_yaw_link',
    'right_shoulder_pitch_link',
    'right_shoulder_roll_link',
    'right_shoulder_yaw_link',
    'right_elbow_link',
    'right_wrist_roll_link',
    'right_wrist_pitch_link',
    'right_wrist_yaw_link',
]
robot_joint_names = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint"
    "right_wrist_yaw_joint",
]

link_pick = ['pelvis',
    'left_hip_pitch_link',
    'left_knee_link',
    'left_ankle_roll_link',
    'right_hip_pitch_link',
    'right_knee_link',
    'right_ankle_roll_link',
    'left_shoulder_roll_link',
    'left_elbow_link',
    'left_wrist_yaw_link',
    'right_shoulder_roll_link',
    'right_elbow_link',
    'right_wrist_yaw_link',
]
smpl_link_pick = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
]


link_pick_visual = ['pelvis',
    'left_hip_pitch_link',
    'left_knee_link',
    'left_ankle_roll_link',
    'right_hip_pitch_link',
    'right_knee_link',
    'right_ankle_roll_link',
    'left_shoulder_roll_link',
    'left_elbow_link',
    'left_wrist_yaw_link',
    'left_wrist_yaw_link',
    'right_shoulder_roll_link',
    'right_elbow_link',
    'right_wrist_yaw_link',
    'right_wrist_yaw_link',
]
smpl_link_visual = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
    "R_Shoulder",
    "R_Elbow",
    "R_Hand",
    "R_Wrist",
]


link_pick_idx = [ links_names.index(j) for j in link_pick]
smpl_link_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_link_pick]

link_visual_idx = [ links_names.index(j) for j in link_pick_visual]
smpl_link_visual_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_link_visual]


#### Preparing fitting varialbes
device = torch.device("cpu")

dof_pos = torch.zeros((1, 29))

initial_joint_position_dict = {
    # "left_hip_pitch_joint": -0.20,
    # "right_hip_pitch_joint": -0.20,
    # "left_knee_joint": 0.42,
    # "right_knee_joint": 0.42,
    # "left_ankle_pitch_joint": -0.23,
    # "right_ankle_pitch_joint": -0.23,
    # "left_elbow_joint": 1.57,
    # "right_elbow_joint": 1.57,
    # "left_hip_pitch_joint": -0.312,
    # "right_hip_pitch_joint": -0.312,
    # "left_knee_joint": 0.669,
    # "right_knee_joint": 0.669,
    # "left_ankle_pitch_joint": -0.363,
    # "right_ankle_pitch_joint": -0.363,
    "left_elbow_joint": 1.57,
    "right_elbow_joint": 1.57,
}
for joint_name, joint_value in initial_joint_position_dict.items():
    dof_pos[0, robot_joint_names.index(joint_name)] = joint_value

pose_aa = torch.cat([torch.zeros((1, 1, 3)), G1_ROTATION_AXIS * dof_pos[..., None]], axis = 1)


root_trans = torch.zeros((1, 1, 3))    

###### prepare SMPL default pause for fitting
pose_aa_stand = np.zeros((1, 72))
rotvec = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).as_rotvec()
pose_aa_stand[:, :3] = rotvec
pose_aa_stand = pose_aa_stand.reshape(-1, 24, 3)
pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('L_Shoulder')] = sRot.from_euler("xyz", [0, 0, -np.pi/2],  degrees = False).as_rotvec()
pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('R_Shoulder')] = sRot.from_euler("xyz", [0, 0, np.pi/2],  degrees = False).as_rotvec()
pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 72))

smpl_parser_n = SMPL_Parser(model_path="../../data/smpl", gender="neutral")

###### Shape fitting
trans = torch.zeros([1, 3])
beta = torch.zeros([1, 10])
verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, beta , trans)
offset = joints[:, 0] - trans
root_trans_offset = trans + offset

fk_return = bh2_fk.fk_batch(pose_aa[None, ], root_trans_offset[None, 0:1])

shape_new = Variable(torch.zeros([1, 10]).to(device), requires_grad=True)
scale = Variable(torch.ones([1]).to(device), requires_grad=True)
optimizer_shape = torch.optim.Adam([shape_new, scale],lr=0.05)


num_iterations = 2001
global_translations = np.zeros((num_iterations, len(link_pick_idx), 3))
smpl_global_translations = np.zeros((num_iterations, len(link_pick_idx), 3))

symmetric_pairs = [
    ("L_Shoulder", "R_Shoulder"),
    ("L_Elbow", "R_Elbow"),
    ("L_Wrist", "R_Wrist"),
    ("L_Hip", "R_Hip"),
    ("L_Knee", "R_Knee"),
    ("L_Ankle", "R_Ankle"),
]
symmetric_smpl_pairs = [(SMPL_BONE_ORDER_NAMES.index(left), SMPL_BONE_ORDER_NAMES.index(right)) for left, right in symmetric_pairs]

for iteration in range(num_iterations):
    verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, shape_new, trans[0:1])
    root_pos = joints[:, 0]
    joints = (joints - joints[:, 0]) * scale + root_pos
    diff = fk_return.global_translation[:, :, link_pick_idx] - joints[:, smpl_link_pick_idx]
    global_translations[iteration] = fk_return.global_translation[:, :, link_pick_idx].detach().numpy()
    smpl_global_translations[iteration] = joints[:, smpl_link_pick_idx].detach().numpy()
    loss_g = diff.norm(dim = -1).mean() 
    
    smpl_joints = joints.clone()
    symmetric_loss = 0.0
    for left_idx, right_idx in symmetric_smpl_pairs:
        left_joint = smpl_joints[:, left_idx] - fk_return.global_translation[:, :, 0]
        right_joint = smpl_joints[:, right_idx] - fk_return.global_translation[:, :, 0]
        # reflect right joint
        right_joint_mirrored = right_joint.clone()
        right_joint_mirrored[:, :, 1] *= -1  # symmetric along y-axis
        symmetric_diff = (left_joint - right_joint_mirrored).norm(dim=-1).mean()
        symmetric_loss += symmetric_diff
    symmetric_loss /= len(symmetric_smpl_pairs)
    
    loss = loss_g + symmetric_loss * 10
    if iteration % 1000 == 0:
        print(iteration, loss.item() * 1000)
        visualize_matching_single(
            fk_return.global_translation[:, :, link_visual_idx].squeeze().detach().numpy(),
            joints[:, smpl_link_visual_idx].squeeze().detach().numpy(),
            iteration
        )
        
    # 

    optimizer_shape.zero_grad()
    loss.backward()
    optimizer_shape.step()
    
optimized_shape_scale = {"shape": shape_new.detach(), "scale": scale.detach()}
joblib.dump(optimized_shape_scale, "../../data/g1/optimized_shape_scale_g1.pkl")