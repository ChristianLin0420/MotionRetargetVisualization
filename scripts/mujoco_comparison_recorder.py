"""
Multi-Stage Motion Retargeting Visualization

This script visualizes all three stages of the motion retargeting process:
1. Original SMPL human skeleton (24 joints)
2. Target keypoints for IK matching (13 joints)
3. Final G1 robot motion (MuJoCo)

Supports two visualization modes:
- Side-by-side: Three separate panels
- Overlay: All stages in a single view with different colors
"""

import os
# Set environment variable BEFORE importing mujoco
os.environ['MUJOCO_GL'] = 'egl'

import joblib
import numpy as np
import mujoco
import cv2
from tqdm import tqdm
from typing import List, Tuple, Optional


# =====================================================================
# SMPL Skeleton Definition
# =====================================================================

SMPL_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee", "Spine",
    "L_Ankle", "R_Ankle", "Chest", "L_Toe", "R_Toe", "Neck", "L_Thorax",
    "R_Thorax", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"
]

# Parent joint indices for drawing skeleton lines
SMPL_PARENT_INDICES = [
    -1,  # Pelvis (root)
    0,   # L_Hip -> Pelvis
    0,   # R_Hip -> Pelvis
    0,   # Torso -> Pelvis
    1,   # L_Knee -> L_Hip
    2,   # R_Knee -> R_Hip
    3,   # Spine -> Torso
    4,   # L_Ankle -> L_Knee
    5,   # R_Ankle -> R_Knee
    6,   # Chest -> Spine
    7,   # L_Toe -> L_Ankle
    8,   # R_Toe -> R_Ankle
    9,   # Neck -> Chest
    9,   # L_Thorax -> Chest
    9,   # R_Thorax -> Chest
    12,  # Head -> Neck
    13,  # L_Shoulder -> L_Thorax
    14,  # R_Shoulder -> R_Thorax
    16,  # L_Elbow -> L_Shoulder
    17,  # R_Elbow -> R_Shoulder
    18,  # L_Wrist -> L_Elbow
    19,  # R_Wrist -> R_Elbow
    20,  # L_Hand -> L_Wrist
    21,  # R_Hand -> R_Wrist
]

# Target keypoints used for IK (13 joints)
TARGET_KEYPOINT_NAMES = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle",
    "R_Hip", "R_Knee", "R_Ankle",
    "L_Shoulder", "L_Elbow", "L_Wrist",
    "R_Shoulder", "R_Elbow", "R_Wrist"
]

# Parent indices for target keypoints skeleton
TARGET_KEYPOINT_PARENTS = [
    -1,  # Pelvis
    0,   # L_Hip -> Pelvis
    1,   # L_Knee -> L_Hip
    2,   # L_Ankle -> L_Knee
    0,   # R_Hip -> Pelvis
    4,   # R_Knee -> R_Hip
    5,   # R_Ankle -> R_Knee
    0,   # L_Shoulder -> Pelvis (simplified)
    7,   # L_Elbow -> L_Shoulder
    8,   # L_Wrist -> L_Elbow
    0,   # R_Shoulder -> Pelvis (simplified)
    10,  # R_Elbow -> R_Shoulder
    11,  # R_Wrist -> R_Elbow
]

# Robot link names corresponding to target keypoints
ROBOT_LINK_PICK = [
    'pelvis',
    'left_hip_pitch_link', 'left_knee_link', 'left_ankle_roll_link',
    'right_hip_pitch_link', 'right_knee_link', 'right_ankle_roll_link',
    'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_yaw_link',
    'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_yaw_link',
]


# =====================================================================
# 3D to 2D Projection Utilities (MuJoCo-compatible)
# =====================================================================

def create_mujoco_camera_params(distance: float, azimuth: float, elevation: float,
                                 lookat: np.ndarray, width: int, height: int,
                                 fovy: float = 45.0) -> dict:
    """
    Create camera parameters matching MuJoCo's camera setup.
    
    Args:
        distance: Camera distance from lookat point
        azimuth: Camera azimuth angle in degrees
        elevation: Camera elevation angle in degrees
        lookat: 3D point the camera looks at
        width: Image width
        height: Image height
        fovy: Vertical field of view in degrees (MuJoCo default is 45)
        
    Returns:
        Dictionary with camera parameters for projection
    """
    # Convert angles to radians
    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)
    
    # Camera position (MuJoCo convention)
    # azimuth: rotation around vertical (Z) axis
    # elevation: rotation from horizontal plane
    cam_x = lookat[0] + distance * np.cos(el_rad) * np.sin(az_rad)
    cam_y = lookat[1] - distance * np.cos(el_rad) * np.cos(az_rad)
    cam_z = lookat[2] + distance * np.sin(el_rad)
    cam_pos = np.array([cam_x, cam_y, cam_z])
    
    # Camera basis vectors
    forward = lookat - cam_pos
    forward = forward / np.linalg.norm(forward)
    
    # World up vector
    world_up = np.array([0, 0, 1])
    
    # Camera right vector
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0, 1, 0])
        right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    
    # Camera up vector
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    # View matrix (world to camera)
    R = np.array([right, up, -forward])  # Camera rotation matrix
    t = -R @ cam_pos  # Camera translation
    
    view_matrix = np.eye(4)
    view_matrix[:3, :3] = R
    view_matrix[:3, 3] = t
    
    # Focal length from FOV
    fovy_rad = np.radians(fovy)
    focal_length = (height / 2.0) / np.tan(fovy_rad / 2.0)
    
    return {
        'view_matrix': view_matrix,
        'cam_pos': cam_pos,
        'focal_length': focal_length,
        'width': width,
        'height': height,
        'cx': width / 2.0,
        'cy': height / 2.0
    }


def project_3d_to_2d(points_3d: np.ndarray, cam_params: dict, flip_x: bool = False) -> np.ndarray:
    """
    Project 3D points to 2D screen coordinates using perspective projection.
    
    Args:
        points_3d: Nx3 array of 3D world coordinates
        cam_params: Camera parameters from create_mujoco_camera_params
        flip_x: If True, flip X axis to correct mirror effect
        
    Returns:
        Nx2 array of 2D pixel coordinates
    """
    view_matrix = cam_params['view_matrix']
    focal_length = cam_params['focal_length']
    cx, cy = cam_params['cx'], cam_params['cy']
    height = cam_params['height']
    width = cam_params['width']
    
    # Transform to camera coordinates
    ones = np.ones((points_3d.shape[0], 1))
    points_homo = np.hstack([points_3d, ones])
    points_cam = (view_matrix @ points_homo.T).T[:, :3]
    
    # Perspective projection
    # Camera looks along -Z in camera space
    z = points_cam[:, 2]
    z = np.where(np.abs(z) < 0.01, -0.01, z)  # Avoid division by zero
    
    # Project to image plane
    x_2d = -focal_length * points_cam[:, 0] / z + cx
    y_2d = -focal_length * points_cam[:, 1] / z + cy
    
    # Flip Y to match image coordinates (origin at top-left)
    y_2d = height - y_2d
    
    # Flip X to correct mirror effect if needed
    if flip_x:
        x_2d = width - x_2d
    
    return np.stack([x_2d, y_2d], axis=1).astype(np.int32)


# =====================================================================
# Drawing Functions
# =====================================================================

def draw_skeleton(frame: np.ndarray, joints_2d: np.ndarray, 
                  parent_indices: List[int], color: Tuple[int, int, int],
                  joint_radius: int = 5, line_thickness: int = 2,
                  alpha: float = 1.0) -> np.ndarray:
    """
    Draw a skeleton on the frame.
    
    Args:
        frame: BGR image
        joints_2d: Nx2 array of 2D joint positions
        parent_indices: List of parent joint indices
        color: BGR color tuple
        joint_radius: Radius for joint circles
        line_thickness: Thickness for bone lines
        alpha: Transparency (1.0 = opaque)
    """
    overlay = frame.copy() if alpha < 1.0 else frame
    
    # Draw bones
    for i, parent_idx in enumerate(parent_indices):
        if parent_idx >= 0 and parent_idx < len(joints_2d):
            pt1 = tuple(joints_2d[i])
            pt2 = tuple(joints_2d[parent_idx])
            cv2.line(overlay, pt1, pt2, color, line_thickness)
    
    # Draw joints
    for i, pt in enumerate(joints_2d):
        cv2.circle(overlay, tuple(pt), joint_radius, color, -1)
    
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    return frame


def draw_keypoints_with_correspondence(frame: np.ndarray, 
                                        smpl_kp_2d: np.ndarray,
                                        robot_kp_2d: np.ndarray,
                                        kp_color: Tuple[int, int, int] = (0, 255, 0),
                                        line_color: Tuple[int, int, int] = (0, 255, 255),
                                        radius: int = 8,
                                        line_thickness: int = 1) -> np.ndarray:
    """
    Draw target keypoints with correspondence lines to robot.
    """
    # Draw correspondence lines
    for i in range(len(smpl_kp_2d)):
        pt1 = tuple(smpl_kp_2d[i])
        pt2 = tuple(robot_kp_2d[i])
        cv2.line(frame, pt1, pt2, line_color, line_thickness, cv2.LINE_AA)
    
    # Draw keypoints
    for pt in smpl_kp_2d:
        cv2.circle(frame, tuple(pt), radius, kp_color, -1)
        cv2.circle(frame, tuple(pt), radius, (255, 255, 255), 2)
    
    return frame


def add_caption(frame: np.ndarray, caption: str, position: str = "top",
                font_scale: float = 0.7, color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Add text caption to frame."""
    if not caption:
        return frame
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    bg_color = (0, 0, 0)
    
    height, width = frame.shape[:2]
    max_width = width - 40
    
    # Word wrap
    words = caption.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + " " + word if current_line else word
        (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if text_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    
    # Draw text
    y_offset = 30 if position == "top" else height - 30 * len(lines)
    for line in lines:
        (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        cv2.rectangle(frame, (10, y_offset - text_height - 5),
                      (20 + text_width, y_offset + baseline + 5), bg_color, -1)
        cv2.putText(frame, line, (15, y_offset), font, font_scale, color, thickness)
        y_offset += text_height + 15
    
    return frame


def add_stage_label(frame: np.ndarray, label: str, position: Tuple[int, int],
                    color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Add stage label (e.g., 'SMPL', 'Target', 'Robot')."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(frame, (x - 5, y - text_height - 5),
                  (x + text_width + 5, y + 5), (0, 0, 0), -1)
    cv2.putText(frame, label, (x, y), font, font_scale, color, thickness)
    
    return frame


# =====================================================================
# Robot Link Position Extraction
# =====================================================================

def get_robot_link_positions(model, data, link_names: List[str]) -> np.ndarray:
    """
    Get 3D positions of specified robot links from MuJoCo data.
    """
    positions = []
    for link_name in link_names:
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name)
            pos = data.xpos[body_id].copy()
            positions.append(pos)
        except:
            positions.append(np.zeros(3))
    return np.array(positions)


# =====================================================================
# Main Recording Functions
# =====================================================================

def record_comparison_video_overlay(motion_data: dict, output_path: str,
                                     model_path: str, caption: str = "",
                                     fps: int = 30, width: int = 1280, height: int = 720) -> bool:
    """
    Record comparison video with all stages overlaid in a single view.
    
    Colors:
    - SMPL skeleton: Blue
    - Target keypoints: Green with yellow correspondence lines
    - Robot: Rendered by MuJoCo
    """
    # Extract data
    root_position = motion_data.get("root_trans_offset")
    root_orientation = motion_data.get("root_rot")
    joint_positions_all = motion_data.get("dof")
    smpl_joints = motion_data.get("mocap_global_translation")
    target_keypoints = motion_data.get("target_keypoints")
    
    if any(x is None for x in [root_position, root_orientation, joint_positions_all, smpl_joints]):
        print("Missing required data")
        return False
    
    # Convert quaternions from XYZW to WXYZ for MuJoCo
    if root_orientation.shape[1] == 4:
        root_orientation = root_orientation[:, [3, 0, 1, 2]]
    
    # Convert tensors to numpy if needed
    if hasattr(smpl_joints, 'numpy'):
        smpl_joints = smpl_joints.numpy()
    if target_keypoints is not None and hasattr(target_keypoints, 'numpy'):
        target_keypoints = target_keypoints.numpy()
    
    # Robot joint names and active indices
    robot_joint_names = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]
    
    active_robot_joint_names = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]
    
    active_robot_joint_indices = [robot_joint_names.index(j) for j in active_robot_joint_names]
    joint_positions = joint_positions_all[:, active_robot_joint_indices]
    
    # Load MuJoCo model
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        model.vis.global_.offwidth = width
        model.vis.global_.offheight = height
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
    except Exception as e:
        print(f"Error loading MuJoCo model: {e}")
        return False
    
    T = root_position.shape[0]
    floating_base_dof = 7
    n_joints = model.nq - floating_base_dof
    
    if joint_positions.shape[1] != n_joints:
        print(f"Joint positions dimension mismatch: expected {n_joints}, got {joint_positions.shape[1]}")
        return False
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        return False
    
    # Camera parameters - skeleton and robot have different azimuth conventions
    cam_distance = 3.5
    cam_elevation = -15
    skeleton_azimuth = 90   # For skeleton projection
    robot_azimuth = 0       # Rotated 180 degrees to match skeleton facing
    
    try:
        for t in tqdm(range(T), desc="Rendering frames", leave=False):
            pos = root_position[t]
            quat = root_orientation[t]
            
            # Update MuJoCo state
            data.qpos[0:3] = pos
            data.qpos[3:7] = quat
            data.qpos[7:] = joint_positions[t]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            
            # Setup camera for robot rendering
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.distance = cam_distance
            camera.azimuth = robot_azimuth
            camera.elevation = cam_elevation
            camera.lookat[:] = pos
            
            # Render robot
            renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Get SMPL data
            smpl_joints_t = smpl_joints[t]
            
            # Create projection for skeleton overlay (use robot_azimuth to match rendered robot)
            cam_params = create_mujoco_camera_params(
                cam_distance, robot_azimuth, cam_elevation, pos, width, height
            )
            
            # Draw SMPL skeleton
            smpl_2d = project_3d_to_2d(smpl_joints_t, cam_params)
            frame_bgr = draw_skeleton(frame_bgr, smpl_2d, SMPL_PARENT_INDICES,
                                       color=(255, 150, 50), joint_radius=4, line_thickness=2)
            
            # Draw target keypoints with correspondence
            if target_keypoints is not None:
                target_kp_t = target_keypoints[t]
                target_kp_2d = project_3d_to_2d(target_kp_t, cam_params)
                
                # Get robot link positions for correspondence
                robot_kp_3d = get_robot_link_positions(model, data, ROBOT_LINK_PICK)
                robot_kp_2d = project_3d_to_2d(robot_kp_3d, cam_params)
                
                frame_bgr = draw_keypoints_with_correspondence(
                    frame_bgr, target_kp_2d, robot_kp_2d,
                    kp_color=(0, 255, 0), line_color=(0, 200, 255),
                    radius=6, line_thickness=1
                )
            
            # Add labels
            frame_bgr = add_stage_label(frame_bgr, "SMPL Skeleton", (10, height - 80), (255, 150, 50))
            frame_bgr = add_stage_label(frame_bgr, "Target Keypoints", (10, height - 50), (0, 255, 0))
            frame_bgr = add_stage_label(frame_bgr, "G1 Robot", (10, height - 20), (255, 255, 255))
            
            # Add caption
            frame_bgr = add_caption(frame_bgr, caption)
            
            video_writer.write(frame_bgr)
    
    except Exception as e:
        print(f"Error during rendering: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        video_writer.release()
        renderer.close()
    
    return True


def record_comparison_video_sidebyside(motion_data: dict, output_path: str,
                                        model_path: str, caption: str = "",
                                        fps: int = 30, panel_width: int = 640, 
                                        panel_height: int = 720) -> bool:
    """
    Record comparison video with three side-by-side panels:
    1. SMPL skeleton
    2. Target keypoints
    3. Robot
    """
    # Extract data
    root_position = motion_data.get("root_trans_offset")
    root_orientation = motion_data.get("root_rot")
    joint_positions_all = motion_data.get("dof")
    smpl_joints = motion_data.get("mocap_global_translation")
    target_keypoints = motion_data.get("target_keypoints")
    
    if any(x is None for x in [root_position, root_orientation, joint_positions_all, smpl_joints]):
        print("Missing required data")
        return False
    
    # Convert quaternions from XYZW to WXYZ for MuJoCo
    if root_orientation.shape[1] == 4:
        root_orientation = root_orientation[:, [3, 0, 1, 2]]
    
    # Convert tensors to numpy
    if hasattr(smpl_joints, 'numpy'):
        smpl_joints = smpl_joints.numpy()
    if target_keypoints is not None and hasattr(target_keypoints, 'numpy'):
        target_keypoints = target_keypoints.numpy()
    
    # Robot joint configuration
    robot_joint_names = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]
    
    active_robot_joint_names = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]
    
    active_robot_joint_indices = [robot_joint_names.index(j) for j in active_robot_joint_names]
    joint_positions = joint_positions_all[:, active_robot_joint_indices]
    
    # Load MuJoCo model
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        model.vis.global_.offwidth = panel_width
        model.vis.global_.offheight = panel_height
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=panel_height, width=panel_width)
    except Exception as e:
        print(f"Error loading MuJoCo model: {e}")
        return False
    
    T = root_position.shape[0]
    floating_base_dof = 7
    n_joints = model.nq - floating_base_dof
    
    if joint_positions.shape[1] != n_joints:
        print(f"Joint mismatch: expected {n_joints}, got {joint_positions.shape[1]}")
        return False
    
    # Total frame size (3 panels)
    total_width = panel_width * 3
    total_height = panel_height
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (total_width, total_height))
    
    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        return False
    
    # Camera parameters
    cam_distance = 3.5
    cam_elevation = -15
    # Skeleton projection and MuJoCo camera have different azimuth conventions
    skeleton_azimuth = 90
    robot_azimuth = 0  # Rotated 180 degrees from previous (180 -> 0)
    
    try:
        for t in tqdm(range(T), desc="Rendering frames", leave=False):
            pos = root_position[t]
            quat = root_orientation[t]
            
            # Update MuJoCo state
            data.qpos[0:3] = pos
            data.qpos[3:7] = quat
            data.qpos[7:] = joint_positions[t]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            
            # Get SMPL data for this frame
            smpl_joints_t = smpl_joints[t]
            smpl_pelvis = smpl_joints_t[0]  # Use SMPL pelvis as center
            
            # Create projection for skeleton panels
            skeleton_cam_params = create_mujoco_camera_params(
                cam_distance, skeleton_azimuth, cam_elevation, smpl_pelvis, panel_width, panel_height
            )
            
            # Panel 1: SMPL Skeleton (flip_x to correct mirror)
            panel1 = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
            panel1[:] = (40, 40, 40)  # Dark gray background
            
            smpl_2d = project_3d_to_2d(smpl_joints_t, skeleton_cam_params, flip_x=True)
            panel1 = draw_skeleton(panel1, smpl_2d, SMPL_PARENT_INDICES,
                                   color=(255, 150, 50), joint_radius=6, line_thickness=3)
            panel1 = add_stage_label(panel1, "SMPL Skeleton (24 joints)", (10, 30), (255, 150, 50))
            
            # Panel 2: Target Keypoints (flip_x to correct mirror)
            panel2 = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
            panel2[:] = (40, 40, 40)
            
            if target_keypoints is not None:
                target_kp_t = target_keypoints[t]
                target_kp_2d = project_3d_to_2d(target_kp_t, skeleton_cam_params, flip_x=True)
                panel2 = draw_skeleton(panel2, target_kp_2d, TARGET_KEYPOINT_PARENTS,
                                       color=(0, 255, 0), joint_radius=8, line_thickness=3)
            panel2 = add_stage_label(panel2, "Target Keypoints (13 joints)", (10, 30), (0, 255, 0))
            
            # Panel 3: Robot (use FREE camera with adjusted azimuth)
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.distance = cam_distance
            camera.azimuth = robot_azimuth
            camera.elevation = cam_elevation
            camera.lookat[:] = pos
            
            renderer.update_scene(data, camera=camera)
            panel3 = renderer.render()
            panel3 = cv2.cvtColor(panel3, cv2.COLOR_RGB2BGR)
            panel3 = add_stage_label(panel3, "G1 Robot (29 DOFs)", (10, 30), (255, 255, 255))
            
            # Combine panels
            combined = np.hstack([panel1, panel2, panel3])
            
            # Add caption at bottom
            combined = add_caption(combined, caption, position="bottom")
            
            video_writer.write(combined)
    
    except Exception as e:
        print(f"Error during rendering: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        video_writer.release()
        renderer.close()
    
    return True


# =====================================================================
# Main Entry Point
# =====================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-stage motion retargeting visualization")
    parser.add_argument("--mode", type=str, default="overlay", choices=["overlay", "sidebyside"],
                        help="Visualization mode: overlay or sidebyside")
    parser.add_argument("--max_motions", type=int, default=5,
                        help="Maximum number of motions to record (None for all)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for videos")
    args = parser.parse_args()
    
    # Path setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    motion_data_path = os.path.join(current_dir, "../data/g1/humanml3d_train_retargeted_wholebody_82.pkl")
    model_path = os.path.join(current_dir, '../resources/robots/g1/g1_27dof.xml')
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(current_dir, f"../output/videos_comparison_{args.mode}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load motion dataset
    print("Loading motion dataset...")
    motion_dataset = joblib.load(motion_data_path)
    motion_dataset.pop("config", None)
    
    # Filter motions (skip mirrored ones)
    motion_names = [name for name in motion_dataset.keys() if not name.startswith("M")]
    
    if args.max_motions:
        motion_names = motion_names[:args.max_motions]
    
    print(f"Recording {len(motion_names)} motions in {args.mode} mode...")
    
    for motion_name in tqdm(motion_names, desc="Recording videos"):
        motion_data = motion_dataset[motion_name]
        captions = motion_data.get("captions", [])
        caption = captions[0] if captions else ""
        
        output_path = os.path.join(output_dir, f"{motion_name.replace('.npz', '')}_{args.mode}.mp4")
        
        if args.mode == "overlay":
            success = record_comparison_video_overlay(
                motion_data, output_path, model_path, caption=caption,
                fps=30, width=1280, height=720
            )
        else:
            success = record_comparison_video_sidebyside(
                motion_data, output_path, model_path, caption=caption,
                fps=30, panel_width=640, panel_height=720
            )
        
        if success:
            print(f"  Saved: {output_path}")
        else:
            print(f"  Failed: {motion_name}")
    
    print(f"\nDone! Videos saved to: {output_dir}")

