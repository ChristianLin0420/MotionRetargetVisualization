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

# Import joint angle plotter for "angles" mode
try:
    from joint_angle_plotter import record_angle_comparison_video
    ANGLES_AVAILABLE = True
except ImportError:
    ANGLES_AVAILABLE = False


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

def rotate_joints_around_z(joints: np.ndarray, angle_deg: float, center: np.ndarray = None) -> np.ndarray:
    """
    Rotate joints around the Z axis.
    
    Args:
        joints: Joint positions, shape (N, 3)
        angle_deg: Rotation angle in degrees
        center: Center of rotation (default: origin)
        
    Returns:
        Rotated joint positions
    """
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    
    if center is not None:
        joints_centered = joints - center
        rotated = joints_centered @ rotation_matrix.T
        return rotated + center
    else:
        return joints @ rotation_matrix.T


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
            
            # Get SMPL data - rotate to align with robot coordinate frame
            smpl_joints_t = smpl_joints[t]
            smpl_pelvis = smpl_joints_t[0]
            
            # Rotate SMPL joints by -90 degrees around Z to align with robot's facing
            # Then shift to robot's position for proper overlay
            smpl_joints_rotated = rotate_joints_around_z(smpl_joints_t, -90, center=smpl_pelvis)
            smpl_joints_aligned = smpl_joints_rotated - smpl_joints_rotated[0] + pos
            
            # Use same camera params as robot render for proper overlay
            cam_params = create_mujoco_camera_params(
                cam_distance, robot_azimuth, cam_elevation, pos, width, height
            )
            
            # Draw SMPL skeleton (aligned with robot)
            smpl_2d = project_3d_to_2d(smpl_joints_aligned, cam_params)
            frame_bgr = draw_skeleton(frame_bgr, smpl_2d, SMPL_PARENT_INDICES,
                                       color=(255, 150, 50), joint_radius=4, line_thickness=2)
            
            # Draw target keypoints with correspondence
            if target_keypoints is not None:
                target_kp_t = target_keypoints[t]
                # Rotate and align target keypoints the same way
                target_kp_rotated = rotate_joints_around_z(target_kp_t, -90, center=smpl_pelvis)
                target_kp_aligned = target_kp_rotated - target_kp_rotated[0] + pos
                target_kp_2d = project_3d_to_2d(target_kp_aligned, cam_params)
                
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
# Joint Correspondence Visualization
# =====================================================================

# Color palette for joint correspondence (13 joints)
CORRESPONDENCE_COLORS = [
    (0, 0, 255),      # Pelvis - Red
    (0, 128, 255),    # L_Hip - Orange
    (0, 255, 0),      # L_Knee - Green
    (255, 255, 0),    # L_Ankle - Cyan
    (0, 255, 255),    # R_Hip - Yellow
    (0, 255, 128),    # R_Knee - Light Green
    (255, 200, 0),    # R_Ankle - Light Cyan
    (255, 0, 0),      # L_Shoulder - Blue
    (255, 0, 128),    # L_Elbow - Purple-blue
    (255, 0, 255),    # L_Wrist - Magenta
    (128, 0, 255),    # R_Shoulder - Purple
    (200, 100, 255),  # R_Elbow - Light Purple
    (180, 150, 255),  # R_Wrist - Pink
]


def draw_correspondence_legend(frame: np.ndarray, x_start: int = 10, y_start: int = 50) -> np.ndarray:
    """Draw legend showing joint color mapping."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    for i, name in enumerate(TARGET_KEYPOINT_NAMES):
        color = CORRESPONDENCE_COLORS[i]
        y = y_start + i * 20
        cv2.circle(frame, (x_start + 10, y), 6, color, -1)
        cv2.putText(frame, name, (x_start + 25, y + 5), font, font_scale, (255, 255, 255), thickness)
    
    return frame


def record_correspondence_video(motion_data: dict, output_path: str,
                                 model_path: str, caption: str = "",
                                 fps: int = 30, width: int = 1280, height: int = 720) -> bool:
    """
    Record video showing color-coded joint correspondence mapping.
    Shows SMPL skeleton and robot with colored lines connecting corresponding joints.
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
    
    if root_orientation.shape[1] == 4:
        root_orientation = root_orientation[:, [3, 0, 1, 2]]
    
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
        return False
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        return False
    
    cam_distance = 3.5
    cam_elevation = -15
    robot_azimuth = 0      # Robot render azimuth (keep unchanged)
    skeleton_azimuth = 90  # Skeleton projection azimuth
    
    try:
        for t in tqdm(range(T), desc="Rendering frames", leave=False):
            pos = root_position[t]
            quat = root_orientation[t]
            
            data.qpos[0:3] = pos
            data.qpos[3:7] = quat
            data.qpos[7:] = joint_positions[t]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            
            # Render robot with robot_azimuth
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.distance = cam_distance
            camera.azimuth = robot_azimuth
            camera.elevation = cam_elevation
            camera.lookat[:] = pos
            
            renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Skeleton projection uses skeleton_azimuth with flip_x
            smpl_joints_t = smpl_joints[t]
            smpl_pelvis = smpl_joints_t[0]
            skeleton_cam_params = create_mujoco_camera_params(
                cam_distance, skeleton_azimuth, cam_elevation, smpl_pelvis, width, height
            )
            
            # Robot link projection uses robot_azimuth (no flip_x)
            robot_cam_params = create_mujoco_camera_params(
                cam_distance, robot_azimuth, cam_elevation, pos, width, height
            )
            
            # Draw SMPL skeleton in gray
            smpl_2d = project_3d_to_2d(smpl_joints_t, skeleton_cam_params, flip_x=True)
            frame_bgr = draw_skeleton(frame_bgr, smpl_2d, SMPL_PARENT_INDICES,
                                      color=(100, 100, 100), joint_radius=3, line_thickness=1)
            
            # Draw color-coded correspondence
            if target_keypoints is not None:
                target_kp_t = target_keypoints[t]
                target_kp_2d = project_3d_to_2d(target_kp_t, skeleton_cam_params, flip_x=True)
                robot_kp_3d = get_robot_link_positions(model, data, ROBOT_LINK_PICK)
                robot_kp_2d = project_3d_to_2d(robot_kp_3d, robot_cam_params)
                
                # Draw correspondence lines with unique colors
                for i in range(len(target_kp_2d)):
                    color = CORRESPONDENCE_COLORS[i]
                    pt1 = tuple(target_kp_2d[i])
                    pt2 = tuple(robot_kp_2d[i])
                    cv2.line(frame_bgr, pt1, pt2, color, 2, cv2.LINE_AA)
                    cv2.circle(frame_bgr, pt1, 8, color, -1)
                    cv2.circle(frame_bgr, pt2, 6, color, 2)
            
            # Add legend
            frame_bgr = draw_correspondence_legend(frame_bgr, width - 150, 50)
            frame_bgr = add_stage_label(frame_bgr, "Joint Correspondence", (10, 30), (255, 255, 255))
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


# =====================================================================
# IK Error Visualization
# =====================================================================

def get_error_color(error: float, max_error: float = 0.1) -> Tuple[int, int, int]:
    """Get color based on error magnitude (green -> yellow -> red)."""
    ratio = min(error / max_error, 1.0)
    
    if ratio < 0.5:
        t = ratio * 2
        return (0, 255, int(255 * t))  # Green to Yellow
    else:
        t = (ratio - 0.5) * 2
        return (0, int(255 * (1 - t)), 255)  # Yellow to Red


def record_error_video(motion_data: dict, output_path: str,
                        model_path: str, caption: str = "",
                        fps: int = 30, width: int = 1280, height: int = 720) -> bool:
    """
    Record video showing IK error with color gradient and statistics.
    """
    root_position = motion_data.get("root_trans_offset")
    root_orientation = motion_data.get("root_rot")
    joint_positions_all = motion_data.get("dof")
    target_keypoints = motion_data.get("target_keypoints")
    
    if any(x is None for x in [root_position, root_orientation, joint_positions_all, target_keypoints]):
        print("Missing required data")
        return False
    
    if root_orientation.shape[1] == 4:
        root_orientation = root_orientation[:, [3, 0, 1, 2]]
    
    if hasattr(target_keypoints, 'numpy'):
        target_keypoints = target_keypoints.numpy()
    
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
        return False
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        return False
    
    cam_distance = 3.5
    cam_elevation = -15
    robot_azimuth = 0      # Robot render azimuth (keep unchanged)
    skeleton_azimuth = 90  # Skeleton projection azimuth
    
    # Collect all errors for statistics
    all_errors = []
    
    try:
        for t in tqdm(range(T), desc="Rendering frames", leave=False):
            pos = root_position[t]
            quat = root_orientation[t]
            
            data.qpos[0:3] = pos
            data.qpos[3:7] = quat
            data.qpos[7:] = joint_positions[t]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.distance = cam_distance
            camera.azimuth = robot_azimuth
            camera.elevation = cam_elevation
            camera.lookat[:] = pos
            
            renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Calculate errors
            target_kp_t = target_keypoints[t]
            target_pelvis = target_kp_t[0]
            robot_kp_3d = get_robot_link_positions(model, data, ROBOT_LINK_PICK)
            errors = np.linalg.norm(target_kp_t - robot_kp_3d, axis=1)
            all_errors.append(errors)
            
            # Rotate and align target keypoints to match robot coordinate frame
            target_kp_rotated = rotate_joints_around_z(target_kp_t, -90, center=target_pelvis)
            target_kp_aligned = target_kp_rotated - target_kp_rotated[0] + pos
            
            # Use same camera params as robot render for proper overlay
            cam_params = create_mujoco_camera_params(
                cam_distance, robot_azimuth, cam_elevation, pos, width, height
            )
            
            target_kp_2d = project_3d_to_2d(target_kp_aligned, cam_params)
            robot_kp_2d = project_3d_to_2d(robot_kp_3d, cam_params)
            
            # Draw error visualization
            for i in range(len(target_kp_2d)):
                error = errors[i]
                color = get_error_color(error, max_error=0.1)
                pt_target = tuple(target_kp_2d[i])
                pt_robot = tuple(robot_kp_2d[i])
                
                # Draw error line
                cv2.line(frame_bgr, pt_target, pt_robot, color, 2, cv2.LINE_AA)
                
                # Draw target keypoint with error-colored circle
                radius = int(10 + error * 100)  # Larger radius for larger error
                cv2.circle(frame_bgr, pt_target, radius, color, 2)
                cv2.circle(frame_bgr, pt_target, 4, color, -1)
            
            # Draw error statistics
            mean_error = np.mean(errors) * 100  # cm
            max_error = np.max(errors) * 100
            stats_text = f"Mean: {mean_error:.1f}cm  Max: {max_error:.1f}cm"
            cv2.rectangle(frame_bgr, (10, height - 60), (300, height - 30), (0, 0, 0), -1)
            cv2.putText(frame_bgr, stats_text, (15, height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw error color bar
            bar_x, bar_y = width - 60, 50
            bar_height = 200
            for i in range(bar_height):
                ratio = i / bar_height
                color = get_error_color(ratio * 0.1, 0.1)
                cv2.line(frame_bgr, (bar_x, bar_y + bar_height - i),
                         (bar_x + 30, bar_y + bar_height - i), color, 1)
            cv2.putText(frame_bgr, "0cm", (bar_x - 5, bar_y + bar_height + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame_bgr, "10cm", (bar_x - 10, bar_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            frame_bgr = add_stage_label(frame_bgr, "IK Error Visualization", (10, 30), (255, 255, 255))
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


# =====================================================================
# Multi-View Visualization
# =====================================================================

def record_multiview_video(motion_data: dict, output_path: str,
                            model_path: str, caption: str = "",
                            fps: int = 30, panel_size: int = 480) -> bool:
    """
    Record video with 2x2 grid showing front, side, top, and 3/4 views.
    """
    root_position = motion_data.get("root_trans_offset")
    root_orientation = motion_data.get("root_rot")
    joint_positions_all = motion_data.get("dof")
    smpl_joints = motion_data.get("mocap_global_translation")
    
    if any(x is None for x in [root_position, root_orientation, joint_positions_all, smpl_joints]):
        print("Missing required data")
        return False
    
    if root_orientation.shape[1] == 4:
        root_orientation = root_orientation[:, [3, 0, 1, 2]]
    
    if hasattr(smpl_joints, 'numpy'):
        smpl_joints = smpl_joints.numpy()
    
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
    
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        model.vis.global_.offwidth = panel_size
        model.vis.global_.offheight = panel_size
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=panel_size, width=panel_size)
    except Exception as e:
        print(f"Error loading MuJoCo model: {e}")
        return False
    
    T = root_position.shape[0]
    floating_base_dof = 7
    n_joints = model.nq - floating_base_dof
    
    if joint_positions.shape[1] != n_joints:
        return False
    
    total_width = panel_size * 2
    total_height = panel_size * 2
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (total_width, total_height))
    
    if not video_writer.isOpened():
        return False
    
    cam_distance = 3.5
    
    # Camera configurations: (azimuth, elevation, label)
    cameras = [
        (0, -15, "Front View"),
        (90, -15, "Side View"),
        (0, 89, "Top View"),
        (45, -15, "3/4 View"),
    ]
    
    try:
        for t in tqdm(range(T), desc="Rendering frames", leave=False):
            pos = root_position[t]
            quat = root_orientation[t]
            
            data.qpos[0:3] = pos
            data.qpos[3:7] = quat
            data.qpos[7:] = joint_positions[t]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            
            panels = []
            for azimuth, elevation, label in cameras:
                camera = mujoco.MjvCamera()
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                camera.distance = cam_distance
                camera.azimuth = azimuth
                camera.elevation = elevation
                camera.lookat[:] = pos
                
                renderer.update_scene(data, camera=camera)
                panel = renderer.render()
                panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
                
                # Add skeleton overlay
                # For skeleton to align with robot: skeleton_azimuth = robot_azimuth + 90
                smpl_joints_t = smpl_joints[t]
                smpl_pelvis = smpl_joints_t[0]
                skeleton_azimuth = azimuth + 90  # Offset by 90 to match robot facing
                cam_params = create_mujoco_camera_params(
                    cam_distance, skeleton_azimuth, elevation, smpl_pelvis, panel_size, panel_size
                )
                smpl_2d = project_3d_to_2d(smpl_joints_t, cam_params, flip_x=True)
                panel = draw_skeleton(panel, smpl_2d, SMPL_PARENT_INDICES,
                                      color=(255, 150, 50), joint_radius=3, line_thickness=2, alpha=0.7)
                
                panel = add_stage_label(panel, label, (10, 25), (255, 255, 255))
                panels.append(panel)
            
            # Arrange in 2x2 grid
            top_row = np.hstack([panels[0], panels[1]])
            bottom_row = np.hstack([panels[2], panels[3]])
            combined = np.vstack([top_row, bottom_row])
            
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
# Trajectory Trails Visualization
# =====================================================================

# Trajectory colors for different joints
TRAJECTORY_COLORS_MAP = {
    0: (255, 255, 255),    # Pelvis - White
    22: (255, 100, 100),   # L_Hand - Light Blue
    23: (100, 100, 255),   # R_Hand - Light Red
    10: (255, 255, 100),   # L_Toe - Cyan
    11: (100, 255, 255),   # R_Toe - Yellow
}

TRAJECTORY_JOINTS = [0, 22, 23, 10, 11]  # Pelvis, L_Hand, R_Hand, L_Toe, R_Toe
TRAJECTORY_NAMES = ["Pelvis", "L_Hand", "R_Hand", "L_Foot", "R_Foot"]
TRAIL_LENGTH = 30


def record_trajectory_video(motion_data: dict, output_path: str,
                             model_path: str, caption: str = "",
                             fps: int = 30, width: int = 1280, height: int = 720) -> bool:
    """
    Record video showing motion trajectory trails for key joints.
    """
    root_position = motion_data.get("root_trans_offset")
    root_orientation = motion_data.get("root_rot")
    joint_positions_all = motion_data.get("dof")
    smpl_joints = motion_data.get("mocap_global_translation")
    
    if any(x is None for x in [root_position, root_orientation, joint_positions_all, smpl_joints]):
        print("Missing required data")
        return False
    
    if root_orientation.shape[1] == 4:
        root_orientation = root_orientation[:, [3, 0, 1, 2]]
    
    if hasattr(smpl_joints, 'numpy'):
        smpl_joints = smpl_joints.numpy()
    
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
        return False
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        return False
    
    cam_distance = 4.0  # Slightly further for trajectory view
    cam_elevation = -20
    robot_azimuth = 0      # Robot render azimuth (keep unchanged)
    skeleton_azimuth = 90  # Skeleton projection azimuth
    
    # Store trajectory history
    trajectory_history = {joint_idx: [] for joint_idx in TRAJECTORY_JOINTS}
    
    try:
        for t in tqdm(range(T), desc="Rendering frames", leave=False):
            pos = root_position[t]
            quat = root_orientation[t]
            
            data.qpos[0:3] = pos
            data.qpos[3:7] = quat
            data.qpos[7:] = joint_positions[t]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.distance = cam_distance
            camera.azimuth = robot_azimuth
            camera.elevation = cam_elevation
            camera.lookat[:] = pos
            
            renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Get SMPL data and rotate to align with robot coordinate frame
            smpl_joints_t = smpl_joints[t]
            smpl_pelvis = smpl_joints_t[0]
            
            # Rotate and align SMPL joints to match robot's coordinate frame
            smpl_joints_rotated = rotate_joints_around_z(smpl_joints_t, -90, center=smpl_pelvis)
            smpl_joints_aligned = smpl_joints_rotated - smpl_joints_rotated[0] + pos
            
            # Update trajectory history with aligned positions
            for joint_idx in TRAJECTORY_JOINTS:
                trajectory_history[joint_idx].append(smpl_joints_aligned[joint_idx].copy())
                if len(trajectory_history[joint_idx]) > TRAIL_LENGTH:
                    trajectory_history[joint_idx].pop(0)
            
            # Use same camera params as robot render for proper overlay
            cam_params = create_mujoco_camera_params(
                cam_distance, robot_azimuth, cam_elevation, pos, width, height
            )
            
            # Draw trajectory trails
            for joint_idx in TRAJECTORY_JOINTS:
                color = TRAJECTORY_COLORS_MAP[joint_idx]
                history = trajectory_history[joint_idx]
                
                if len(history) > 1:
                    # Project all points
                    points_3d = np.array(history)
                    points_2d = project_3d_to_2d(points_3d, cam_params)
                    
                    # Draw trail with fading alpha
                    for i in range(len(points_2d) - 1):
                        alpha = (i + 1) / len(points_2d)
                        faded_color = tuple(int(c * alpha) for c in color)
                        pt1 = tuple(points_2d[i])
                        pt2 = tuple(points_2d[i + 1])
                        cv2.line(frame_bgr, pt1, pt2, faded_color, 2, cv2.LINE_AA)
                    
                    # Draw current position
                    cv2.circle(frame_bgr, tuple(points_2d[-1]), 6, color, -1)
            
            # Draw skeleton
            smpl_2d = project_3d_to_2d(smpl_joints_aligned, cam_params)
            frame_bgr = draw_skeleton(frame_bgr, smpl_2d, SMPL_PARENT_INDICES,
                                      color=(255, 150, 50), joint_radius=3, line_thickness=1, alpha=0.5)
            
            # Add legend
            legend_y = 50
            for i, (joint_idx, name) in enumerate(zip(TRAJECTORY_JOINTS, TRAJECTORY_NAMES)):
                color = TRAJECTORY_COLORS_MAP[joint_idx]
                cv2.circle(frame_bgr, (width - 100, legend_y + i * 25), 6, color, -1)
                cv2.putText(frame_bgr, name, (width - 85, legend_y + i * 25 + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            frame_bgr = add_stage_label(frame_bgr, "Trajectory Trails", (10, 30), (255, 255, 255))
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


# =====================================================================
# Main Entry Point
# =====================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-stage motion retargeting visualization")
    parser.add_argument("--mode", type=str, default="sidebyside",
                        choices=["overlay", "sidebyside", "correspondence", "error", "multiview", "trajectory", "angles", "all"],
                        help="Visualization mode (use 'all' to generate all modes)")
    parser.add_argument("--max_motions", type=int, default=5,
                        help="Maximum number of motions to record")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for videos")
    args = parser.parse_args()
    
    # Path setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    motion_data_path = os.path.join(current_dir, "../data/g1/humanml3d_train_retargeted_wholebody_82.pkl")
    model_path = os.path.join(current_dir, '../resources/robots/g1/g1_27dof.xml')
    
    # Define all available modes (excluding "all")
    ALL_MODES = ["overlay", "sidebyside", "correspondence", "error", "multiview", "trajectory"]
    if ANGLES_AVAILABLE:
        ALL_MODES.append("angles")
    
    # Determine which modes to run
    if args.mode == "all":
        modes_to_run = ALL_MODES
        base_output_dir = args.output_dir if args.output_dir else os.path.join(current_dir, "../output/videos_all")
    else:
        modes_to_run = [args.mode]
        base_output_dir = args.output_dir if args.output_dir else os.path.join(current_dir, f"../output/videos_{args.mode}")
    
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Load motion dataset
    print("Loading motion dataset...")
    motion_dataset = joblib.load(motion_data_path)
    motion_dataset.pop("config", None)
    
    # Filter motions (skip mirrored ones)
    motion_names = [name for name in motion_dataset.keys() if not name.startswith("M")]
    
    if args.max_motions:
        motion_names = motion_names[:args.max_motions]
    
    print(f"Recording {len(motion_names)} motions in {', '.join(modes_to_run)} mode(s)...")
    
    def record_single_mode(motion_data, output_path, mode, caption):
        """Record a single visualization mode."""
        if mode == "overlay":
            return record_comparison_video_overlay(
                motion_data, output_path, model_path, caption=caption,
                fps=30, width=1280, height=720
            )
        elif mode == "sidebyside":
            return record_comparison_video_sidebyside(
                motion_data, output_path, model_path, caption=caption,
                fps=30, panel_width=640, panel_height=720
            )
        elif mode == "correspondence":
            return record_correspondence_video(
                motion_data, output_path, model_path, caption=caption,
                fps=30, width=1280, height=720
            )
        elif mode == "error":
            return record_error_video(
                motion_data, output_path, model_path, caption=caption,
                fps=30, width=1280, height=720
            )
        elif mode == "multiview":
            return record_multiview_video(
                motion_data, output_path, model_path, caption=caption,
                fps=30, panel_size=480
            )
        elif mode == "trajectory":
            return record_trajectory_video(
                motion_data, output_path, model_path, caption=caption,
                fps=30, width=1280, height=720
            )
        elif mode == "angles":
            if ANGLES_AVAILABLE:
                return record_angle_comparison_video(
                    motion_data, output_path, model_path, caption=caption,
                    fps=30, width=1280
                )
            else:
                print("  Warning: angles mode not available (joint_angle_plotter.py not found)")
                return False
        return False
    
    total_videos = len(motion_names) * len(modes_to_run)
    print(f"Total videos to generate: {total_videos}")
    
    for motion_name in tqdm(motion_names, desc="Recording motions"):
        motion_data = motion_dataset[motion_name]
        captions = motion_data.get("captions", [])
        caption = captions[0] if captions else ""
        motion_base_name = motion_name.replace('.npz', '')
        
        for mode in modes_to_run:
            # Create mode-specific subdirectory for "all" mode
            if args.mode == "all":
                mode_output_dir = os.path.join(base_output_dir, mode)
                os.makedirs(mode_output_dir, exist_ok=True)
                output_path = os.path.join(mode_output_dir, f"{motion_base_name}_{mode}.mp4")
            else:
                output_path = os.path.join(base_output_dir, f"{motion_base_name}_{mode}.mp4")
            
            success = record_single_mode(motion_data, output_path, mode, caption)
            
            if success:
                tqdm.write(f"  Saved: {output_path}")
            else:
                tqdm.write(f"  Failed: {motion_name} ({mode})")
    
    print(f"\nDone! Videos saved to: {base_output_dir}")
    if args.mode == "all":
        print(f"Subdirectories: {', '.join(ALL_MODES)}")

