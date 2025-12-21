"""
Shared configuration for motion retargeting visualization.

This module contains all shared constants, colors, and camera parameters
to ensure consistent direction alignment across all visualization modes.
"""

import numpy as np

# =====================================================================
# Camera Configuration (Critical for Direction Alignment)
# =====================================================================

# These values ensure all visualizations face the same direction
SKELETON_AZIMUTH = 90    # For SMPL skeleton, mesh, and keypoints projection
ROBOT_AZIMUTH = 0        # For MuJoCo robot rendering
CAM_DISTANCE = 3.5       # Default camera distance
CAM_ELEVATION = -15      # Default camera elevation
CAM_FOVY = 45.0          # Field of view (vertical)

# Projection correction
FLIP_X = True            # Flip X axis to correct mirror effect for skeleton/keypoints


# =====================================================================
# Color Palette (BGR format for OpenCV)
# =====================================================================

# Main stage colors
COLOR_SMPL_SKELETON = (255, 150, 50)    # Orange-ish blue
COLOR_TARGET_KEYPOINTS = (0, 255, 0)     # Green
COLOR_ROBOT = (255, 255, 255)            # White
COLOR_SMPL_MESH = (200, 180, 160)        # Skin-like color

# Joint correspondence colors (13 joints)
CORRESPONDENCE_COLORS = {
    'Pelvis': (0, 0, 255),           # Red
    'L_Hip': (0, 128, 255),          # Orange
    'L_Knee': (0, 255, 0),           # Green
    'L_Ankle': (255, 255, 0),        # Cyan
    'R_Hip': (0, 255, 255),          # Yellow
    'R_Knee': (0, 255, 128),         # Light Green
    'R_Ankle': (255, 200, 0),        # Light Cyan
    'L_Shoulder': (255, 0, 0),       # Blue
    'L_Elbow': (255, 0, 128),        # Purple-blue
    'L_Wrist': (255, 0, 255),        # Magenta
    'R_Shoulder': (128, 0, 255),     # Purple
    'R_Elbow': (200, 100, 255),      # Light Purple
    'R_Wrist': (180, 150, 255),      # Pink
}

# Correspondence color list (ordered by TARGET_KEYPOINT_NAMES)
CORRESPONDENCE_COLOR_LIST = [
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

# Trajectory trail colors
TRAJECTORY_COLORS = {
    'pelvis': (255, 255, 255),       # White
    'left_hand': (255, 100, 100),    # Light Blue
    'right_hand': (100, 100, 255),   # Light Red
    'left_foot': (255, 255, 100),    # Light Cyan
    'right_foot': (100, 255, 255),   # Light Yellow
}

# Error visualization gradient (low to high)
ERROR_COLOR_LOW = (0, 255, 0)         # Green
ERROR_COLOR_MID = (0, 255, 255)       # Yellow
ERROR_COLOR_HIGH = (0, 0, 255)        # Red

# Background colors
BG_DARK_GRAY = (40, 40, 40)
BG_BLACK = (0, 0, 0)


# =====================================================================
# SMPL Skeleton Definition
# =====================================================================

SMPL_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee", "Spine",
    "L_Ankle", "R_Ankle", "Chest", "L_Toe", "R_Toe", "Neck", "L_Thorax",
    "R_Thorax", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"
]

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


# =====================================================================
# Target Keypoints Definition (13 joints for IK)
# =====================================================================

TARGET_KEYPOINT_NAMES = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle",
    "R_Hip", "R_Knee", "R_Ankle",
    "L_Shoulder", "L_Elbow", "L_Wrist",
    "R_Shoulder", "R_Elbow", "R_Wrist"
]

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

# SMPL indices for target keypoints extraction
SMPL_TO_TARGET_INDICES = [0, 1, 4, 7, 2, 5, 8, 16, 18, 20, 17, 19, 21]


# =====================================================================
# Robot Configuration
# =====================================================================

ROBOT_LINK_PICK = [
    'pelvis',
    'left_hip_pitch_link', 'left_knee_link', 'left_ankle_roll_link',
    'right_hip_pitch_link', 'right_knee_link', 'right_ankle_roll_link',
    'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_yaw_link',
    'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_yaw_link',
]

ROBOT_JOINT_NAMES = [
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

ACTIVE_ROBOT_JOINT_NAMES = [
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

# Joint indices for angle comparison
COMPARISON_JOINTS = {
    'left_hip': ('left_hip_pitch_joint', 0),    # SMPL L_Hip
    'left_knee': ('left_knee_joint', 4),         # SMPL L_Knee
    'right_hip': ('right_hip_pitch_joint', 2),   # SMPL R_Hip
    'right_knee': ('right_knee_joint', 5),       # SMPL R_Knee
    'left_shoulder': ('left_shoulder_pitch_joint', 16),  # SMPL L_Shoulder
    'left_elbow': ('left_elbow_joint', 18),      # SMPL L_Elbow
    'right_shoulder': ('right_shoulder_pitch_joint', 17),  # SMPL R_Shoulder
    'right_elbow': ('right_elbow_joint', 19),    # SMPL R_Elbow
}


# =====================================================================
# Trajectory Configuration
# =====================================================================

# Joint indices for trajectory visualization
TRAJECTORY_JOINT_INDICES = {
    'pelvis': 0,
    'left_hand': 22,     # L_Hand in SMPL
    'right_hand': 23,    # R_Hand in SMPL
    'left_foot': 10,     # L_Toe in SMPL
    'right_foot': 11,    # R_Toe in SMPL
}

# Trail length (number of frames to show)
TRAJECTORY_TRAIL_LENGTH = 30


# =====================================================================
# Multi-view Configuration
# =====================================================================

MULTIVIEW_CAMERAS = {
    'front': {'azimuth': 0, 'elevation': -15},
    'side': {'azimuth': 90, 'elevation': -15},
    'top': {'azimuth': 0, 'elevation': 89},
    'three_quarter': {'azimuth': 45, 'elevation': -15},
}


# =====================================================================
# Helper Functions
# =====================================================================

def get_error_color(error: float, max_error: float = 0.1) -> tuple:
    """
    Get color based on error magnitude (green -> yellow -> red).
    
    Args:
        error: Error magnitude in meters
        max_error: Maximum error for full red color
        
    Returns:
        BGR color tuple
    """
    ratio = min(error / max_error, 1.0)
    
    if ratio < 0.5:
        # Green to Yellow
        t = ratio * 2
        r = int(ERROR_COLOR_LOW[2] + t * (ERROR_COLOR_MID[2] - ERROR_COLOR_LOW[2]))
        g = int(ERROR_COLOR_LOW[1] + t * (ERROR_COLOR_MID[1] - ERROR_COLOR_LOW[1]))
        b = int(ERROR_COLOR_LOW[0] + t * (ERROR_COLOR_MID[0] - ERROR_COLOR_LOW[0]))
    else:
        # Yellow to Red
        t = (ratio - 0.5) * 2
        r = int(ERROR_COLOR_MID[2] + t * (ERROR_COLOR_HIGH[2] - ERROR_COLOR_MID[2]))
        g = int(ERROR_COLOR_MID[1] + t * (ERROR_COLOR_HIGH[1] - ERROR_COLOR_MID[1]))
        b = int(ERROR_COLOR_MID[0] + t * (ERROR_COLOR_HIGH[0] - ERROR_COLOR_MID[0]))
    
    return (b, g, r)


def get_trail_alpha(frame_idx: int, current_frame: int, trail_length: int) -> float:
    """
    Get alpha value for trajectory trail based on age.
    
    Args:
        frame_idx: Frame index of the trail point
        current_frame: Current frame being rendered
        trail_length: Maximum trail length
        
    Returns:
        Alpha value (0.0 to 1.0)
    """
    age = current_frame - frame_idx
    if age < 0 or age >= trail_length:
        return 0.0
    return 1.0 - (age / trail_length)


