"""
Joint Angle Comparison Plotter

This script generates synchronized videos showing:
- Top: Robot animation
- Bottom: Real-time joint angle plots comparing SMPL vs Robot

Ensures direction alignment with the main pipeline.
"""

import os
# Set environment variable BEFORE importing mujoco
os.environ['MUJOCO_GL'] = 'egl'

import joblib
import numpy as np
import mujoco
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm
from typing import Dict, List, Tuple
from scipy.spatial.transform import Rotation

from visualization_config import (
    SKELETON_AZIMUTH, ROBOT_AZIMUTH, CAM_DISTANCE, CAM_ELEVATION,
    ROBOT_JOINT_NAMES, ACTIVE_ROBOT_JOINT_NAMES, BG_DARK_GRAY
)


# =====================================================================
# Joint Angle Extraction
# =====================================================================

# Joints to compare (robot_joint_name, smpl_joint_index_for_parent, smpl_joint_index_for_child)
COMPARISON_JOINTS = {
    'Left Hip': {
        'robot_joint': 'left_hip_pitch_joint',
        'smpl_parent': 0,   # Pelvis
        'smpl_child': 1,    # L_Hip
        'color': '#FF6B6B'
    },
    'Left Knee': {
        'robot_joint': 'left_knee_joint',
        'smpl_parent': 1,   # L_Hip
        'smpl_child': 4,    # L_Knee
        'color': '#4ECDC4'
    },
    'Right Hip': {
        'robot_joint': 'right_hip_pitch_joint',
        'smpl_parent': 0,   # Pelvis
        'smpl_child': 2,    # R_Hip
        'color': '#45B7D1'
    },
    'Right Knee': {
        'robot_joint': 'right_knee_joint',
        'smpl_parent': 2,   # R_Hip
        'smpl_child': 5,    # R_Knee
        'color': '#96CEB4'
    },
    'Left Shoulder': {
        'robot_joint': 'left_shoulder_pitch_joint',
        'smpl_parent': 13,  # L_Thorax
        'smpl_child': 16,   # L_Shoulder
        'color': '#FFEAA7'
    },
    'Left Elbow': {
        'robot_joint': 'left_elbow_joint',
        'smpl_parent': 16,  # L_Shoulder
        'smpl_child': 18,   # L_Elbow
        'color': '#DDA0DD'
    },
    'Right Shoulder': {
        'robot_joint': 'right_shoulder_pitch_joint',
        'smpl_parent': 14,  # R_Thorax
        'smpl_child': 17,   # R_Shoulder
        'color': '#98D8C8'
    },
    'Right Elbow': {
        'robot_joint': 'right_elbow_joint',
        'smpl_parent': 17,  # R_Shoulder
        'smpl_child': 19,   # R_Elbow
        'color': '#F7DC6F'
    },
}


def extract_smpl_joint_angles(smpl_pose_aa: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract joint angles from SMPL axis-angle pose parameters.
    
    Args:
        smpl_pose_aa: (T, 72) axis-angle pose parameters
        
    Returns:
        Dictionary mapping joint names to angle arrays (in degrees)
    """
    T = smpl_pose_aa.shape[0]
    angles = {}
    
    for joint_name, config in COMPARISON_JOINTS.items():
        child_idx = config['smpl_child']
        # Extract axis-angle for this joint (3 values per joint)
        joint_aa = smpl_pose_aa[:, child_idx * 3:(child_idx + 1) * 3]
        
        # Convert axis-angle to angle magnitude (rotation amount)
        angle_rad = np.linalg.norm(joint_aa, axis=1)
        angle_deg = np.degrees(angle_rad)
        
        # For joints like knee, we want flexion angle which is typically around one axis
        # Use the primary axis (usually pitch for limbs)
        angles[joint_name] = angle_deg
    
    return angles


def extract_robot_joint_angles(joint_positions: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract robot joint angles from DOF values.
    
    Args:
        joint_positions: (T, num_joints) joint position array
        
    Returns:
        Dictionary mapping joint names to angle arrays (in degrees)
    """
    angles = {}
    
    # Get indices for active joints
    active_indices = {name: i for i, name in enumerate(ACTIVE_ROBOT_JOINT_NAMES)}
    
    for joint_name, config in COMPARISON_JOINTS.items():
        robot_joint = config['robot_joint']
        if robot_joint in active_indices:
            idx = active_indices[robot_joint]
            # Robot DOF is already in radians
            angles[joint_name] = np.degrees(joint_positions[:, idx])
        else:
            angles[joint_name] = np.zeros(joint_positions.shape[0])
    
    return angles


# =====================================================================
# Plotting Functions
# =====================================================================

def create_angle_plot_frame(smpl_angles: Dict[str, np.ndarray],
                            robot_angles: Dict[str, np.ndarray],
                            current_frame: int, total_frames: int,
                            width: int = 1280, height: int = 400) -> np.ndarray:
    """
    Create a single frame of the joint angle comparison plot.
    
    Args:
        smpl_angles: Dictionary of SMPL joint angles
        robot_angles: Dictionary of robot joint angles
        current_frame: Current frame index
        total_frames: Total number of frames
        width: Plot width
        height: Plot height
        
    Returns:
        BGR image of the plot
    """
    # Create figure with subplots for each joint pair
    n_joints = len(COMPARISON_JOINTS)
    n_cols = 4
    n_rows = 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width / 100, height / 100), dpi=100)
    axes = axes.flatten()
    
    fig.patch.set_facecolor('#282828')
    
    time = np.arange(total_frames) / 30.0  # Assuming 30 fps
    current_time = current_frame / 30.0
    
    for i, (joint_name, config) in enumerate(COMPARISON_JOINTS.items()):
        ax = axes[i]
        ax.set_facecolor('#282828')
        
        color = config['color']
        
        # Plot SMPL angles (dashed)
        smpl_angle = smpl_angles[joint_name]
        ax.plot(time, smpl_angle, '--', color=color, alpha=0.7, linewidth=1.5, label='SMPL')
        
        # Plot robot angles (solid)
        robot_angle = robot_angles[joint_name]
        ax.plot(time, robot_angle, '-', color=color, linewidth=2, label='Robot')
        
        # Draw playhead
        ax.axvline(x=current_time, color='white', linewidth=1.5, alpha=0.8)
        
        # Mark current values
        if current_frame < len(smpl_angle):
            ax.scatter([current_time], [smpl_angle[current_frame]], color=color, s=30, zorder=5, alpha=0.7)
            ax.scatter([current_time], [robot_angle[current_frame]], color=color, s=50, zorder=5)
        
        # Styling
        ax.set_title(joint_name, color='white', fontsize=9, pad=3)
        ax.set_xlabel('Time (s)', color='gray', fontsize=7)
        ax.set_ylabel('Angle (°)', color='gray', fontsize=7)
        ax.tick_params(colors='gray', labelsize=6)
        ax.spines['bottom'].set_color('gray')
        ax.spines['left'].set_color('gray')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2, color='gray')
        
        # Set x limits
        ax.set_xlim(0, time[-1])
    
    # Add legend to first subplot
    axes[0].legend(loc='upper right', fontsize=6, framealpha=0.5,
                   facecolor='#282828', edgecolor='gray', labelcolor='white')
    
    plt.tight_layout(pad=0.5)
    
    # Convert to image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    # Get the RGBA buffer
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)
    
    # Convert RGBA to BGR
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    
    plt.close(fig)
    
    return img_bgr


# =====================================================================
# Video Recording
# =====================================================================

def record_angle_comparison_video(motion_data: dict, output_path: str,
                                   model_path: str, caption: str = "",
                                   fps: int = 30, width: int = 1280) -> bool:
    """
    Record video with robot animation on top and joint angle plots on bottom.
    
    Args:
        motion_data: Dictionary containing motion data
        output_path: Output video path
        model_path: Path to MuJoCo model XML
        caption: Optional caption text
        fps: Frames per second
        width: Video width
        
    Returns:
        True if successful
    """
    # Extract data
    root_position = motion_data.get("root_trans_offset")
    root_orientation = motion_data.get("root_rot")
    joint_positions_all = motion_data.get("dof")
    smpl_pose_aa = motion_data.get("smpl_pose_aa")
    
    if any(x is None for x in [root_position, root_orientation, joint_positions_all]):
        print("Missing required data")
        return False
    
    # Convert quaternions from XYZW to WXYZ for MuJoCo
    if root_orientation.shape[1] == 4:
        root_orientation = root_orientation[:, [3, 0, 1, 2]]
    
    # Robot joint configuration
    active_robot_joint_indices = [ROBOT_JOINT_NAMES.index(j) for j in ACTIVE_ROBOT_JOINT_NAMES]
    joint_positions = joint_positions_all[:, active_robot_joint_indices]
    
    # Load MuJoCo model
    robot_height = 480
    plot_height = 320
    total_height = robot_height + plot_height
    
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        model.vis.global_.offwidth = width
        model.vis.global_.offheight = robot_height
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=robot_height, width=width)
    except Exception as e:
        print(f"Error loading MuJoCo model: {e}")
        return False
    
    T = root_position.shape[0]
    floating_base_dof = 7
    n_joints = model.nq - floating_base_dof
    
    if joint_positions.shape[1] != n_joints:
        print(f"Joint mismatch: expected {n_joints}, got {joint_positions.shape[1]}")
        return False
    
    # Extract joint angles
    if smpl_pose_aa is not None:
        smpl_angles = extract_smpl_joint_angles(smpl_pose_aa)
    else:
        # If no SMPL pose data, use zeros
        smpl_angles = {name: np.zeros(T) for name in COMPARISON_JOINTS.keys()}
    
    robot_angles = extract_robot_joint_angles(joint_positions)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, total_height))
    
    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        return False
    
    # Camera parameters (aligned with main pipeline)
    cam_distance = CAM_DISTANCE
    cam_elevation = CAM_ELEVATION
    robot_azimuth = ROBOT_AZIMUTH
    
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
            
            # Setup camera
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.distance = cam_distance
            camera.azimuth = robot_azimuth
            camera.elevation = cam_elevation
            camera.lookat[:] = pos
            
            # Render robot
            renderer.update_scene(data, camera=camera)
            robot_frame = renderer.render()
            robot_frame = cv2.cvtColor(robot_frame, cv2.COLOR_RGB2BGR)
            
            # Add label to robot view
            cv2.putText(robot_frame, "G1 Robot Motion", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add caption if provided
            if caption:
                cv2.rectangle(robot_frame, (10, robot_height - 50),
                              (min(len(caption) * 10, width - 10), robot_height - 20),
                              (0, 0, 0), -1)
                cv2.putText(robot_frame, caption[:100], (15, robot_height - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Create angle plot frame
            plot_frame = create_angle_plot_frame(
                smpl_angles, robot_angles, t, T,
                width=width, height=plot_height
            )
            
            # Resize plot if needed
            if plot_frame.shape[1] != width or plot_frame.shape[0] != plot_height:
                plot_frame = cv2.resize(plot_frame, (width, plot_height))
            
            # Combine robot view and plot
            combined = np.vstack([robot_frame, plot_frame])
            
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
    
    parser = argparse.ArgumentParser(description="Joint angle comparison visualization")
    parser.add_argument("--motion", type=str, default=None,
                        help="Specific motion name to visualize (e.g., '000000')")
    parser.add_argument("--max_motions", type=int, default=5,
                        help="Maximum number of motions to process")
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
        output_dir = os.path.join(current_dir, "../output/videos_angles")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load motion dataset
    print("Loading motion dataset...")
    motion_dataset = joblib.load(motion_data_path)
    motion_dataset.pop("config", None)
    
    # Filter motions
    if args.motion:
        motion_names = [name for name in motion_dataset.keys() 
                        if args.motion in name and not name.startswith("M")]
    else:
        motion_names = [name for name in motion_dataset.keys() if not name.startswith("M")]
        if args.max_motions:
            motion_names = motion_names[:args.max_motions]
    
    print(f"Processing {len(motion_names)} motions...")
    
    for motion_name in tqdm(motion_names, desc="Recording angle videos"):
        motion_data = motion_dataset[motion_name]
        captions = motion_data.get("captions", [])
        caption = captions[0] if captions else ""
        
        output_path = os.path.join(output_dir, f"{motion_name.replace('.npz', '')}_angles.mp4")
        
        success = record_angle_comparison_video(
            motion_data, output_path, model_path, caption=caption,
            fps=30, width=1280
        )
        
        if success:
            print(f"  Saved: {output_path}")
        else:
            print(f"  Failed: {motion_name}")
    
    print(f"\nDone! Videos saved to: {output_dir}")


