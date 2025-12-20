"""
Headless MuJoCo video recorder for servers without display.
Uses EGL for offscreen rendering to record motion visualizations to video files.
"""

import os
# Set environment variable BEFORE importing mujoco
os.environ['MUJOCO_GL'] = 'egl'

import joblib
import numpy as np
import mujoco
import cv2
from tqdm import tqdm


def record_motion_video_with_caption(root_positions, root_orientations_quat, joint_positions,
                                      output_path, model_path, caption="", fps=30, 
                                      width=1280, height=720):
    """
    Record motion with caption overlay using EGL offscreen rendering.
    """
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        # Set offscreen buffer size
        model.vis.global_.offwidth = width
        model.vis.global_.offheight = height
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading MuJoCo model: {e}")
        return False

    T = root_positions.shape[0]
    floating_base_dof = 7
    n_joints = model.nq - floating_base_dof
    
    if joint_positions.shape[1] != n_joints:
        print(f"Joint positions dimension mismatch: expected {n_joints}, got {joint_positions.shape[1]}")
        return False

    # Create renderer
    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
    except Exception as e:
        print(f"Error creating renderer: {e}")
        return False
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        return False

    try:
        for t in tqdm(range(T), desc="Rendering frames", leave=False):
            pos = root_positions[t]
            quat = root_orientations_quat[t]

            data.qpos[0:3] = pos
            data.qpos[3:7] = quat
            data.qpos[7:] = joint_positions[t]
            data.qvel[:] = 0

            mujoco.mj_forward(model, data)

            # Setup camera
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            camera.trackbodyid = 0
            camera.distance = 3.0
            camera.azimuth = 90
            camera.elevation = -20
            camera.lookat[:] = pos

            renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Add caption overlay
            if caption:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                color = (255, 255, 255)
                bg_color = (0, 0, 0)
                
                # Word wrap the caption
                max_width = width - 40
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
                
                # Draw text with background
                y_offset = 30
                for line in lines:
                    (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
                    cv2.rectangle(frame_bgr, (10, y_offset - text_height - 5), 
                                  (20 + text_width, y_offset + baseline + 5), bg_color, -1)
                    cv2.putText(frame_bgr, line, (15, y_offset), font, font_scale, color, thickness)
                    y_offset += text_height + 15

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


if __name__ == "__main__":
    # Path setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    motion_data_path = os.path.join(current_dir, "../data/g1/humanml3d_train_retargeted_wholebody_82.pkl")
    model_path = os.path.join(current_dir, '../resources/robots/g1/g1_27dof.xml')
    output_dir = os.path.join(current_dir, "../output/videos")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load motion dataset
    print("Loading motion dataset...")
    motion_dataset = joblib.load(motion_data_path)
    motion_dataset.pop("config", None)

    # All joint names in the robot model
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

    # Active joint names
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

    # Get indices of active joints
    active_robot_joint_indices = [robot_joint_names.index(j) for j in active_robot_joint_names]

    # Filter motions (skip mirrored ones)
    motion_names = [name for name in motion_dataset.keys() if not name.startswith("M")]
    
    # Limit to first N motions for testing (set to None for all)
    max_motions = 5  # Change to None to record all
    if max_motions:
        motion_names = motion_names[:max_motions]

    print(f"Recording {len(motion_names)} motions to videos...")
    
    for motion_name in tqdm(motion_names, desc="Recording videos"):
        motion_data = motion_dataset[motion_name]
        
        # Extract data
        root_position = motion_data.get("root_trans_offset")
        root_orientation = motion_data.get("root_rot")
        joint_positions_all = motion_data.get("dof")
        captions = motion_data.get("captions", [])
        
        if root_position is None or root_orientation is None or joint_positions_all is None:
            print(f"Skipping {motion_name}: missing data")
            continue
        
        # Convert quaternions from XYZW to WXYZ
        if root_orientation.shape[1] == 4:
            root_orientation = root_orientation[:, [3, 0, 1, 2]]
        
        # Get active joint positions
        joint_positions = joint_positions_all[:, active_robot_joint_indices]
        
        # Prepare caption
        caption = captions[0] if captions else ""
        
        # Output path
        output_path = os.path.join(output_dir, f"{motion_name.replace('.npz', '')}.mp4")
        
        # Record video
        success = record_motion_video_with_caption(
            root_position, root_orientation, joint_positions,
            output_path, model_path, caption=caption,
            fps=30, width=1280, height=720
        )
        
        if success:
            print(f"  Saved: {output_path}")
        else:
            print(f"  Failed: {motion_name}")

    print(f"\nDone! Videos saved to: {output_dir}")
