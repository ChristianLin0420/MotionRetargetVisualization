"""
SMPL Mesh Renderer for Motion Retargeting Visualization

This module renders SMPL human body meshes using PyRender with EGL backend
for headless rendering. Ensures direction alignment with the main pipeline.
"""

import os
# Set environment variables BEFORE importing pyrender
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import cv2
import trimesh
from typing import Tuple, Optional

try:
    import pyrender
    PYRENDER_AVAILABLE = True
except ImportError:
    PYRENDER_AVAILABLE = False
    print("Warning: PyRender not available. Install with: pip install pyrender trimesh")

try:
    import torch
    import smplx
    SMPL_AVAILABLE = True
except ImportError:
    SMPL_AVAILABLE = False
    print("Warning: SMPL not available. Install smplx package.")

from visualization_config import (
    SKELETON_AZIMUTH, ROBOT_AZIMUTH, CAM_DISTANCE, CAM_ELEVATION, CAM_FOVY,
    FLIP_X, COLOR_SMPL_MESH, BG_DARK_GRAY
)


class SMPLMeshRenderer:
    """
    Renders SMPL meshes with consistent direction alignment.
    """
    
    def __init__(self, smpl_model_path: str, width: int = 640, height: int = 720,
                 gender: str = "neutral"):
        """
        Initialize SMPL mesh renderer.
        
        Args:
            smpl_model_path: Path to SMPL model directory
            width: Render width
            height: Render height
            gender: SMPL gender ("neutral", "male", "female")
        """
        if not PYRENDER_AVAILABLE:
            raise RuntimeError("PyRender not available. Install with: pip install pyrender trimesh")
        
        self.width = width
        self.height = height
        self.gender = gender
        
        # Load SMPL model
        if SMPL_AVAILABLE:
            self.smpl_model = smplx.create(
                smpl_model_path,
                model_type='smpl',
                gender=gender,
                batch_size=1
            )
        else:
            self.smpl_model = None
        
        # Setup PyRender scene
        self.scene = pyrender.Scene(
            bg_color=np.array([BG_DARK_GRAY[2], BG_DARK_GRAY[1], BG_DARK_GRAY[0], 255]) / 255.0,
            ambient_light=np.array([0.3, 0.3, 0.3])
        )
        
        # Create renderer
        self.renderer = pyrender.OffscreenRenderer(width, height)
        
        # Setup lighting
        self._setup_lighting()
        
        # Mesh node (will be updated per frame)
        self.mesh_node = None
        
        # Camera parameters matching the main pipeline
        self.azimuth = SKELETON_AZIMUTH
        self.elevation = CAM_ELEVATION
        self.distance = CAM_DISTANCE
        self.fovy = CAM_FOVY
        
    def _setup_lighting(self):
        """Setup scene lighting for good mesh visualization."""
        # Main directional light
        main_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        main_light_pose = np.eye(4)
        main_light_pose[:3, :3] = self._rotation_matrix(np.radians(-45), np.radians(45), 0)
        self.scene.add(main_light, pose=main_light_pose)
        
        # Fill light from opposite side
        fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
        fill_light_pose = np.eye(4)
        fill_light_pose[:3, :3] = self._rotation_matrix(np.radians(45), np.radians(-30), 0)
        self.scene.add(fill_light, pose=fill_light_pose)
        
    def _rotation_matrix(self, rx: float, ry: float, rz: float) -> np.ndarray:
        """Create rotation matrix from euler angles."""
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        return Rz @ Ry @ Rx
    
    def _create_camera_pose(self, lookat: np.ndarray) -> np.ndarray:
        """
        Create camera pose matrix matching the main pipeline's projection.
        
        Args:
            lookat: 3D point to look at
            
        Returns:
            4x4 camera pose matrix
        """
        az_rad = np.radians(self.azimuth)
        el_rad = np.radians(self.elevation)
        
        # Camera position (same as main pipeline)
        cam_x = lookat[0] + self.distance * np.cos(el_rad) * np.sin(az_rad)
        cam_y = lookat[1] - self.distance * np.cos(el_rad) * np.cos(az_rad)
        cam_z = lookat[2] + self.distance * np.sin(el_rad)
        cam_pos = np.array([cam_x, cam_y, cam_z])
        
        # Camera basis vectors
        forward = lookat - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        world_up = np.array([0, 0, 1])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            world_up = np.array([0, 1, 0])
            right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Build camera pose matrix (PyRender convention)
        # PyRender camera looks along -Z, with Y up
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward
        pose[:3, 3] = cam_pos
        
        return pose
    
    def generate_mesh_from_params(self, pose_aa: np.ndarray, betas: np.ndarray,
                                   trans: np.ndarray = None) -> trimesh.Trimesh:
        """
        Generate SMPL mesh from pose and shape parameters.
        
        Args:
            pose_aa: Axis-angle pose parameters (72,) or (24, 3)
            betas: Shape parameters (10,)
            trans: Translation (3,), optional
            
        Returns:
            Trimesh object
        """
        if self.smpl_model is None:
            raise RuntimeError("SMPL model not loaded")
        
        # Reshape pose if needed
        if pose_aa.ndim == 1:
            pose_aa = pose_aa.reshape(1, -1)
        elif pose_aa.ndim == 2 and pose_aa.shape[0] == 24:
            pose_aa = pose_aa.reshape(1, -1)
        
        # Ensure correct shapes
        if betas.ndim == 1:
            betas = betas.reshape(1, -1)
        
        # Convert to torch tensors
        pose_tensor = torch.tensor(pose_aa, dtype=torch.float32)
        betas_tensor = torch.tensor(betas, dtype=torch.float32)
        
        # Handle global orientation and body pose
        global_orient = pose_tensor[:, :3]
        body_pose = pose_tensor[:, 3:72]
        
        # Forward pass through SMPL
        with torch.no_grad():
            output = self.smpl_model(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas_tensor,
                return_verts=True
            )
        
        vertices = output.vertices[0].numpy()
        faces = self.smpl_model.faces
        
        # Apply translation if provided
        if trans is not None:
            vertices = vertices + trans
        
        # Create trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        return mesh
    
    def render_mesh(self, mesh: trimesh.Trimesh, lookat: np.ndarray = None,
                    color: Tuple[int, int, int] = None) -> np.ndarray:
        """
        Render a mesh to an image.
        
        Args:
            mesh: Trimesh object to render
            lookat: 3D point to center camera on (default: mesh centroid)
            color: BGR color for mesh (default: skin-like color)
            
        Returns:
            BGR image as numpy array
        """
        if lookat is None:
            lookat = mesh.centroid
        
        if color is None:
            color = COLOR_SMPL_MESH
        
        # Convert BGR to RGB and normalize
        rgb_color = np.array([color[2], color[1], color[0]]) / 255.0
        
        # Remove old mesh if exists
        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)
        
        # Create PyRender mesh with material
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=np.append(rgb_color, 1.0),
            metallicFactor=0.0,
            roughnessFactor=0.5
        )
        
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        self.mesh_node = self.scene.add(pyrender_mesh)
        
        # Setup camera
        camera = pyrender.PerspectiveCamera(yfov=np.radians(self.fovy))
        camera_pose = self._create_camera_pose(lookat)
        
        # Remove old camera if exists
        for node in list(self.scene.camera_nodes):
            self.scene.remove_node(node)
        
        self.scene.add(camera, pose=camera_pose)
        
        # Render
        color_img, _ = self.renderer.render(self.scene)
        
        # Convert to BGR for OpenCV
        frame = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        
        # Apply horizontal flip to match skeleton direction alignment
        if FLIP_X:
            frame = cv2.flip(frame, 1)
        
        return frame
    
    def render_from_params(self, pose_aa: np.ndarray, betas: np.ndarray,
                           trans: np.ndarray = None, lookat: np.ndarray = None,
                           color: Tuple[int, int, int] = None) -> np.ndarray:
        """
        Render SMPL mesh directly from parameters.
        
        Args:
            pose_aa: Axis-angle pose parameters
            betas: Shape parameters
            trans: Translation (optional)
            lookat: Camera lookat point (optional, default: pelvis)
            color: Mesh color (optional)
            
        Returns:
            BGR image as numpy array
        """
        mesh = self.generate_mesh_from_params(pose_aa, betas, trans)
        
        if lookat is None:
            # Use pelvis position (vertex 0 area or mesh centroid)
            lookat = mesh.centroid
        
        return self.render_mesh(mesh, lookat, color)
    
    def close(self):
        """Clean up renderer resources."""
        self.renderer.delete()


def render_smpl_mesh_panel(pose_aa: np.ndarray, betas: np.ndarray,
                           smpl_model_path: str, width: int = 640, height: int = 720,
                           lookat: np.ndarray = None) -> np.ndarray:
    """
    Convenience function to render a single SMPL mesh panel.
    
    Args:
        pose_aa: Axis-angle pose parameters (72,)
        betas: Shape parameters (10,)
        smpl_model_path: Path to SMPL model directory
        width: Image width
        height: Image height
        lookat: Camera lookat point
        
    Returns:
        BGR image
    """
    renderer = SMPLMeshRenderer(smpl_model_path, width, height)
    try:
        frame = renderer.render_from_params(pose_aa, betas, lookat=lookat)
    finally:
        renderer.close()
    return frame


# Test code
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    print("Testing SMPL Mesh Renderer...")
    print(f"PyRender available: {PYRENDER_AVAILABLE}")
    print(f"SMPL available: {SMPL_AVAILABLE}")
    
    if PYRENDER_AVAILABLE and SMPL_AVAILABLE:
        # Test with sample data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        smpl_path = os.path.join(current_dir, "../data/smpl")
        
        # Create renderer
        renderer = SMPLMeshRenderer(smpl_path, 640, 720)
        
        # Create dummy pose (T-pose)
        pose_aa = np.zeros(72)
        betas = np.zeros(10)
        
        # Render
        frame = renderer.render_from_params(pose_aa, betas)
        
        # Save test image
        output_path = os.path.join(current_dir, "../output/test_smpl_mesh.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, frame)
        print(f"Saved test image to: {output_path}")
        
        renderer.close()
    else:
        print("Cannot run test - missing dependencies")


