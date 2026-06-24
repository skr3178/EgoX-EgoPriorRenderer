import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

# PyTorch3D for rendering and camera transformations
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)
from pytorch3d.renderer.fisheyecameras import FishEyeCameras

# Import ViPE modules
from vipe.slam.interface import SLAMMap
from vipe.utils.io import (
    ArtifactPath,
    read_depth_artifacts,
    read_instance_artifacts,
    read_intrinsics_artifacts,
    read_pose_artifacts,
    read_rgb_artifacts,
)
from vipe.utils.cameras import CameraType
from vipe.utils.depth import reliable_depth_mask_range

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_background(W, H, mode="solid", color=(0.0, 0.0, 0.0), noise_range=(0, 255), seed=42):
    """
    Generate background for inpainting-friendly rendering.
    
    Args:
        W, H: Image dimensions
        mode: 'solid', 'uniform_noise', 'gaussian_noise'
        color: RGB tuple in [0,1] for solid mode
        noise_range: (min, max) for noise generation in [0,255]
        seed: Random seed for reproducible noise
        
    Returns:
        Background image as (H, W, 3) float32 array in [0,1]
    """
    np.random.seed(seed)
    
    if mode == "solid":
        return np.full((H, W, 3), color, dtype=np.float32)
    
    elif mode == "uniform_noise":
        min_val, max_val = noise_range
        noise = np.random.uniform(min_val, max_val, (H, W, 3))
        return (noise / 255.0).astype(np.float32)
    
    elif mode == "gaussian_noise":
        min_val, max_val = noise_range
        center = (min_val + max_val) / 2.0
        std = (max_val - min_val) / 6.0  # 99.7% of values within range
        noise = np.random.normal(center, std, (H, W, 3))
        noise = np.clip(noise, min_val, max_val)
        return (noise / 255.0).astype(np.float32)
    
    else:
        raise ValueError(f"Unknown background mode: {mode}")
    
def scale_intrinsics(*args):
    """
    Scale camera intrinsics to a new image size.

    지원하는 입력 형식:
    1) fx, fy, cx, cy, original_size, new_size
       -> (new_fx, new_fy, new_cx, new_cy) 반환
    2) f, cx, cy, original_size, new_size   (fx = fy = f)
       -> (new_f, new_cx, new_cy) 반환
    """
    if len(args) == 6:
        fx, fy, cx, cy, original_size, new_size = args
        mode = "full"
    elif len(args) == 5:
        f, cx, cy, original_size, new_size = args
        fx, fy = f, f
        mode = "single"
    else:
        raise ValueError(
            "지원하는 입력 형식:\n"
            "  scale_intrinsics(fx, fy, cx, cy, original_size, new_size)\n"
            "  scale_intrinsics(f, cx, cy, original_size, new_size)"
        )

    original_H, original_W = original_size
    new_H, new_W = new_size

    if original_H == 0 or original_W == 0:
        logger.warning("Original image size has a zero dimension, cannot scale intrinsics.")
        if mode == "full":
            return fx, fy, cx, cy
        else:
            return fx, cx, cy

    scale_w = new_W / original_W
    scale_h = new_H / original_H

    new_fx = fx * scale_w
    new_fy = fy * scale_h
    new_cx = cx * scale_w
    new_cy = cy * scale_h

    logger.info(f"Scaled intrinsics from {original_W}x{original_H} to {new_W}x{new_H}")
    logger.info(f"  fx: {fx:.1f} -> {new_fx:.1f}")
    logger.info(f"  fy: {fy:.1f} -> {new_fy:.1f}")
    logger.info(f"  cx: {cx:.1f} -> {new_cx:.1f}")
    logger.info(f"  cy: {cy:.1f} -> {new_cy:.1f}")

    if mode == "full":
        return new_fx, new_fy, new_cx, new_cy
    else:
        return new_fx, new_cx, new_cy

def load_aria_distortion_coeffs(
        online_calibration_path: Optional[str] = None,
        frame_idx: int = 0
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load Aria camera distortion coefficients from Ego4D online calibration file.
    If online_calibration_path is None, returns preset fisheye distortion coefficients for FisheyeRadTanThinPrism model
    
    Args:
        online_calibration_path: Path to online_calibration.jsonl file (optional)
        frame_idx: Frame index to extract calibration from (default: 0 for first frame)
    
    Returns:
        A tuple containing:
        - radial_distortion_coeffs: Radial distortion coefficients [k1, k2, k3, k4, k5, k6]
        - tangential_distortion_coeffs: Tangential distortion coefficients [p1, p2]
        - thinPrism_distortion_coeffs: Thin prism distortion coefficients [s1, s2, s3, s4]
        - focal_length: Single focal length [f] (None if using defaults)
        - principal_point: Principal point [cx, cy] (None if using defaults)
    """
    import json
    
    # If no calibration file provided, return default Ego-Exo4D distortion coefficients
    if online_calibration_path is None:
        logger.info("No online_calibration_path provided. Using default Ego-Exo4D fisheye distortion coefficients.")
        
        # Default Ego-Exo4D Aria camera distortion coefficients
        radial_distortion_coeffs = np.array([
            -0.02340373583137989, 0.09388021379709244, -0.06088035926222801,
            0.0053304750472307205, 0.003342868760228157, -0.0006356257363222539
        ])
        tangential_distortion_coeffs = np.array([0.0005087381578050554, -0.0004747129278257489])
        thinPrism_distortion_coeffs = np.array([
            -0.0011330085108056664, -0.00025734835071489215,
            0.00009328465239377692, 0.00009424977179151028
        ])
        
        logger.info(f"Default radial distortion coeffs: k1={radial_distortion_coeffs[0]:.6f}, k2={radial_distortion_coeffs[1]:.6f}, k3={radial_distortion_coeffs[2]:.6f}, k4={radial_distortion_coeffs[3]:.6f}, k5={radial_distortion_coeffs[4]:.6f}, k6={radial_distortion_coeffs[5]:.6f}")
        logger.info(f"Default tangential distortion coeffs: p1={tangential_distortion_coeffs[0]:.10f}, p2={tangential_distortion_coeffs[1]:.10f}")
        logger.info(f"Default thin prism distortion coeffs: s1={thinPrism_distortion_coeffs[0]:.10f}, s2={thinPrism_distortion_coeffs[1]:.10f}, s3={thinPrism_distortion_coeffs[2]:.10f}, s4={thinPrism_distortion_coeffs[3]:.10f}")
        
        # Return distortion coefficients without original_size (caller should provide it)
        # focal_length and principal_point are also None (will be taken from ego_intrinsics)
        return radial_distortion_coeffs, tangential_distortion_coeffs, thinPrism_distortion_coeffs, None, None
    
    logger.info(f"Loading Aria distortion coefficients from {online_calibration_path}")
    
    # Read the JSONL file
    with open(online_calibration_path, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        logger.error("online_calibration.jsonl is empty.")
        return None, None, None, None, None

    if frame_idx >= len(lines):
        logger.warning(f"Frame {frame_idx} not found, using frame 0")
        frame_idx = 0
    
    # Parse the JSON for the specified frame
    calibration_data = json.loads(lines[frame_idx])
    
    # Find the camera-rgb (Aria RGB camera) calibration
    aria_calib = None
    for cam_calib in calibration_data.get('CameraCalibrations', []):
        if cam_calib.get('Label') == 'camera-rgb':
            aria_calib = cam_calib
            break
    
    if aria_calib is None:
        logger.error("camera-rgb (Aria) calibration not found in online_calibration.jsonl")
        return None, None, None, None, None
    
    # Extract projection parameters
    projection = aria_calib.get('Projection')
    if not projection or projection.get('Name') != 'FisheyeRadTanThinPrism':
        logger.warning(f"Expected FisheyeRadTanThinPrism, but not found or type is different.")
        return None, None, None, None, None
    
    params = projection.get('Params', [])
    if len(params) < 10:
        logger.error("Insufficient parameters for FisheyeRadTanThinPrism model.")
        return None, None, None, None, None

    # FisheyeRadTanThinPrism parameter layout:
    # [f, cx, cy, k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4] maybe...
    f = params[0]
    cx, cy = params[1], params[2]
    # (k: radial distortion, p: tangential distortion, s: thinPrism distortion)
    k1, k2, k3, k4, k5, k6 = params[3], params[4], params[5], params[6], params[7], params[8]
    p1, p2 = params[9], params[10]
    s1, s2, s3, s4 = params[11], params[12], params[13], params[14]

    
    radial_distortion_coeffs = np.array([k1, k2, k3, k4, k5, k6])
    tangential_distortion_coeffs = np.array([p1, p2])
    thinPrism_distortion_coeffs = np.array([s1, s2, s3, s4])
    focal_length = np.array([f])
    principal_point = np.array([cx, cy])
    
    logger.info(f"Loaded Aria distortion coefficients:")
    logger.info(f"  Focal length: f={f:.1f}")
    logger.info(f"  Principal point: cx={cx:.1f}, cy={cy:.1f}")
    logger.info(f"  Radial distortion coeffs: k1={k1:.6f}, k2={k2:.6f}, k3={k3:.6f}, k4={k4:.6f}, k5={k5:.6f}, k6={k6:.6f}")
    logger.info(f"  Tangential distortion coeffs: p1={p1:.6f}, p2={p2:.6f}")
    logger.info(f"  ThinPrism distortion coeffs: s1={s1:.6f}, s2={s2:.6f}, s3={s3:.6f}, s4={s4:.6f}")
    
    return radial_distortion_coeffs, tangential_distortion_coeffs, thinPrism_distortion_coeffs, focal_length, principal_point


def load_camera_params_from_meta(meta_json_path: str, input_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Load camera parameters from meta.json based on input directory.
    
    Args:
        meta_json_path: Path to meta.json file
        input_dir: Input directory path (e.g., 'vipe_results/joker')
    
    Returns:
        exo_intrinsic: 3x3 exo camera intrinsics matrix
        exo_extrinsic: 4x4 exo camera extrinsics matrix (world to cam)
        ego_intrinsic: 3x3 ego camera intrinsics matrix  
        ego_extrinsics: List of 4x4 ego camera extrinsics matrices for each frame (world to cam)
    """
    import json
    from pathlib import Path
    
    # Get video name from input_dir stem (e.g., 'joker' from 'vipe_results/joker')
    video_name = Path(input_dir).stem
    logger.info(f"Looking for video_name='{video_name}' in meta.json")
    
    # Load meta.json
    with open(meta_json_path, 'r') as f:
        meta_data = json.load(f)
    
    # Find matching sample
    matching_sample = None
    for sample in meta_data.get('test_datasets', []):
        exo_path = sample.get('exo_path', '')
        # Check if parent directory name matches video_name
        # e.g., './example/in_the_wild/videos/joker/exo.mp4' -> parent = 'joker'
        exo_parent = Path(exo_path).parent.name
        if exo_parent == video_name:
            matching_sample = sample
            logger.info(f"Found matching sample: exo_path='{exo_path}'")
            break
    
    if matching_sample is None:
        raise ValueError(f"No matching sample found for video_name='{video_name}' in {meta_json_path}")
    
    # Extract camera parameters
    exo_intrinsic = np.array(matching_sample['camera_intrinsics'], dtype=np.float32)
    exo_extrinsic_3x4 = np.array(matching_sample['camera_extrinsics'], dtype=np.float32)
    
    ego_intrinsic = np.array(matching_sample['ego_intrinsics'], dtype=np.float32)
    ego_extrinsics_3x4 = matching_sample['ego_extrinsics']
    
    # Convert exo extrinsics from 3x4 to 4x4 by adding [0, 0, 0, 1] row
    exo_extrinsic = np.eye(4, dtype=np.float32)
    exo_extrinsic[:3, :] = exo_extrinsic_3x4
    
    # Convert ego extrinsics from 3x4 to 4x4 by adding [0, 0, 0, 1] row
    ego_extrinsics = []
    for ext_3x4 in ego_extrinsics_3x4:
        ext_3x4_array = np.array(ext_3x4, dtype=np.float32)
        ext_4x4 = np.eye(4, dtype=np.float32)
        ext_4x4[:3, :] = ext_3x4_array
        ego_extrinsics.append(ext_4x4)
    
    logger.info(f"Loaded exo_intrinsic: {exo_intrinsic.shape}")
    logger.info(f"Loaded exo_extrinsic: {exo_extrinsic.shape} (world to cam, 4x4)")
    logger.info(f"Loaded ego_intrinsic: {ego_intrinsic.shape}")
    logger.info(f"Loaded {len(ego_extrinsics)} ego_extrinsics (world to cam, 4x4)")
    
    return exo_intrinsic, exo_extrinsic, ego_intrinsic, ego_extrinsics


def build_background_pointcloud(
        input_dir: str,
        T_cam_to_world: np.ndarray,
        spatial_subsample: int = 2
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], np.ndarray]:
    """
    Build background-only global point cloud from all frames in world coordinates.

    This function prioritizes SLAM map if available, otherwise falls back to depth-based reconstruction.
    It always keeps only background points (instance id == 0 when using depth method).

    Args:
        input_dir: Path to ViPE inference results
        T_cam_to_world: 4x4 transformation matrix from camera to world coordinates
        spatial_subsample: Spatial subsampling factor for point cloud
    
    Returns:
        global_points: Nx3 array of 3D points in world coordinates
        global_colors: Nx3 array of RGB colors (0-255)
        image_size: (height, width) of original frames
    """
    # Use ViPE's ArtifactPath to find inference artifacts
    artifacts = list(ArtifactPath.glob_artifacts(Path(input_dir), use_video=True))
    if not artifacts:
        raise ValueError(f"No ViPE artifacts found in {input_dir}")
    artifact_path = artifacts[0]


    # Fallback to original depth-based method
    global_points = []
    global_colors = []

    def none_it(inner_it):
        try:
            for item in inner_it:
                yield item
        except FileNotFoundError:
            while True:
                yield None, None

    def none_it_mask(inner_it):
        try:
            for item in inner_it:
                yield item
        except FileNotFoundError:
            while True:
                yield None, None

    logger.info("Building background point cloud from all frames using depth...")

    rays = None
    frame_count = 0
    image_size = None  # Will be set from first frame

    for frame_idx, (c2w, (_, rgb), intr, camera_type, (_, depth), (_, instance_mask)) in enumerate(
        zip(
            read_pose_artifacts(artifact_path.pose_path)[1].matrix().numpy(),
            read_rgb_artifacts(artifact_path.rgb_path),
            *read_intrinsics_artifacts(artifact_path.intrinsics_path, artifact_path.camera_type_path)[1:3],
            none_it(read_depth_artifacts(artifact_path.depth_path)),
            none_it_mask(read_instance_artifacts(artifact_path.mask_path)),
        )
    ):
        if depth is None:
            continue

        frame_count += 1
        logger.info(f"Processing frame {frame_idx}")

        frame_height, frame_width = rgb.shape[:2]

        # Set image size from first frame
        if image_size is None:
            image_size = (frame_height, frame_width)
            logger.info(f"Detected original image size: {frame_height} x {frame_width}")

        sampled_rgb = (rgb.cpu().numpy() * 255).astype(np.uint8)
        sampled_rgb = sampled_rgb[::spatial_subsample, ::spatial_subsample]

        # Build rays if not already built
        if rays is None:
            camera_model = camera_type.build_camera_model(intr)
            disp_v, disp_u = torch.meshgrid(
                torch.arange(frame_height).float()[::spatial_subsample],
                torch.arange(frame_width).float()[::spatial_subsample],
                indexing="ij",
            )
            if camera_type == CameraType.PANORAMA:
                disp_v = disp_v / (frame_height - 1)
                disp_u = disp_u / (frame_width - 1)
            disp = torch.ones_like(disp_v)
            pts, _, _ = camera_model.iproj_disp(disp, disp_u, disp_v)
            rays = pts[..., :3].numpy()
            if camera_type != CameraType.PANORAMA:
                rays /= rays[..., 2:3]

        # Generate point cloud in camera coordinates
        pcd_camera = rays * depth.numpy()[::spatial_subsample, ::spatial_subsample, None]

        # Convert to world coordinates
        pcd_camera_flat = pcd_camera.reshape(-1, 3)  # (N, 3)
        pcd_world_flat = (c2w[:3, :3] @ pcd_camera_flat.T + c2w[:3, 3:4]).T
        pcd_world = pcd_world_flat.reshape(pcd_camera.shape)  # Restore original shape

        # Apply depth mask
        depth_mask = reliable_depth_mask_range(depth)[::spatial_subsample, ::spatial_subsample].numpy()

        # Keep only background (instance == 0) when instance mask is available
        if instance_mask is not None:
            instance_mask_np = instance_mask.cpu().numpy() if hasattr(instance_mask, 'cpu') else instance_mask
            static_mask = (instance_mask_np == 0)  # Keep only background
            static_mask_sub = static_mask[::spatial_subsample, ::spatial_subsample]
            depth_mask = depth_mask & static_mask_sub
            logger.info(f"Frame {frame_idx}: Applied background filter (kept {np.sum(depth_mask)} / {depth_mask.size} points)")

        # Flatten and filter valid points
        pcd_flat = pcd_world.reshape(-1, 3)
        rgb_flat = sampled_rgb.reshape(-1, 3)
        mask_flat = depth_mask.reshape(-1)

        valid_points = pcd_flat[mask_flat]
        valid_colors = rgb_flat[mask_flat]

        global_points.append(valid_points)
        global_colors.append(valid_colors)

        logger.info(f"Frame {frame_idx}: Added {len(valid_points)} points to background point cloud")

    # Concatenate all points
    if global_points:
        global_points = np.concatenate(global_points, axis=0)
        global_colors = np.concatenate(global_colors, axis=0)
        logger.info(f"Built background point cloud with {len(global_points)} total points from {frame_count} frames")

        # Transform points from ViPE coordinate system to world coordinate system
        logger.info("Transforming background point cloud from ViPE to world coordinate system...")
        logger.info(f"Using T_cam_to_world transformation matrix:\n{T_cam_to_world}")

        points_homogeneous = np.hstack([global_points, np.ones((len(global_points), 1))])
        global_points_transformed = (T_cam_to_world @ points_homogeneous.T).T[:, :3]

        global_points = global_points_transformed
        logger.info("Successfully transformed background point cloud to world coordinate system")

        # Store transformation info for consistency checks
        # (caller unpacks 3; T_cam_to_world is the same object it passed in)
        return global_points, global_colors, image_size

    else:
        global_points = np.empty((0, 3))
        global_colors = np.empty((0, 3))
        logger.warning("No valid background points found in any frame")
        T_cam_to_world = np.eye(4)
        return global_points, global_colors, image_size
    

def compute_robust_mean_tensors(stacked_depth: torch.Tensor, stacked_rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU 벡터화된 버전: 픽셀별 NaN-aware 평균을 병렬로 계산.
    """
    logger.info("Computing mean background using nanmean with GPU vectorization...")
    
    # NaN을 무시하고 평균 계산
    robust_depth = torch.nanmean(stacked_depth, dim=0)
    robust_rgb = torch.nanmean(stacked_rgb, dim=0)
    
    # NaN 값들을 0으로 대체
    robust_depth = torch.nan_to_num(robust_depth, nan=0.0)
    robust_rgb = torch.nan_to_num(robust_rgb, nan=0.0)
    
    return robust_depth, robust_rgb



def build_mean_background_pointcloud(
        input_dir: str,
        T_cam_to_world: np.ndarray,
        spatial_subsample: int = 2,
        max_frames: int = None
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], np.ndarray]:
    """
    Build a static background point cloud using nanmean.
    
    Args:
        input_dir: Directory containing ViPE artifacts
        T_cam_to_world: 4x4 transformation matrix from camera to world coordinates
        spatial_subsample: Spatial subsampling factor for point cloud
        max_frames: Maximum number of frames to process (None for all)

    Returns:
        mean_bg_points: Nx3 array of 3D points in world coordinates (nanmean background)  
        mean_bg_colors: Nx3 array of RGB colors (0-255) (nanmean background)
        image_size: (height, width) of original frames
    """
    artifacts = list(ArtifactPath.glob_artifacts(Path(input_dir), use_video=True))
    if not artifacts:
        raise ValueError(f"No ViPE artifacts found in {input_dir}")
    artifact_path = artifacts[0]

    # 최적화: 모든 유효한 프레임을 스택으로 수집
    valid_depth_frames = []
    valid_rgb_frames = []
    image_size = None

    def none_it(inner_it):
        try:
            for item in inner_it:
                yield item
        except FileNotFoundError:
            while True:
                yield None, None

    logger.info("Collecting valid background frames for GPU-optimized batch processing...")

    # GPU 최적화: 모든 데이터를 GPU에서 처리하도록 device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    frame_count = 0
    for frame_idx, ((_, rgb), (_, depth), (_, instance_mask)) in enumerate(
        zip(
            read_rgb_artifacts(artifact_path.rgb_path),
            none_it(read_depth_artifacts(artifact_path.depth_path)),
            none_it(read_instance_artifacts(artifact_path.mask_path)),
        )
    ):
        if depth is None:
            continue
        
        # 메모리 효율성을 위한 프레임 수 제한
        if max_frames and frame_count >= max_frames:
            break

        if image_size is None:
            image_size = rgb.shape[:2]

        # GPU 최적화: 모든 텐서를 GPU로 이동
        rgb = rgb.to(device)
        depth = depth.to(device)
        if instance_mask is not None:
            instance_mask = instance_mask.to(device)

        # Keep only background (instance_id == 0)
        # 기존 build_background_pointcloud와 동일한 로직 사용
        if instance_mask is not None:
            # 기존 방식: 프레임별로 background만 유지하지만 전체적으로 관대하게 처리
            static_mask = (instance_mask == 0)
        else:
            # instance mask가 없으면 모든 픽셀을 유효하게 간주 (기존 방식과 동일)
            static_mask = torch.ones_like(depth, dtype=torch.bool, device=device)
        
        # Mask out invalid depth values
        depth_mask = reliable_depth_mask_range(depth)
        final_mask = static_mask & depth_mask

        # GPU 최적화: 무효한 픽셀을 NaN으로 설정하여 GPU에서 처리
        masked_depth = depth.clone().float()
        masked_depth[~final_mask] = float('nan')
        
        masked_rgb = rgb.clone().float()
        masked_rgb[~final_mask.unsqueeze(-1).expand_as(rgb)] = float('nan')

        valid_depth_frames.append(masked_depth)
        valid_rgb_frames.append(masked_rgb)
        
        frame_count += 1
        if frame_idx % 10 == 0:  # 로그 빈도 감소
            logger.info(f"Collected frame {frame_idx} for GPU batch processing.")

    if not valid_depth_frames:
        logger.warning("No valid background frames found.")
        return np.empty((0, 3)), np.empty((0, 3)), image_size

    # nanmean으로 대표값 계산
    logger.info(f"Computing nanmean across {len(valid_depth_frames)} frames...")
    
    stacked_depth = torch.stack(valid_depth_frames, dim=0)  # Shape: [N_frames, H, W]
    stacked_rgb = torch.stack(valid_rgb_frames, dim=0)      # Shape: [N_frames, H, W, 3]
    
    # nanmean 계산 (GPU 벡터화)
    mean_depth, mean_rgb = compute_robust_mean_tensors(stacked_depth, stacked_rgb)
    valid_pixel_mask = mean_depth > 0

    # --- GPU 최적화된 포인트 클라우드 프로젝션 ---
    logger.info("Projecting optimized mean depth map to a static point cloud using GPU...")
    
    # We need intrinsics and a single pose to project. We can use the first frame's pose.
    poses_iter = read_pose_artifacts(artifact_path.pose_path)[1].matrix().numpy()
    c2w = next(iter(poses_iter))
    intr, camera_type = next(iter(zip(*read_intrinsics_artifacts(artifact_path.intrinsics_path, artifact_path.camera_type_path)[1:3])))

    frame_height, frame_width = image_size
    
    # GPU 최적화: 모든 연산을 GPU에서 수행
    camera_model = camera_type.build_camera_model(intr)
    disp_v, disp_u = torch.meshgrid(
        torch.arange(frame_height, device=device).float()[::spatial_subsample],
        torch.arange(frame_width, device=device).float()[::spatial_subsample],
        indexing="ij",
    )
    if camera_type == CameraType.PANORAMA:
        disp_v = disp_v / (frame_height - 1)
        disp_u = disp_u / (frame_width - 1)
    disp = torch.ones_like(disp_v)
    
    # camera_model.iproj_disp는 CPU 텐서를 기대하므로 CPU로 변환
    disp_cpu = disp.cpu()
    disp_u_cpu = disp_u.cpu()
    disp_v_cpu = disp_v.cpu()
    
    pts, _, _ = camera_model.iproj_disp(disp_cpu, disp_u_cpu, disp_v_cpu)
    
    # 결과를 다시 GPU로 이동
    if isinstance(pts, torch.Tensor):
        pts = pts.to(device)
    else:
        pts = torch.from_numpy(pts).to(device).float()
    
    rays = pts[..., :3]  # GPU 텐서로 유지
    if camera_type != CameraType.PANORAMA:
        rays = rays / rays[..., 2:3]

    # GPU에서 포인트 클라우드 계산
    c2w_tensor = torch.from_numpy(c2w).to(device).float()
    pcd_camera = rays * mean_depth[::spatial_subsample, ::spatial_subsample].unsqueeze(-1)
    pcd_camera_flat = pcd_camera.reshape(-1, 3)  # [N, 3]
    
    # 변환 행렬 적용 (GPU에서)
    pcd_world_flat = (c2w_tensor[:3, :3] @ pcd_camera_flat.T + c2w_tensor[:3, 3:4]).T
    pcd_world = pcd_world_flat.reshape(pcd_camera.shape)

    # GPU 최적화: 마스킹 및 필터링도 GPU에서 수행
    depth_mask = reliable_depth_mask_range(mean_depth)[::spatial_subsample, ::spatial_subsample]
    
    # RGB 샘플링 (GPU에서 수행)
    sampled_rgb = (mean_rgb * 255).clamp(0, 255).to(torch.uint8)
    sampled_rgb = sampled_rgb[::spatial_subsample, ::spatial_subsample]

    # GPU에서 유효한 포인트 필터링
    depth_mask_flat = depth_mask.reshape(-1)
    valid_points_gpu = pcd_world.reshape(-1, 3)[depth_mask_flat]
    valid_colors_gpu = sampled_rgb.reshape(-1, 3)[depth_mask_flat]

    # 최종 결과만 CPU로 이동
    valid_points = valid_points_gpu.cpu().numpy()
    valid_colors = valid_colors_gpu.cpu().numpy()

    logger.info(f"Built static background point cloud with {len(valid_points)} points using nanmean method.")

    # Transform to Ego4D coordinate system (마지막에만 CPU에서 수행)
    points_homogeneous = np.hstack([valid_points, np.ones((len(valid_points), 1))])
    mean_bg_points_transformed = (T_cam_to_world @ points_homogeneous.T).T[:, :3]

    return mean_bg_points_transformed, valid_colors, image_size

def build_dynamic_points_for_frame(artifact_path: ArtifactPath, frame_idx: int,
                                   T_cam_to_world: np.ndarray,
                                   spatial_subsample: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build dynamic (non-background) point cloud for a single frame and transform to Ego4D.

    This function prioritizes SLAM map if available for the specific frame,
    otherwise falls back to depth-based reconstruction. It extracts only
    dynamic points (instance id != 0 when using depth method).

    Returns:
        dynamic_points_transformed: Mx3 array in Ego4D world coordinates
        dynamic_colors_transformed: Mx3 RGB colors
    """

    # Fallback to original depth-based method for dynamic points
    # Iterate through artifacts until we hit the desired frame
    none_it = lambda inner_it: (item for item in inner_it)

    artifact = artifact_path

    # Walk the same zipped iterators but only process the requested frame
    rays = None

    for idx, (c2w, (_, rgb), intr, camera_type, (_, depth), (_, instance_mask)) in enumerate(
        zip(
            read_pose_artifacts(artifact.pose_path)[1].matrix().numpy(),
            read_rgb_artifacts(artifact.rgb_path),
            *read_intrinsics_artifacts(artifact.intrinsics_path, artifact.camera_type_path)[1:3],
            none_it(read_depth_artifacts(artifact.depth_path)),
            none_it(read_instance_artifacts(artifact.mask_path)),
        )
    ):
        if idx != frame_idx:
            continue

        if depth is None:
            return np.empty((0, 3)), np.empty((0, 3))

        frame_height, frame_width = rgb.shape[:2]

        sampled_rgb = (rgb.cpu().numpy() * 255).astype(np.uint8)
        sampled_rgb = sampled_rgb[::spatial_subsample, ::spatial_subsample]

        # Build rays for this frame
        camera_model = camera_type.build_camera_model(intr)
        disp_v, disp_u = torch.meshgrid(
            torch.arange(frame_height).float()[::spatial_subsample],
            torch.arange(frame_width).float()[::spatial_subsample],
            indexing="ij",
        )
        if camera_type == CameraType.PANORAMA:
            disp_v = disp_v / (frame_height - 1)
            disp_u = disp_u / (frame_width - 1)
        disp = torch.ones_like(disp_v)
        pts, _, _ = camera_model.iproj_disp(disp, disp_u, disp_v)
        rays = pts[..., :3].numpy()
        if camera_type != CameraType.PANORAMA:
            rays /= rays[..., 2:3]

        # Generate point cloud in camera coordinates
        pcd_camera = rays * depth.numpy()[::spatial_subsample, ::spatial_subsample, None]
        pcd_camera_flat = pcd_camera.reshape(-1, 3)
        pcd_world_flat = (c2w[:3, :3] @ pcd_camera_flat.T + c2w[:3, 3:4]).T
        pcd_world = pcd_world_flat.reshape(pcd_camera.shape)

        # Apply depth mask
        depth_mask = reliable_depth_mask_range(depth)[::spatial_subsample, ::spatial_subsample].numpy()

        # Keep only dynamic instances (instance != 0)
        if instance_mask is None:
            return np.empty((0, 3)), np.empty((0, 3))

        instance_mask_np = instance_mask.cpu().numpy() if hasattr(instance_mask, 'cpu') else instance_mask
        dynamic_mask = (instance_mask_np != 0)
        dynamic_mask_sub = dynamic_mask[::spatial_subsample, ::spatial_subsample]
        depth_mask = depth_mask & dynamic_mask_sub

        # Flatten and filter
        pcd_flat = pcd_world.reshape(-1, 3)
        rgb_flat = sampled_rgb.reshape(-1, 3)
        mask_flat = depth_mask.reshape(-1)

        valid_points = pcd_flat[mask_flat]
        valid_colors = rgb_flat[mask_flat]

        if valid_points.size == 0:
            return np.empty((0, 3)), np.empty((0, 3))

        # Transform to Ego4D using provided T_cam_to_world
        points_homogeneous = np.hstack([valid_points, np.ones((len(valid_points), 1))])
        points_transformed = (T_cam_to_world @ points_homogeneous.T).T[:, :3]

        return points_transformed, valid_colors


def render_points_pytorch3d(points_world, colors_world, K, T_c2w=None, T_w2c=None,
                            W=640, H=480, point_size=2, device="cuda",
                            background_mode="solid", background_color=(0.0, 0.0, 0.0),
                            noise_range=(0, 255), seed=42):
    """
    Render point cloud from ego view using PyTorch3D for a single image.

    Notes:
    - We use pytorch3d.utils.cameras_from_opencv_projection to correctly convert
      OpenCV-style intrinsics/extrinsics into a PyTorch3D camera (handles axis conventions).
    - No manual Y/Z flip or extra inverses are applied.
    """

    # Lazy import here to avoid touching your global imports
    from pytorch3d.utils import cameras_from_opencv_projection

    # Resolve device
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Resolve T_w2c (world -> camera, OpenCV convention)
    if T_w2c is None:
        assert T_c2w is not None, "Provide either T_w2c or T_c2w."
        T_w2c = np.linalg.inv(T_c2w)

    # Accept (3,4) or (4,4) and extract R (3x3), t (3,)
    if T_w2c.shape == (4, 4):
        R_cv = T_w2c[:3, :3]
        t_cv = T_w2c[:3, 3]
    elif T_w2c.shape == (3, 4):
        R_cv = T_w2c[:, :3]
        t_cv = T_w2c[:, 3]
    else:
        raise ValueError(f"T_w2c must be (3,4) or (4,4), got {T_w2c.shape}")

    # Basic sanity checks
    assert np.isfinite(R_cv).all(), "R has NaN/Inf"
    assert np.isfinite(t_cv).all(), "t has NaN/Inf"
    assert np.isfinite(K).all(), "K has NaN/Inf"

    # Background (H x W x 3), float32 [0,1]
    if background_mode != "solid":
        bg_img = generate_background(W, H, background_mode, background_color, noise_range, seed)
    else:
        bg_img = np.full((H, W, 3), background_color, dtype=np.float32)

    # Points / colors to tensors (with validity filtering)
    pts_np = np.asarray(points_world, dtype=np.float32)
    cols_np = np.asarray(colors_world, dtype=np.float32)

    logger.info(f"Total points before filtering: {pts_np.shape[0]}")
    logger.info(f"Point cloud bounds: min={pts_np.min(axis=0)}, max={pts_np.max(axis=0)}")
    logger.info(f"Camera position: {t_cv}")

    finite_mask = np.isfinite(pts_np).all(axis=1)
    logger.info(f"Finite points: {finite_mask.sum()}/{len(finite_mask)}")
    if finite_mask.sum() == 0:
        logger.warning("No finite points found!")
        return (bg_img * 255).astype(np.uint8)

    pts_np = pts_np[finite_mask]
    cols_np = cols_np[finite_mask]

    # (Optional) clip out extremely large coordinates to avoid numeric issues
    mag = np.linalg.norm(pts_np, axis=1)
    keep = mag < 1e6
    logger.info(f"Points after magnitude filter: {keep.sum()}/{len(keep)}")
    pts_np = pts_np[keep]
    cols_np = cols_np[keep]
    if pts_np.shape[0] == 0:
        logger.warning("No points after magnitude filtering!")
        return (bg_img * 255).astype(np.uint8)
    
    # Check distance from camera to points
    distances = np.linalg.norm(pts_np - t_cv.reshape(1, 3), axis=1)
    logger.info(f"Distance to points: min={distances.min():.3f}, max={distances.max():.3f}, mean={distances.mean():.3f}")
    logger.info(f"Final points for rendering: {pts_np.shape[0]}")

    pts = torch.from_numpy(pts_np).to(device=device, dtype=torch.float32)
    cols = torch.from_numpy(cols_np).to(device=device, dtype=torch.float32)
    if cols.max() > 1.0:
        cols = cols / 255.0

    # Pointclouds expects lists
    point_cloud = Pointclouds(points=[pts], features=[cols])

    # Build camera from OpenCV intrinsics/extrinsics
    R_cv_t = torch.as_tensor(R_cv, dtype=torch.float32, device=device).unsqueeze(0)    # (1,3,3)
    t_cv_t = torch.as_tensor(t_cv, dtype=torch.float32, device=device).unsqueeze(0)    # (1,3)
    K_t    = torch.as_tensor(K,    dtype=torch.float32, device=device).unsqueeze(0)    # (1,3,3)
    image_size_t = torch.tensor([[H, W]], dtype=torch.int64, device=device)            # (1,2)

    camera = cameras_from_opencv_projection(
        R=R_cv_t,
        tvec=t_cv_t, 
        camera_matrix=K_t, 
        image_size=image_size_t
    ).to(device)

    # Convert pixel point_size to NDC radius
    px_to_ndc = 2.0 / max(W, H)
    radius_ndc = float(point_size) * px_to_ndc
    radius_ndc = max(1e-5, min(0.25, radius_ndc))  # safety clamp

    # Rasterizer / Renderer
    raster_settings = PointsRasterizationSettings(
        image_size=(H, W),
        radius=radius_ndc,
        points_per_pixel=20,
    )
    rasterizer = PointsRasterizer(cameras=camera, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(background_color=(0.0, 0.0, 0.0)),
    )

    # Render
    image = renderer(point_cloud)  # (1, H, W, C)
    img = image[0]                 # (H, W, C)

    # Composite with background
    if img.shape[-1] == 4:  # RGBA
        rendered_rgb = img[..., :3].clamp(0, 1).detach().cpu().numpy()
        alpha = img[..., 3].clamp(0, 1).detach().cpu().numpy()[..., None]
        final_img = bg_img * (1.0 - alpha) + rendered_rgb * alpha
    else:  # RGB
        rendered_rgb = img[..., :3].clamp(0, 1).detach().cpu().numpy()
        background_mask = np.all(rendered_rgb < 1e-2, axis=2)
        final_img = rendered_rgb.copy()
        final_img[background_mask] = bg_img[background_mask]

    # Convert to uint8
    final_img_uint8 = (np.clip(final_img, 0.0, 1.0) * 255).astype(np.uint8)
    
    # Apply simple 90-degree rotation to fix image orientation
    final_img_rotated = cv2.rotate(final_img_uint8, cv2.ROTATE_90_CLOCKWISE)

    return final_img_rotated

def render_points_fisheye(points_world, colors_world, T_w2c, ego_intrinsics, W=640, H=480, 
                         point_size=2, device="cuda",
                         radial_distortion_coeffs=None,
                         tangential_distortion_coeffs=None,
                         thinPrism_distortion_coeffs=None,
                         focal_length=None, principal_point=None,
                         original_image_size=None,
                         background_mode="solid", background_color=(0.0, 0.0, 0.0),
                         noise_range=(0, 255), seed=42):
    """
    Render point cloud with fish-eye view using PyTorch3D FishEyeCameras.
    
    Args:
        points_world: Nx3 array of 3D points in world coordinates
        colors_world: Nx3 array of RGB colors
        T_w2c: 4x4 world-to-camera transformation matrix
        ego_intrinsics: 3x3 ego camera intrinsics matrix  
        W, H: Image dimensions
        point_size: Size of rendered points
        device: Device for rendering
        distortion_coeffs: Fish-eye distortion coefficients
        focal_length: (f) focal lengths. If None, extracts from ego_intrinsics
        principal_point: (cx, cy) principal point. If None, extracts from ego_intrinsics
        original_image_size: (H, W) of the original camera sensor, for scaling intrinsics
        background_mode: Background generation mode
        background_color: Background color
        noise_range: Noise range for background
        seed: Random seed
    
    Note:
        Uses default fish-eye distortion coefficients [k1=-0.2, k2=0.1, k3=0.0, k4=0.0, k5=0.0, k6=0.0]
        to create fish-eye effect. For real fish-eye cameras, specific distortion 
        coefficients should be calibrated and provided.
    
    Returns:
        Rendered image as (H, W, 3) uint8 array
    """
    # Resolve device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Extract rotation and translation from T_w2c
    if T_w2c.shape == (4, 4):
        R_cv = T_w2c[:3, :3]
        t_cv = T_w2c[:3, 3]
    elif T_w2c.shape == (3, 4):
        R_cv = T_w2c[:, :3]
        t_cv = T_w2c[:, 3]
    else:
        raise ValueError(f"T_w2c must be (3,4) or (4,4), got {T_w2c.shape}")
    
    # Basic sanity checks
    assert np.isfinite(R_cv).all(), "R has NaN/Inf"
    assert np.isfinite(t_cv).all(), "t has NaN/Inf"
    
    # Background (H x W x 3), float32 [0,1]
    if background_mode != "solid":
        bg_img = generate_background(W, H, background_mode, background_color, noise_range, seed)
    else:
        bg_img = np.full((H, W, 3), background_color, dtype=np.float32)
    
    # Points / colors to tensors (with validity filtering)
    pts_np = np.asarray(points_world, dtype=np.float32)
    cols_np = np.asarray(colors_world, dtype=np.float32)
    
    logger.info(f"Fish-eye rendering - Total points before filtering: {pts_np.shape[0]}")
    logger.info("Using FishEyeCameras for authentic fish-eye distortion")
    
    finite_mask = np.isfinite(pts_np).all(axis=1)
    if finite_mask.sum() == 0:
        logger.warning("No finite points found for fish-eye rendering!")
        return (bg_img * 255).astype(np.uint8)
    
    pts_np = pts_np[finite_mask]
    cols_np = cols_np[finite_mask]
    
    # Filter out extremely large coordinates
    mag = np.linalg.norm(pts_np, axis=1)
    keep = mag < 1e6
    pts_np = pts_np[keep]
    cols_np = cols_np[keep]
    if pts_np.shape[0] == 0:
        logger.warning("No points after magnitude filtering for fish-eye!")
        return (bg_img * 255).astype(np.uint8)
    
    pts = torch.from_numpy(pts_np).to(device=device, dtype=torch.float32)
    cols = torch.from_numpy(cols_np).to(device=device, dtype=torch.float32)
    if cols.max() > 1.0:
        cols = cols / 255.0
    
    logger.info(f"Fish-eye camera point statistics after filtering:")
    logger.info(f"  Distance to points: min={np.linalg.norm(pts_np - t_cv.reshape(1, 3), axis=1).min():.3f}, max={np.linalg.norm(pts_np - t_cv.reshape(1, 3), axis=1).max():.3f}")
    logger.info(f"  Points after coordinate transform: {pts_np.shape[0]}")
    
    # Create Pointclouds
    point_cloud = Pointclouds(points=[pts], features=[cols])
    
    # Flip Y and Z axes to convert from OpenCV to PyTorch3D
    # This is the same transformation that cameras_from_opencv_projection does internally
    coord_transform_between_opencv_and_pytorch = np.array([
        [-1,  0,  0],
        [ 0, -1,  0], 
        [ 0,  0,  1]
    ], dtype=np.float32)
    coord_transform_for_aria_cam = np.array([
        [ 0,  1,  0],
        [-1,  0,  0], 
        [ 0,  0,  1]
    ], dtype=np.float32) # coord ccw 90 rotate

    # Apply coordinate transformation to rotation and translation
    coord_transform_total = coord_transform_between_opencv_and_pytorch @ coord_transform_for_aria_cam
    R_pt3d = R_cv.T @ coord_transform_total
    t_pt3d = coord_transform_total.T @ t_cv

    R_t = torch.from_numpy(R_pt3d).to(device=device, dtype=torch.float32).unsqueeze(0)  # (1,3,3)
    T_t = torch.from_numpy(t_pt3d).to(device=device, dtype=torch.float32).unsqueeze(0)  # (1,3)
    
    # FishEyeCameras requires radial distortion parameters
    if radial_distortion_coeffs is None or len(radial_distortion_coeffs) == 0:
        raise ValueError("radial_distortion_coeffs must be provided")
    if tangential_distortion_coeffs is None or len(tangential_distortion_coeffs) == 0:
        raise ValueError("tangential_distortion_coeffs must be provided")
    if thinPrism_distortion_coeffs is None or len(thinPrism_distortion_coeffs) == 0:
        raise ValueError("thinPrism_distortion_coeffs must be provided")
    
    radial_distortion = torch.tensor([radial_distortion_coeffs], device=device, dtype=torch.float32)
    tangential_distortion = torch.tensor([tangential_distortion_coeffs], device=device, dtype=torch.float32)
    thinPrism_distortion = torch.tensor([thinPrism_distortion_coeffs], device=device, dtype=torch.float32)

    # Use focal_length and principal_point from online_calibration if available
    if focal_length is not None and principal_point is not None:
        f = focal_length[0]
        cx, cy = principal_point[0], principal_point[1]
        logger.info(f"Using intrinsics from online_calibration: f={f:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        
        # Scale intrinsics if original image size is provided
        if original_image_size:
            f, cx, cy = scale_intrinsics(f, cx, cy, original_image_size, (H, W))
        else:
            logger.warning("original_image_size not provided, intrinsics may be incorrect for the target resolution.")
    else:
        # Fallback to ego_intrinsics if online calibration data is not provided
        f = ego_intrinsics[0,0]#, ego_intrinsics[1,1]
        cx, cy = ego_intrinsics[0,2], ego_intrinsics[1,2]
        logger.info(f"Using intrinsics from ego_intrinsics: f={f:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    def _pix_to_ndc(f, cx, cy, W, H):
        f_ndc = 2.0 * f / W # W = H
        cx_ndc = 2.0 * (cx / W) - 1.0
        cy_ndc = 1.0 - 2.0 * (cy / H)
        return f_ndc, cx_ndc, cy_ndc

    f_ndc, cx_ndc, cy_ndc = _pix_to_ndc(f, cx, cy, W, H) # FishEyeCameras.in_ndc == True

    logger.info(f"OpenCV principal point: cx={cx:.1f}, cy={cy:.1f}")
    logger.info(f"PyTorch3D principal point(ndc): cx={cx_ndc:.1f}, cy={cy_ndc:.1f}")
    
    focal_length_tensor = torch.tensor([f_ndc], device=device, dtype=torch.float32)
    principal_point_tensor = torch.tensor([[cx_ndc, cy_ndc]], device=device, dtype=torch.float32)
    
    camera = FishEyeCameras(
        device=device,
        R=R_t,
        T=T_t,
        radial_params=radial_distortion,
        tangential_params=tangential_distortion,
        thin_prism_params=thinPrism_distortion,
        focal_length=focal_length_tensor,
        principal_point=principal_point_tensor,
        world_coordinates=True # default: False
    )
    
    # Convert pixel point_size to appropriate radius
    px_to_ndc = 2.0 / max(W, H)
    radius_ndc = float(point_size) * px_to_ndc
    radius_ndc = max(1e-5, min(0.25, radius_ndc))  # safety clamp
    
    # Rasterizer / Renderer
    raster_settings = PointsRasterizationSettings(
        image_size=(H, W),
        radius=radius_ndc,
        points_per_pixel=20,
    )
    rasterizer = PointsRasterizer(cameras=camera, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(background_color=(0.0, 0.0, 0.0)),
    )
    
    # Render
    image = renderer(point_cloud)  # (1, H, W, C)
    img = image[0]                 # (H, W, C)
    
    # Composite with background
    if img.shape[-1] == 4:  # RGBA
        rendered_rgb = img[..., :3].clamp(0, 1).detach().cpu().numpy()
        alpha = img[..., 3].clamp(0, 1).detach().cpu().numpy()[..., None]
        final_img = bg_img * (1.0 - alpha) + rendered_rgb * alpha
    else:  # RGB
        rendered_rgb = img[..., :3].clamp(0, 1).detach().cpu().numpy()
        background_mask = np.all(rendered_rgb < 1e-2, axis=2)
        final_img = rendered_rgb.copy()
        final_img[background_mask] = bg_img[background_mask]
    
    # Convert to uint8
    final_img_uint8 = (np.clip(final_img, 0.0, 1.0) * 255).astype(np.uint8)
    
    # Apply simple 90-degree rotation to fix image orientation
    # final_img_rotated = cv2.rotate(final_img_uint8, cv2.ROTATE_90_CLOCKWISE)

    return final_img_uint8 #final_img_rotated

def project_points_to_image_sequential(bg_points_3d: np.ndarray, bg_colors: np.ndarray,
                                      ego_extrinsics_list: List[np.ndarray], ego_intrinsics: np.ndarray,
                                      image_size: Tuple[int, int], point_size: float = 1.0,
                                      use_fisheye: bool = False,
                                      online_calibration_path: str = None,
                                      original_image_size: Optional[Tuple[int, int]] = None,
                                      artifact_path: Optional[ArtifactPath] = None,
                                      T_cam_to_world: Optional[np.ndarray] = None,
                                      only_bg: bool = False) -> List[np.ndarray]:
    """
    Project 3D points to 2D images using ego camera poses with individual rendering.
    
    Args:
        points_3d: Nx3 array of 3D points in world coordinates
        colors: Nx3 array of RGB colors
        ego_extrinsics_list: List of 3x4 ego camera extrinsics matrices (world to camera)
        ego_intrinsics: 3x3 ego camera intrinsics matrix
        image_size: (height, width) of output images
        point_size: Size of rendered points
        use_fisheye: Whether to use fish-eye rendering (180-degree FOV)
        online_calibration_path: Path to online_calibration.jsonl for real distortion coeffs
        original_image_size: (H, W) of the original ego camera(Project Aria in Ego-Exo4D), for scaling intrinsics
    
    Returns:
        List of rendered images as HxWx3 arrays
    """
    height, width = image_size
    num_frames = len(ego_extrinsics_list)
    
    if len(bg_points_3d) == 0 and artifact_path is None:
        return [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(num_frames)]
    
    rendered_images = []
    
    # Load Aria distortion coefficients if using fish-eye
    # load_aria_distortion_coeffs will return defaults if online_calibration_path is None
    aria_radial_distortion = None
    aria_tan_distortion = None
    aria_thin_distortion = None
    aria_focal_length = None
    aria_principal_point = None
    if original_image_size is not None:
        aria_original_size = original_image_size
    else:
        # Fallback to Aria standard resolution
        aria_original_size = (2880, 2880)
        logger.info("Using Aria standard resolution (2880x2880) as fallback for original_image_size")
    
    if use_fisheye:
        try:
            (   aria_radial_distortion,
                aria_tan_distortion,
                aria_thin_distortion,
                aria_focal_length,
                aria_principal_point
            ) = load_aria_distortion_coeffs(
                online_calibration_path,
                frame_idx=0
            )
            
            if (aria_radial_distortion is not None and 
                aria_tan_distortion is not None and 
                aria_thin_distortion is not None):
                if online_calibration_path:
                    logger.info("Successfully loaded Aria camera distortion coefficients from online_calibration.jsonl")
                else:
                    logger.info("Using default Ego-Exo4D distortion coefficients")
            else:
                raise ValueError("Failed to load distortion coefficients.")
        except Exception as e:
            logger.error(f"Failed to load distortion coefficients: {e}")
            raise
    
    # Process each frame individually, building dynamic points per-frame and concatenating with background
    for frame_idx, ego_extrinsic in enumerate(tqdm(ego_extrinsics_list, desc="Rendering frames")):
        # Build dynamic points for this frame if artifact_path is provided
        # If only_bg is True, skip building dynamic points to speed up rendering
        if (not only_bg) and artifact_path is not None and T_cam_to_world is not None:
            dyn_points, dyn_colors = build_dynamic_points_for_frame(artifact_path, frame_idx, T_cam_to_world)
        else:
            dyn_points, dyn_colors = np.empty((0, 3)), np.empty((0, 3))

        # Concatenate background + dynamic
        if dyn_points.size == 0:
            points_to_render = bg_points_3d
            colors_to_render = bg_colors
        else:
            points_to_render = np.vstack([bg_points_3d, dyn_points])
            colors_to_render = np.vstack([bg_colors, dyn_colors])

        # ego_extrinsics is already W2C format in 4x4, use directly as transformation matrix
        T_w2c = ego_extrinsic

        # Choose rendering function based on fish-eye option
        if use_fisheye:
            rendered_image = render_points_fisheye(
                points_world=points_to_render,
                colors_world=colors_to_render,
                T_w2c=T_w2c,
                ego_intrinsics=ego_intrinsics,
                W=width,
                H=height,
                point_size=point_size,
                device="cuda" if torch.cuda.is_available() else "cpu",
                radial_distortion_coeffs=aria_radial_distortion,
                tangential_distortion_coeffs=aria_tan_distortion,
                thinPrism_distortion_coeffs=aria_thin_distortion,
                focal_length=aria_focal_length,
                principal_point=aria_principal_point,
                original_image_size=aria_original_size, # Pass the correct original size
                background_mode="solid",
                background_color=(0.0, 0.0, 0.0)
            )
        else:
            # Regular perspective rendering
            rendered_image = render_points_pytorch3d(
                points_world=points_to_render,
                colors_world=colors_to_render,
                K=ego_intrinsics,
                T_w2c=T_w2c,
                W=width,
                H=height,
                point_size=point_size,
                device="cuda" if torch.cuda.is_available() else "cpu",
                background_mode="solid",
                background_color=(0.0, 0.0, 0.0)
            )

        rendered_images.append(rendered_image)

        if frame_idx % 10 == 0:
            logger.info(f"Rendered frame {frame_idx+1}/{num_frames}")
    
    return rendered_images

def get_parser():
    parser = argparse.ArgumentParser(description="Render ViPE point cloud from ego view")
    parser.add_argument("--input_dir", required=True, help="Directory containing ViPE artifacts")
    parser.add_argument("--out_dir", required=True, help="Output directory for rendered images")
    parser.add_argument("--meta_json_path", required=True, help="Path to meta.json file containing camera parameters")
    parser.add_argument("--point_size", type=float, default=1.0, help="Size of rendered points")
    parser.add_argument("--start_frame", type=int, default=0, help="Starting frame number for rendering (default: 0)")
    parser.add_argument("--end_frame", type=int, required=True, help="Ending frame number for rendering (inclusive)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for rendering (default: 8)")
    parser.add_argument("--only_bg", action="store_true", help="Only render background points")
    parser.add_argument("--use_mean_bg", action="store_true", help="Use nanmean background instead of standard background")
    parser.add_argument("--fish_eye_rendering", action="store_true", help="Enable fish-eye rendering with 360-degree view")
    parser.add_argument("--online_calibration_path", type=str, default=None, help="(Optional) Path to online_calibration.jsonl file for real Aria distortion coefficients. If not provided, uses default Ego-Exo4D fisheye distortion coefficients.")

    return parser

def configure_output_directory(args) -> str:
    """
    Configure and construct the output directory path based on input arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        str: Final configured output directory path
    """
    # Extract video name from input_dir (e.g., 'vipe_results/joker' -> 'joker')
    try:
        input_path = Path(args.input_dir)
        video_name = input_path.stem  # Get the last component without extension
        
        # Final output: <provided_out_dir>/<video_name>
        output_dir = os.path.join(args.out_dir, video_name)
        logger.info(f"Derived output directory: {output_dir}")
    except Exception as e:
        logger.warning(f"Failed to derive output dir from input_dir={getattr(args, 'input_dir', None)}: {e}")
        output_dir = args.out_dir

    return output_dir

def get_inference_frame_range(input_dir: str) -> tuple[int, int]:
    """
    Get the frame range from ViPE inference results.
    Returns (start_frame, end_frame_inclusive) based on available pose data.
    """
    try:
        # Find artifact paths in the input directory
        artifact_paths = list(ArtifactPath.glob_artifacts(Path(input_dir), use_video=True))
        if not artifact_paths:
            raise ValueError(f"No VIPE artifacts found in {input_dir}")
        
        artifact_path = artifact_paths[0]
        
        # Load pose data to determine available frame range
        if not artifact_path.pose_path.exists():
            raise FileNotFoundError(f"Pose file not found: {artifact_path.pose_path}")
        
        pose_inds, _ = read_pose_artifacts(artifact_path.pose_path)
        
        if len(pose_inds) == 0:
            raise ValueError("No pose data found in inference results")
        
        min_frame = int(pose_inds.min())
        max_frame = int(pose_inds.max())  # Keep as inclusive end
        
        logger.info(f"Inference results contain frames {min_frame} to {max_frame} ({len(pose_inds)} frames)")
        
        return min_frame, max_frame
        
    except Exception as e:
        logger.error(f"Failed to determine inference frame range: {e}")
        raise

def validate_frame_range(args, total_frames_available: int, input_dir: str):
    """
    Validate that frame range arguments exactly match the inference results range.
    
    Args:
        args: Parsed command line arguments with start_frame and end_frame
        total_frames_available: Total number of frames available from ego extrinsics
        input_dir: Input directory containing inference results
        
    Raises:
        ValueError: If frame range doesn't exactly match inference results or is invalid
    """
    # Get inference frame range for validation
    inference_start, inference_end = get_inference_frame_range(input_dir)
    
    # Validate start_frame and end_frame
    if args.start_frame < 0:
        raise ValueError(f"start_frame must be non-negative, got {args.start_frame}")
    
    if args.end_frame <= args.start_frame:
        raise ValueError(f"end_frame ({args.end_frame}) must be greater than start_frame ({args.start_frame})")
    
    # Check exact match with inference range
    if args.start_frame != inference_start:
        raise ValueError(f"start_frame ({args.start_frame}) must match inference range start ({inference_start})")
    
    if args.end_frame != inference_end:
        raise ValueError(f"end_frame ({args.end_frame}) must match inference range end ({inference_end})")
    
    if args.start_frame >= total_frames_available:
        raise ValueError(f"start_frame ({args.start_frame}) exceeds available frames ({total_frames_available})")
    
    if args.end_frame >= total_frames_available:
        raise ValueError(f"end_frame ({args.end_frame}) exceeds available frames ({total_frames_available})")
    
    num_frames_to_render = args.end_frame - args.start_frame + 1
    logger.info(f"Frame range validation passed. Rendering frames [{args.start_frame}, {args.end_frame}] ({num_frames_to_render} frames)")

def main():
    parser = get_parser()
    
    args = parser.parse_args()

    # Configure output directory
    args.out_dir = configure_output_directory(args)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load camera parameters from meta.json
    logger.info(f"Loading camera parameters from {args.meta_json_path}")
    exo_intrinsic, exo_extrinsic, ego_intrinsic, ego_extrinsics_list = load_camera_params_from_meta(
        args.meta_json_path, args.input_dir
    )
    
    # Convert exo_extrinsic (4x4 world to cam) to T_cam_to_world (4x4 cam to world)
    T_cam_to_world = np.linalg.inv(exo_extrinsic)
    
    # Choose background building method based on use_mean_bg flag
    if getattr(args, 'use_mean_bg', False):
        logger.info(f"Building NANMEAN background point cloud from {args.input_dir}")
        global_points_bg, global_colors_bg, image_size = build_mean_background_pointcloud(
            args.input_dir, T_cam_to_world
        )
    else:
        logger.info(f"Building standard background point cloud from {args.input_dir}")
        global_points_bg, global_colors_bg, image_size = build_background_pointcloud(
            args.input_dir, T_cam_to_world
        )
    
    if len(global_points_bg) == 0:
        logger.error("No points in background point cloud. Exiting.")
        return
    
    logger.info(f"Original video resolution: {image_size[0]} x {image_size[1]} (H x W)")
    
    # Use fixed resolution for rendering
    fixed_image_size = (448, 448)  # (height, width)
    logger.info(f"Using fixed rendering resolution: {fixed_image_size[0]} x {fixed_image_size[1]} (H x W)")
    
    # Validate frame range using dedicated function
    validate_frame_range(args, len(ego_extrinsics_list), args.input_dir)
    
    # Check if fish-eye rendering is enabled
    fish_eye_enabled = getattr(args, 'fish_eye_rendering', False)
    
    # Get online calibration path
    online_calib_path = getattr(args, 'online_calibration_path', None)
    
    num_frames_to_render = args.end_frame - args.start_frame + 1
    render_mode = "fish-eye" if fish_eye_enabled else "perspective"
    logger.info(f"Rendering {num_frames_to_render} frames sequentially with {render_mode} mode")
    
    if fish_eye_enabled and online_calib_path:
        logger.info(f"Using real Aria distortion coefficients from {online_calib_path}")
    elif fish_eye_enabled:
        logger.info("Using default fish-eye distortion coefficients")
    
    # Select extrinsics for the frames to render
    extrinsics_start = args.start_frame
    extrinsics_end = args.end_frame
    logger.info(f"Using frame range for extrinsics: {extrinsics_start} to {extrinsics_end}")
    
    ego_extrinsics_to_render = ego_extrinsics_list[extrinsics_start:extrinsics_end + 1]  # args.end_frame(enclusive) + 1
    # Find artifact path for dynamic per-frame construction
    artifact_paths = list(ArtifactPath.glob_artifacts(Path(args.input_dir), use_video=True))
    artifact_path = artifact_paths[0] if artifact_paths else None

    # Render all frames using sequential processing; per-frame dynamic points are built inside
    rendered_images = project_points_to_image_sequential(
        global_points_bg, global_colors_bg, ego_extrinsics_to_render, ego_intrinsic,
        fixed_image_size, args.point_size, use_fisheye=fish_eye_enabled,
        online_calibration_path=online_calib_path,
        original_image_size=(2880, 2880), # Ego-Exo4D Ego view resolution
        artifact_path=artifact_path,
        T_cam_to_world=T_cam_to_world,
        only_bg=args.only_bg
    )

    # Save images returned by the renderer as MP4 video
    import imageio
    
    # Extract video name from input_dir (e.g., vipe_results/YOUR_VIPE_RESULT -> YOUR_VIPE_RESULT)
    # Use the configured output directory (already includes video_name)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Save as {out_dir}/ego_Prior.mp4
    output_video_path = os.path.join(args.out_dir, "ego_Prior.mp4")
    
    # Save as MP4 with 30 FPS
    logger.info(f"Saving rendered frames as MP4 video at 30 FPS: {output_video_path}")
    
    with imageio.get_writer(output_video_path, fps=30, codec='libx264', quality=8, pixelformat='yuv420p') as writer:
        for relative_idx, rendered_image in enumerate(rendered_images):
            actual_frame_idx = args.start_frame + relative_idx
            writer.append_data(rendered_image)
            
            if relative_idx % 10 == 0:
                logger.info(f"Processing frame {actual_frame_idx} (relative idx: {relative_idx})")
    
    logger.info(f"Rendering complete. Saved {num_frames_to_render} frames to {output_video_path}")
    logger.info(f"Used sequential processing for stability with {render_mode} rendering")

if __name__ == "__main__":
    main()
