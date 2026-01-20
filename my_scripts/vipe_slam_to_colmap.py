#!/usr/bin/env python3
"""
Convert ViPE SLAM results to COLMAP format.
Saves both text and binary formats, supports multiple image sizes.
"""

import argparse
import collections
import logging
import struct
import shutil
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# COLMAP Data Structures
# ============================================================================

Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

CAMERA_MODEL_IDS = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
}


# ============================================================================
# Binary I/O Helpers
# ============================================================================

def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """Pack and write data to a binary file."""
    if isinstance(data, (list, tuple)):
        fid.write(struct.pack(endian_character + format_char_sequence, *data))
    else:
        fid.write(struct.pack(endian_character + format_char_sequence, data))


# ============================================================================
# COLMAP Writers - Text Format
# ============================================================================

def write_cameras_text(cameras: dict, path: Path):
    """Write COLMAP cameras.txt file."""
    header = (
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"# Number of cameras: {len(cameras)}\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for _, cam in cameras.items():
            params_str = " ".join(f"{p:.10f}" for p in cam.params)
            f.write(f"{cam.id} {cam.model} {cam.width} {cam.height} {params_str}\n")
    logger.info(f"Written cameras.txt with {len(cameras)} cameras")


def write_images_text(images: dict, path: Path):
    """Write COLMAP images.txt file."""
    if len(images) == 0:
        mean_observations = 0
    else:
        mean_observations = sum(len(img.point3D_ids) for img in images.values()) / len(images)
    
    header = (
        "# Image list with two lines of data per image:\n"
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        f"# Number of images: {len(images)}, mean observations per image: {mean_observations:.2f}\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for _, img in images.items():
            qw, qx, qy, qz = img.qvec
            tx, ty, tz = img.tvec
            f.write(f"{img.id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {tx:.9f} {ty:.9f} {tz:.9f} {img.camera_id} {img.name}\n")
            # Write 2D points
            points_str = " ".join(f"{xy[0]:.2f} {xy[1]:.2f} {p3d_id}" for xy, p3d_id in zip(img.xys, img.point3D_ids))
            f.write(points_str + "\n")
    logger.info(f"Written images.txt with {len(images)} images")


def write_points3D_text(points3D: dict, path: Path):
    """Write COLMAP points3D.txt file."""
    if len(points3D) == 0:
        mean_track_length = 0
    else:
        mean_track_length = sum(len(pt.image_ids) for pt in points3D.values()) / len(points3D)
    
    header = (
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        f"# Number of points: {len(points3D)}, mean track length: {mean_track_length:.4f}\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for _, pt in points3D.items():
            x, y, z = pt.xyz
            r, g, b = pt.rgb
            track_str = " ".join(f"{img_id} {p2d_idx}" for img_id, p2d_idx in zip(pt.image_ids, pt.point2D_idxs))
            f.write(f"{pt.id} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {pt.error:.6f} {track_str}\n")
    logger.info(f"Written points3D.txt with {len(points3D)} points")


# ============================================================================
# COLMAP Writers - Binary Format
# ============================================================================

def write_cameras_binary(cameras: dict, path: Path):
    """Write COLMAP cameras.bin file."""
    with open(path, "wb") as f:
        write_next_bytes(f, len(cameras), "Q")
        for _, cam in cameras.items():
            model_id = CAMERA_MODEL_IDS[cam.model]
            write_next_bytes(f, [cam.id, model_id, cam.width, cam.height], "iiQQ")
            for p in cam.params:
                write_next_bytes(f, float(p), "d")
    logger.info(f"Written cameras.bin with {len(cameras)} cameras")


def write_images_binary(images: dict, path: Path):
    """Write COLMAP images.bin file."""
    with open(path, "wb") as f:
        write_next_bytes(f, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(f, img.id, "i")
            write_next_bytes(f, img.qvec.tolist(), "dddd")
            write_next_bytes(f, img.tvec.tolist(), "ddd")
            write_next_bytes(f, img.camera_id, "i")
            # Write image name as null-terminated string
            for char in img.name:
                write_next_bytes(f, char.encode("utf-8"), "c")
            write_next_bytes(f, b"\x00", "c")
            # Write 2D points
            write_next_bytes(f, len(img.point3D_ids), "Q")
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(f, [xy[0], xy[1], p3d_id], "ddq")
    logger.info(f"Written images.bin with {len(images)} images")


def write_points3D_binary(points3D: dict, path: Path):
    """Write COLMAP points3D.bin file."""
    with open(path, "wb") as f:
        write_next_bytes(f, len(points3D), "Q")
        for _, pt in points3D.items():
            write_next_bytes(f, pt.id, "Q")
            write_next_bytes(f, pt.xyz.tolist(), "ddd")
            write_next_bytes(f, pt.rgb.tolist(), "BBB")
            write_next_bytes(f, pt.error, "d")
            track_length = len(pt.image_ids)
            write_next_bytes(f, track_length, "Q")
            for img_id, p2d_idx in zip(pt.image_ids, pt.point2D_idxs):
                write_next_bytes(f, [int(img_id), int(p2d_idx)], "ii")
    logger.info(f"Written points3D.bin with {len(points3D)} points")


def write_colmap_model(cameras: dict, images: dict, points3D: dict, output_dir: Path):
    """Write COLMAP model in both text and binary formats."""
    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    # Write text format
    write_cameras_text(cameras, sparse_dir / "cameras.txt")
    write_images_text(images, sparse_dir / "images.txt")
    write_points3D_text(points3D, sparse_dir / "points3D.txt")
    
    # Write binary format
    write_cameras_binary(cameras, sparse_dir / "cameras.bin")
    write_images_binary(images, sparse_dir / "images.bin")
    write_points3D_binary(points3D, sparse_dir / "points3D.bin")


# ============================================================================
# Pose Conversion Utilities
# ============================================================================

def c2w_to_colmap_qvec_tvec(c2w_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert camera-to-world matrix to COLMAP quaternion and translation.
    COLMAP uses world-to-camera transformation.
    Returns quaternion as (qw, qx, qy, qz) and translation vector.
    """
    # VIPE stores c2w (camera-to-world) matrices
    # COLMAP needs w2c (world-to-camera) matrices
    w2c = np.linalg.inv(c2w_matrix)
    rotation = Rotation.from_matrix(w2c[:3, :3])
    quat_xyzw = rotation.as_quat()  # scipy returns (x, y, z, w)
    qvec = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # COLMAP wants (w, x, y, z)
    tvec = w2c[:3, 3]
    return qvec, tvec


# ============================================================================
# SLAM Map Loading
# ============================================================================

def load_slam_map(slam_map_path: Path, device: torch.device = torch.device("cpu")) -> dict:
    """Load SLAM map from .pt file."""
    logger.info(f"Loading SLAM map from {slam_map_path}")
    data = torch.load(slam_map_path, map_location=device)
    logger.info(f"SLAM map keys: {list(data.keys())}")
    return data


def get_slam_point_cloud(slam_data: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Extract point cloud (xyz, rgb) from SLAM map data."""
    xyz = slam_data["dense_disp_xyz"].numpy()
    rgb = slam_data["dense_disp_rgb"].numpy()
    
    # RGB is in [0, 1], convert to [0, 255]
    rgb = (rgb * 255).astype(np.uint8)
    logger.info(f"Loaded {len(xyz)} points from SLAM map")
    return xyz, rgb


# ============================================================================
# Image Directory Handling
# ============================================================================

def copy_images_for_colmap(
    images_dir: Path,
    output_dir: Path,
) -> tuple[int, int]:
    """
    Copy images from source directory to COLMAP structure.
    Images should already be at the desired resolution.
    Returns (width, height) of images.
    """
    logger.info(f"Setting up images from {images_dir}")
    
    # Use images_dir directly
    src_dir = images_dir
    
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
    
    # Destination directory (always 'images')
    dst_dir = output_dir / "images"
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = sorted(src_dir.glob("*.jpg")) + sorted(src_dir.glob("*.png"))
    if not image_files:
        raise FileNotFoundError(f"No images found in {src_dir}")
    
    # Get dimensions from first image
    from PIL import Image
    first_img = Image.open(image_files[0])
    width, height = first_img.width, first_img.height
    
    # Copy images
    copied_count = 0
    for img_path in image_files:
        dst_path = dst_dir / img_path.name
        if not dst_path.exists():
            shutil.copy2(img_path, dst_path)
            copied_count += 1
    
    logger.info(f"Copied {copied_count} images ({width}x{height}) to {dst_dir}")
    
    return width, height


# ============================================================================
# COLMAP Model Building
# ============================================================================

def scale_intrinsics(
    intrinsics_data: np.ndarray,
    ratio_x: float,
    ratio_y: float,
) -> np.ndarray:
    """Scale intrinsics for resized images (fx, fy, cx, cy)."""
    scaled = intrinsics_data.copy()
    scaled[:, 0] *= ratio_x  # fx
    scaled[:, 1] *= ratio_y  # fy
    scaled[:, 2] *= ratio_x  # cx
    scaled[:, 3] *= ratio_y  # cy
    return scaled


def build_cameras(
    intrinsics_data: np.ndarray,
    width: int,
    height: int,
) -> dict:
    """
    Build COLMAP cameras dict from intrinsics.
    Assumes PINHOLE camera model with fx, fy, cx, cy.
    """
    # Use first frame intrinsics (assuming constant)
    fx, fy, cx, cy = intrinsics_data[0]
    
    cameras = {
        1: Camera(
            id=1,
            model="PINHOLE",
            width=width,
            height=height,
            params=np.array([fx, fy, cx, cy]),
        )
    }
    return cameras


def build_images(
    pose_data: np.ndarray,
    pose_indices: np.ndarray,
    images_dir: Path,
) -> dict:
    """Build COLMAP images dict from poses and match existing image filenames."""
    images = {}

    scale_dir = images_dir / "images"

    for i, (pose_matrix, frame_idx) in enumerate(zip(pose_data, pose_indices)):
        qvec, tvec = c2w_to_colmap_qvec_tvec(pose_matrix)

        image_id = i + 1
        # Try to match existing filenames
        candidates = [
            f"frame_{frame_idx:06d}.jpg",
            f"{frame_idx:05d}.jpg",
            f"{frame_idx:06d}.jpg",
        ]
        image_name = candidates[0]
        for cand in candidates:
            if (scale_dir / cand).exists():
                image_name = cand
                break

        images[image_id] = Image(
            id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=1,
            name=image_name,
            xys=np.zeros((0, 2)),  # No 2D points
            point3D_ids=np.array([], dtype=np.int64),
        )

    return images


def build_points3D_from_slam(
    slam_data: dict,
    subsample: int = 1,
) -> dict:
    """Build COLMAP points3D dict from SLAM map."""
    xyz, rgb = get_slam_point_cloud(slam_data)
    
    # Subsample points if requested
    if subsample > 1:
        indices = np.arange(0, len(xyz), subsample)
        xyz = xyz[indices]
        rgb = rgb[indices]
        logger.info(f"Subsampled to {len(xyz)} points")
    
    points3D = {}
    for i in range(len(xyz)):
        point_id = i + 1
        points3D[point_id] = Point3D(
            id=point_id,
            xyz=xyz[i],
            rgb=rgb[i],
            error=0.0,
            image_ids=np.array([], dtype=np.int64),
            point2D_idxs=np.array([], dtype=np.int64),
        )
    
    return points3D


# ============================================================================
# Main Conversion Function
# ============================================================================

def convert_vipe_slam_to_colmap(
    vipe_path: Path,
    images_dir: Path,
    output_path: Path,
    sequence_name: str = None,
    point_subsample: int = 1,
):
    """
    Convert ViPE SLAM results to COLMAP format.
    
    Args:
        vipe_path: Path to ViPE output directory (e.g., output/zavod_1920_1)
        images_dir: Path to directory containing images at desired resolution
        output_path: Output directory for COLMAP format
        sequence_name: Name of the sequence (auto-detected if None)
        point_subsample: Subsample factor for 3D points
    """
    # Auto-detect sequence name from pose files
    if sequence_name is None:
        pose_files = list((vipe_path / "pose").glob("*.npz"))
        if not pose_files:
            raise FileNotFoundError(f"No pose files found in {vipe_path / 'pose'}")
        sequence_name = pose_files[0].stem
    
    logger.info(f"Converting sequence: {sequence_name}")
    
    # Define paths
    pose_path = vipe_path / "pose" / f"{sequence_name}.npz"
    intrinsics_path = vipe_path / "intrinsics" / f"{sequence_name}.npz"
    slam_map_path = vipe_path / "vipe" / f"{sequence_name}_slam_map.pt"
    
    # Verify files exist
    for p in [pose_path, intrinsics_path, slam_map_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Load data
    logger.info("Loading pose data...")
    pose_npz = np.load(pose_path)
    pose_data = pose_npz["data"]  # (N, 4, 4)
    pose_indices = pose_npz["inds"]
    
    # Log camera positions for debugging
    cam_positions = pose_data[:, :3, 3]
    logger.info(f"Camera positions: {len(cam_positions)} cameras")
    logger.info(f"  X range: {cam_positions[:, 0].min():.2f} to {cam_positions[:, 0].max():.2f}")
    logger.info(f"  Y range: {cam_positions[:, 1].min():.2f} to {cam_positions[:, 1].max():.2f}")
    logger.info(f"  Z range: {cam_positions[:, 2].min():.2f} to {cam_positions[:, 2].max():.2f}")
    logger.info(f"First 5 camera positions:")
    for i in range(min(5, len(cam_positions))):
        logger.info(f"  Cam {i}: [{cam_positions[i, 0]:.3f}, {cam_positions[i, 1]:.3f}, {cam_positions[i, 2]:.3f}]")
    logger.info(f"Last 5 camera positions:")
    for i in range(max(0, len(cam_positions) - 5), len(cam_positions)):
        logger.info(f"  Cam {i}: [{cam_positions[i, 0]:.3f}, {cam_positions[i, 1]:.3f}, {cam_positions[i, 2]:.3f}]")
    
    logger.info("Loading intrinsics data...")
    intrinsics_npz = np.load(intrinsics_path)
    intrinsics_data = intrinsics_npz["data"]  # (N, 4) - fx, fy, cx, cy
    
    logger.info("Loading SLAM map...")
    slam_data = load_slam_map(slam_map_path)
    
    # Create output directory
    output_dir = output_path / sequence_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images
    width, height = copy_images_for_colmap(images_dir, output_dir)
    
    logger.info(
        "Using images with dimensions %dx%d",
        width,
        height,
    )
    
    # Build COLMAP model (single sparse/0 directory)
    logger.info("Building COLMAP model...")
    
    # Build cameras with VIPE intrinsics (no scaling)
    cameras = build_cameras(intrinsics_data, width, height)
    
    # Build images
    images = build_images(pose_data, pose_indices, output_dir)
    
    # Build points3D from SLAM map
    points3D = build_points3D_from_slam(slam_data, point_subsample)
    
    # Write COLMAP model
    write_colmap_model(cameras, images, points3D, output_dir)
    
    logger.info(f"Conversion complete! Output: {output_dir}")
    logger.info(f"  - sparse/0/: COLMAP model (txt + bin)")
    logger.info(f"  - images/: Copied images ({width}x{height})")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ViPE SLAM results to COLMAP format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "vipe_path",
        type=Path,
        help="Path to ViPE output directory (e.g., output/zavod_1920_1)",
    )
    parser.add_argument(
        "--images", "-i",
        type=Path,
        required=True,
        help="Path to images directory (images should be at the desired resolution)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory (default: <vipe_path>_colmap)",
    )
    parser.add_argument(
        "--sequence", "-s",
        type=str,
        default=None,
        help="Sequence name (auto-detected if not provided)",
    )
    parser.add_argument(
        "--point-subsample",
        type=int,
        default=1,
        help="Subsample factor for 3D points (1 = no subsampling)",
    )
    
    args = parser.parse_args()
    
    if not args.vipe_path.exists():
        logger.error(f"ViPE path does not exist: {args.vipe_path}")
        return 1
    if not args.images.exists():
        logger.error(f"Images directory does not exist: {args.images}")
        return 1
    
    if args.output is None:
        args.output = args.vipe_path.parent / f"{args.vipe_path.name}_colmap"
    
    convert_vipe_slam_to_colmap(
        vipe_path=args.vipe_path,
        images_dir=args.images,
        output_path=args.output,
        sequence_name=args.sequence,
        point_subsample=args.point_subsample,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
