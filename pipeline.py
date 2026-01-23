#!/usr/bin/env python3
"""
Simple pipeline: (ViPE OR COLMAP) → GSplat
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def run_cmd(cmd, conda_env=None, env_vars=None):
    """Execute command in conda environment"""
    if conda_env:
        cmd = ["conda", "run", "-n", conda_env] + cmd
    
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    
    result = subprocess.run(cmd, check=False, env=env)
    if result.returncode != 0:
        print(f"\n✗ Command failed with code {result.returncode}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Pipeline: (ViPE or COLMAP) → GSplat")
    parser.add_argument("input_dir", help="Path to input video file or images directory")
    parser.add_argument("output_dir", help="Path to output directory")
    parser.add_argument("--mode", choices=["vipe", "colmap"], default="vipe", help="Pipeline mode")
    parser.add_argument("--max-size", type=str, default=None, help="Max size: single int for longest side (e.g., 640) or WIDTHxHEIGHT (e.g., 640x480)")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame. Default: 1 (all frames)")
    parser.add_argument("--optimized-trajectory", action="store_true", help="Use optimized ViPE config with denser keyframes for better camera pose preservation (ViPE mode only)")
    
    args = parser.parse_args()
    
    workspace = Path.cwd()
    scripts_dir = workspace / "my_scripts"
    external_dir = workspace / "external"
    
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_dir = output_dir / "processed"
    
    # Detect if input is video or directory
    is_video = input_dir.is_file() and input_dir.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.m4v', '.webm']
    input_type = "Video" if is_video else "Image Directory"
    
    print(f"\n{'='*80}")
    print(f"Pipeline Mode: {args.mode.upper()} → GSplat")
    print(f"Input Type: {input_type}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    if args.max_size:
        print(f"Max Size: {args.max_size}")
    if args.frame_skip > 1:
        print(f"Frame Skip: {args.frame_skip}")
    if args.optimized_trajectory:
        print("Trajectory Mode: Optimized (denser keyframes)")
    print(f"{'='*80}\n")
    
    # Step 1: Prepare dataset
    print("\n[STEP 1] Preparing dataset...")
    prepare_cmd = [
        "python", str(scripts_dir / "prepare_dataset.py"),
        "--input", str(input_dir),
        "--output", str(processed_dir),
        "--enforce-even"
    ]
    if args.max_size:
        prepare_cmd.extend(["--max-size", str(args.max_size)])
    if args.frame_skip > 1:
        prepare_cmd.extend(["--frame-skip", str(args.frame_skip)])
    run_cmd(prepare_cmd, conda_env="vipe")
    
    images_dir = processed_dir / "images"
    
    if args.mode == "vipe":
        # ViPE pipeline
        vipe_output = output_dir / "vipe_output"
        vipe_output.mkdir(exist_ok=True)
        
        # Step 2: ViPE SLAM
        print("\n[STEP 2] Running ViPE SLAM...")
        vipe_cmd = [
            "python", str(external_dir / "vipe" / "run.py"),
            "pipeline=default",
            "streams=frame_dir_stream",
            f"streams.base_path={images_dir}",
            "pipeline.output.save_artifacts=true",
            "pipeline.output.save_slam_map=true",
            "pipeline.output.save_viz=true",
            f"pipeline.output.path={vipe_output}"
        ]
        
        # Add optimized trajectory parameters via Hydra overrides
        if args.optimized_trajectory:
            vipe_cmd.extend([
                # Maximum keyframe density for best tracking
                "pipeline.init.instance.kf_gap_sec=0.25",     # was 2.0 - very frequent keyframes
                "pipeline.slam.keyframe_thresh=1.5",          # was 4.0 - very easy keyframe creation
                "pipeline.slam.filter_thresh=1.0",            # was 2.4 - highly sensitive to motion
                
                # Maximum frame connections (prevent tracking loss)
                "pipeline.slam.frontend_window=50",           # was 25 - very large optimization window
                "pipeline.slam.frontend_radius=4",            # was 2 - maximum neighbor connections
                "pipeline.slam.frontend_thresh=25.0",         # was 16.0 - connect distant frames
                "pipeline.slam.frontend_nms=0",               # was 1 - keep all edges (no NMS)
                
                # Aggressive backend optimization (prevent drift)
                "pipeline.slam.backend_iters=48",             # was 24 - maximum backend iterations
                "pipeline.slam.backend_thresh=32.0",          # was 22.0 - very inclusive backend
                "pipeline.slam.backend_radius=4",             # was 2 - maximum backend connections
                "pipeline.slam.backend_nms=1",                # was 3 - more backend edges
                
                # Extended initialization and cross-view matching
                "pipeline.slam.warmup=16",                    # was 8 - extended initialization
                "pipeline.slam.adaptive_cross_view=true",     # was false - recompute cross-view
            ])
        
        run_cmd(vipe_cmd, conda_env="vipe")
        
        # Step 3: Convert to COLMAP
        print("\n[STEP 3] Converting ViPE to COLMAP...")
        colmap_dir = output_dir / "vipe_colmap"
        run_cmd([
            "python", str(scripts_dir / "vipe_slam_to_colmap.py"),
            str(vipe_output),
            "--images", str(images_dir),
            "--output", str(colmap_dir),
            "--point-subsample", "1"
        ], conda_env="vipe")
        
        gsplat_output = output_dir / "vipe_gsplat"
        
    else:
        # COLMAP pipeline
        colmap_dir = output_dir / "colmap_output"
        colmap_dir.mkdir(exist_ok=True)
        
        # Copy images
        images_dest = colmap_dir / "images"
        images_dest.mkdir(exist_ok=True)
        for img in images_dir.glob("*"):
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                import shutil
                shutil.copy(img, images_dest / img.name)
        
        # Step 2: COLMAP reconstruction
        print("\n[STEP 2] Running COLMAP reconstruction...")
        run_cmd([
            "python", str(scripts_dir / "colmap.py"),
            str(colmap_dir),
            "--matcher", "sequential",
            "--resize"
        ], conda_env="colmap")
        
        gsplat_output = output_dir / "colmap_gsplat"
    
    # Find COLMAP data directory
    if not (colmap_dir / "images").exists():
        subdirs = [d for d in colmap_dir.iterdir() if d.is_dir()]
        if subdirs:
            colmap_dir = subdirs[0]
    
    # Step 4: Train GSplat
    print(f"\n[STEP {3 if args.mode == 'vipe' else 3}] Training Gaussian Splatting...")
    run_cmd([
        "python", str(external_dir / "gsplat" / "examples" / "simple_trainer.py"),
        "default",
        "--data_dir", str(colmap_dir),
        "--result_dir", str(gsplat_output),
        "--data_factor", "1",
        "--save_ply"
    ], conda_env="gsplat", env_vars={"CUDA_VISIBLE_DEVICES": "0"})
    
    print(f"\n{'='*80}")
    print("✓ PIPELINE COMPLETED!")
    print(f"{'='*80}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Output: {output_dir}")
    print(f"GSplat model: {gsplat_output}/point_cloud.ply")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
