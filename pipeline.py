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
    parser.add_argument("input_dir", help="Path to input images directory")
    parser.add_argument("output_dir", help="Path to output directory")
    parser.add_argument("--mode", choices=["vipe", "colmap"], default="vipe", help="Pipeline mode")
    parser.add_argument("--scale", type=int, default=4, choices=[1,2,4,8], help="Image scale")
    parser.add_argument("--gsplat-factor", type=int, default=1, help="GSplat data factor")
    
    args = parser.parse_args()
    
    workspace = Path.cwd()
    scripts_dir = workspace / "my_scripts"
    external_dir = workspace / "external"
    
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_dir = output_dir / "processed"
    
    print(f"\n{'='*80}")
    print(f"Pipeline Mode: {args.mode.upper()} → GSplat")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Scale: {args.scale}")
    print(f"{'='*80}\n")
    
    # Step 1: Prepare dataset
    print("\n[STEP 1] Preparing dataset...")
    run_cmd([
        "python", str(scripts_dir / "prepare_dataset.py"),
        "--input", str(input_dir),
        "--output", str(processed_dir),
        "--scales", "1", "2", "4", "8",
        "--enforce-even"
    ], conda_env="vipe")
    
    images_dir = processed_dir / f"images_{args.scale}"
    
    if args.mode == "vipe":
        # ViPE pipeline
        vipe_output = output_dir / f"vipe_scale_{args.scale}"
        vipe_output.mkdir(exist_ok=True)
        
        # Step 2: ViPE SLAM
        print("\n[STEP 2] Running ViPE SLAM...")
        run_cmd([
            "python", str(external_dir / "vipe" / "run.py"),
            "pipeline=default",
            "streams=frame_dir_stream",
            f"streams.base_path={images_dir}",
            "pipeline.output.save_artifacts=true",
            "pipeline.output.save_slam_map=true",
            "pipeline.output.save_viz=true",
            f"pipeline.output.path={vipe_output}"
        ], conda_env="vipe")
        
        # Step 3: Convert to COLMAP
        print("\n[STEP 3] Converting ViPE to COLMAP...")
        colmap_dir = output_dir / f"vipe_colmap_scale_{args.scale}"
        run_cmd([
            "python", str(scripts_dir / "vipe_slam_to_colmap.py"),
            str(vipe_output),
            "--images", str(images_dir),
            "--output", str(colmap_dir),
            "--point-subsample", "1"
        ], conda_env="vipe")
        
        gsplat_output = output_dir / f"vipe_gsplat_scale_{args.scale}"
        
    else:
        # COLMAP pipeline
        colmap_dir = output_dir / f"colmap_scale_{args.scale}"
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
        
        gsplat_output = output_dir / f"colmap_gsplat_scale_{args.scale}"
    
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
        "--data_factor", str(args.gsplat_factor),
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
