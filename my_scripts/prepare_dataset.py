#!/usr/bin/env python3
"""
Script to prepare dataset from images or video.
Extracts frames, renames them properly, and creates multiple resolution versions.
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import cv2


def extract_frames_from_video(video_path, output_dir, frame_skip=1, frame_start=0, frame_end=None):
    """Extract frames from video file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_end is None:
        frame_end = total_frames
    
    print(f"Extracting frames from video: {video_path}")
    print(f"Total frames: {total_frames}, extracting: {frame_start} to {frame_end}, skip: {frame_skip}")
    
    frame_idx = 0
    output_idx = 0
    
    with tqdm(total=(frame_end - frame_start) // frame_skip) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= frame_end:
                break
            
            if frame_idx >= frame_start and (frame_idx - frame_start) % frame_skip == 0:
                output_path = output_dir / f"{output_idx:05d}.jpg"
                cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                output_idx += 1
                pbar.update(1)
            
            frame_idx += 1
    
    cap.release()
    print(f"Extracted {output_idx} frames to {output_dir}")
    return output_idx


def process_image_directory(input_dir, output_dir, frame_skip=1, frame_start=0, frame_end=None):
    """Copy and rename images from directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(sorted(input_dir.glob(f"*{ext}")))
    
    if not image_files:
        raise ValueError(f"No images found in {input_dir}")
    
    # Apply frame range
    if frame_end is None:
        frame_end = len(image_files)
    image_files = image_files[frame_start:frame_end:frame_skip]
    
    print(f"Processing {len(image_files)} images from {input_dir}")
    
    for idx, img_path in enumerate(tqdm(image_files, desc="Copying images")):
        output_path = output_dir / f"{idx:05d}.jpg"
        
        # Open and save as JPEG to standardize format
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img.save(output_path, 'JPEG', quality=95)
    
    print(f"Processed {len(image_files)} images to {output_dir}")
    return len(image_files)


def create_resized_versions(input_dir, output_base_dir, scales=[1, 2, 4, 8], format='jpg', enforce_even=False):
    """Create multiple resolution versions of images.
    If enforce_even=True, ensures output images have even width and height (required by yuv420p video encoders).
    """
    input_dir = Path(input_dir)
    output_base_dir = Path(output_base_dir)
    
    # Get all images
    image_files = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))
    if not image_files:
        raise ValueError(f"No images found in {input_dir}")
    
    print(f"Creating resized versions with scales: {scales}")
    
    for scale in scales:
        if scale == 1:
            output_dir = output_base_dir / "images"
        else:
            output_dir = output_base_dir / f"images_{scale}"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating scale 1/{scale} in {output_dir}")
        
        if scale == 1:
            # For scale 1, optionally crop to even dimensions to avoid encoder issues
            for img_path in tqdm(image_files, desc=f"Scale 1/{scale}"):
                output_path = output_dir / img_path.name
                if format.lower() == 'png':
                    output_path = output_path.with_suffix('.png')
                else:
                    output_path = output_path.with_suffix('.jpg')
                if enforce_even:
                    img = Image.open(img_path)
                    w, h = img.width, img.height
                    new_w = w - (w % 2)
                    new_h = h - (h % 2)
                    if new_w != w or new_h != h:
                        # Crop right/bottom by 1px when odd
                        new_w = max(2, new_w)
                        new_h = max(2, new_h)
                        img = img.crop((0, 0, new_w, new_h))
                        img.save(output_path, 'PNG' if format.lower() == 'png' else 'JPEG', quality=95)
                    else:
                        shutil.copy2(img_path, output_path)
                else:
                    shutil.copy2(img_path, output_path)
        else:
            # Resize for other scales
            for img_path in tqdm(image_files, desc=f"Scale 1/{scale}"):
                img = Image.open(img_path)
                new_width = img.width // scale
                new_height = img.height // scale
                # Ensure even dimensions; crop by 1px if necessary, and avoid zero
                new_width = max(2, new_width - (new_width % 2))
                new_height = max(2, new_height - (new_height % 2))
                img_resized = img.resize((new_width, new_height), Image.LANCZOS)
                
                output_path = output_dir / img_path.name
                
                if format.lower() == 'png':
                    img_resized.save(output_path.with_suffix('.png'), 'PNG')
                else:
                    img_resized.save(output_path.with_suffix('.jpg'), 'JPEG', quality=95)
    
    print(f"Created {len(scales)} resolution versions")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset from images or video with multiple resolutions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from video with resize
  python prepare_dataset.py --input video.mp4 --output ./dataset --scales 1 2 4 8
  
  # Process image directory with frame skip
  python prepare_dataset.py --input ./images --output ./dataset --frame-skip 2 --scales 1 2 4
  
  # Extract frames 100-500 from video
  python prepare_dataset.py --input video.mp4 --output ./dataset --frame-start 100 --frame-end 500
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input video file or directory with images')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output directory for processed dataset')
    parser.add_argument('--scales', '-s', type=int, nargs='+', default=[1],
                        help='Resolution scales (1=original, 2=half, 4=quarter, 8=eighth). Default: [1]')
    parser.add_argument('--frame-skip', type=int, default=1,
                        help='Process every Nth frame. Default: 1 (all frames)')
    parser.add_argument('--frame-start', type=int, default=0,
                        help='Start frame index. Default: 0')
    parser.add_argument('--frame-end', type=int, default=None,
                        help='End frame index. Default: None (all frames)')
    parser.add_argument('--format', type=str, choices=['jpg', 'png'], default='jpg',
                        help='Output image format. Default: jpg')
    parser.add_argument('--temp-dir', type=str, default=None,
                        help='Temporary directory for intermediate files')
    parser.add_argument('--enforce-even', action='store_true',
                        help='Ensure output images have even width and height (yuv420p-safe)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")
    
    # If scale 1 is in the list, write directly to images/ directory
    # Otherwise use temporary directory
    needs_temp = 1 not in args.scales
    
    if needs_temp:
        if args.temp_dir:
            base_dir = Path(args.temp_dir)
        else:
            base_dir = output_path / "temp_original"
    else:
        base_dir = output_path / "images"
    
    base_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Extract/copy frames to base directory
        if input_path.is_file():
            # Input is video
            num_frames = extract_frames_from_video(
                input_path, 
                base_dir,
                frame_skip=args.frame_skip,
                frame_start=args.frame_start,
                frame_end=args.frame_end
            )
        elif input_path.is_dir():
            # Input is directory
            num_frames = process_image_directory(
                input_path,
                base_dir,
                frame_skip=args.frame_skip,
                frame_start=args.frame_start,
                frame_end=args.frame_end
            )
        else:
            raise ValueError(f"Input must be a file or directory: {input_path}")
        
        # Step 2: Create resized versions (skip scale 1 if already created)
        scales_to_create = [s for s in args.scales if s != 1]
        if scales_to_create:
            create_resized_versions(
                base_dir,
                output_path,
                scales=scales_to_create,
                format=args.format,
                enforce_even=args.enforce_even
            )
        
        # Step 3: Clean up temporary directory if needed
        if needs_temp and not args.temp_dir and base_dir.exists():
            shutil.rmtree(base_dir)
            print("Cleaned up temporary files")
        
        print(f"\nDataset preparation complete!")
        print(f"Total frames processed: {num_frames}")
        print(f"Output directory: {output_path}")
        print(f"Resolution scales: {args.scales}")
        
    except Exception as e:
        print(f"Error: {e}")
        # Clean up on error
        if needs_temp and not args.temp_dir and base_dir.exists():
            shutil.rmtree(base_dir)
        raise


if __name__ == "__main__":
    main()
