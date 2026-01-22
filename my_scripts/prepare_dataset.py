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


def extract_frames_from_video(video_path, output_dir, frame_skip=1, frame_start=0, frame_end=None, max_size=None):
    """Extract frames from video file.
    Args:
        max_size: int for longest side, or tuple (width, height) for exact max dimensions
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_end is None:
        frame_end = total_frames
    
    print(f"Extracting frames from video: {video_path}")
    print(f"Total frames: {total_frames}, extracting: {frame_start} to {frame_end}, skip: {frame_skip}")
    if max_size:
        if isinstance(max_size, tuple):
            print(f"Resizing to max: {max_size[0]}x{max_size[1]}px")
        else:
            print(f"Resizing longest side to: {max_size}px")
    
    frame_idx = 0
    output_idx = 0
    
    with tqdm(total=(frame_end - frame_start) // frame_skip) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= frame_end:
                break
            
            if frame_idx >= frame_start and (frame_idx - frame_start) % frame_skip == 0:
                # Resize if max_size specified
                if max_size:
                    h, w = frame.shape[:2]
                    if isinstance(max_size, tuple):
                        # max_size is (max_width, max_height)
                        scale = min(max_size[0] / w, max_size[1] / h)
                        if scale < 1:
                            new_w = int(w * scale)
                            new_h = int(h * scale)
                            # Ensure even dimensions
                            new_w = new_w - (new_w % 2)
                            new_h = new_h - (new_h % 2)
                            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                    else:
                        # max_size is int (longest side)
                        longest_side = max(w, h)
                        if longest_side > max_size:
                            scale = max_size / longest_side
                            new_w = int(w * scale)
                            new_h = int(h * scale)
                            # Ensure even dimensions
                            new_w = new_w - (new_w % 2)
                            new_h = new_h - (new_h % 2)
                            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                
                output_path = output_dir / f"{output_idx:05d}.jpg"
                cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                output_idx += 1
                pbar.update(1)
            
            frame_idx += 1
    
    cap.release()
    print(f"Extracted {output_idx} frames to {output_dir}")
    return output_idx


def process_image_directory(input_dir, output_dir, frame_skip=1, frame_start=0, frame_end=None, max_size=None):
    """Copy and rename images from directory.
    Args:
        max_size: int for longest side, or tuple (width, height) for exact max dimensions
    """
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
    if max_size:
        if isinstance(max_size, tuple):
            print(f"Resizing to max: {max_size[0]}x{max_size[1]}px")
        else:
            print(f"Resizing longest side to: {max_size}px")
    
    for idx, img_path in enumerate(tqdm(image_files, desc="Copying images")):
        output_path = output_dir / f"{idx:05d}.jpg"
        
        # Open and save as JPEG to standardize format
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # Resize if max_size specified
        if max_size:
            w, h = img.size
            if isinstance(max_size, tuple):
                # max_size is (max_width, max_height)
                scale = min(max_size[0] / w, max_size[1] / h)
                if scale < 1:
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    # Ensure even dimensions
                    new_w = new_w - (new_w % 2)
                    new_h = new_h - (new_h % 2)
                    img = img.resize((new_w, new_h), Image.LANCZOS)
            else:
                # max_size is int (longest side)
                longest_side = max(w, h)
                if longest_side > max_size:
                    scale = max_size / longest_side
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    # Ensure even dimensions
                    new_w = new_w - (new_w % 2)
                    new_h = new_h - (new_h % 2)
                    img = img.resize((new_w, new_h), Image.LANCZOS)
        
        img.save(output_path, 'JPEG', quality=95)
    
    print(f"Processed {len(image_files)} images to {output_dir}")
    return len(image_files)


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
    parser.add_argument('--max-size', type=str, default=None,
                        help='Max size: single int for longest side (e.g., 640) or WIDTHxHEIGHT (e.g., 640x480)')
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
    
    # Parse max_size if provided
    max_size = None
    if args.max_size:
        if 'x' in args.max_size.lower():
            # Format: WIDTHxHEIGHT
            try:
                w, h = map(int, args.max_size.lower().split('x'))
                max_size = (w, h)
            except:
                raise ValueError(f"Invalid --max-size format: {args.max_size}. Use integer (e.g., 640) or WIDTHxHEIGHT (e.g., 640x480)")
        else:
            # Format: single integer
            try:
                max_size = int(args.max_size)
            except:
                raise ValueError(f"Invalid --max-size format: {args.max_size}. Use integer (e.g., 640) or WIDTHxHEIGHT (e.g., 640x480)")
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")
    
    # Output to images/ directory
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
                frame_end=args.frame_end,
                max_size=max_size
            )
        elif input_path.is_dir():
            # Input is directory
            num_frames = process_image_directory(
                input_path,
                base_dir,
                frame_skip=args.frame_skip,
                frame_start=args.frame_start,
                frame_end=args.frame_end,
                max_size=max_size
            )
        else:
            raise ValueError(f"Input must be a file or directory: {input_path}")
        
        print(f"\nDataset preparation complete!")
        print(f"Total frames processed: {num_frames}")
        print(f"Output directory: {output_path}")
        if max_size:
            if isinstance(max_size, tuple):
                print(f"Max resolution: {max_size[0]}x{max_size[1]}px")
            else:
                print(f"Max resolution: {max_size}px (longest side)")
        
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
