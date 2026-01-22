# Gaussian Splatting Pipeline

Two simple files for complete pipeline: ViPE or COLMAP → GSplat

## 📦 Files

- `setup_environments.sh` - creates all conda environments
- `pipeline.py` - runs the full pipeline

## 🚀 Quick Start

### 1. Create environments (one time)

```bash
./setup_environments.sh
```

This will create 3 conda environments:
- `vipe` - for ViPE SLAM
- `colmap` - for COLMAP
- `gsplat` - for Gaussian Splatting

### 2. Run test example

```bash
# Process test video with frame skip
python pipeline.py data/input/dog-example.mp4 output/dog_test --frame-skip 2

# Process test video with max size and frame skip
python pipeline.py data/input/dog-example.mp4 output/dog_test --max-size 640 --frame-skip 3
```

### 3. Run on your data

```bash
# From video file
python pipeline.py /path/to/video.mp4 /path/to/output

# From image directory
python pipeline.py /path/to/images/ /path/to/output

# With COLMAP mode
python pipeline.py /path/to/video.mp4 /path/to/output --mode colmap
```

## 📝 Parameters

```
pipeline.py INPUT OUTPUT [OPTIONS]

Arguments:
  INPUT                   Path to input video file or images directory
  OUTPUT                  Directory for results

Options:
  --mode {vipe,colmap}    Pipeline mode (default: vipe)
  --max-size SIZE         Max resolution: single int for longest side (e.g., 640)
                          or WIDTHxHEIGHT (e.g., 640x480). Optional.
                          Paper uses: 640x480
  --frame-skip N          Process every Nth frame (default: 1 = all frames)
                          Use 2-5 to speed up processing
```

## 📁 Results Structure

### ViPE mode:
```
output_dir/
├── processed/
│   └── images/                # Processed images
├── vipe_output/               # ViPE SLAM results
├── vipe_colmap/               # Converted to COLMAP
└── vipe_gsplat/               # Gaussian Splatting
    └── point_cloud.ply        # 🎯 Result
```

### COLMAP mode:
```
output_dir/
├── processed/
│   └── images/                # Processed images
├── colmap_output/             # COLMAP reconstruction
└── colmap_gsplat/             # Gaussian Splatting
    └── point_cloud.ply        # 🎯 Result
```

## 💡 Examples

```bash
# Test example with dog video
python pipeline.py data/input/dog-example.mp4 output/dog_test --frame-skip 2

# Test with paper resolution
python pipeline.py data/input/dog-example.mp4 output/dog_640 --max-size 640x480 --frame-skip 3

# Video with frame skip (faster processing)
python pipeline.py video.mp4 output/fast --frame-skip 5

# Video with max size and frame skip
python pipeline.py video.mp4 output/result --max-size 640 --frame-skip 2

# Image directory with ViPE mode
python pipeline.py ~/data/images/ output/vipe_run --mode vipe --max-size 640x480

# COLMAP mode with high resolution
python pipeline.py video.mp4 output/colmap_run --mode colmap --max-size 1280x960

# Quick test with low resolution
python pipeline.py video.mp4 output/quick --max-size 480 --frame-skip 4
```

## 🔍 Two Modes

**ViPE mode**: Faster, better for video, more robust
```
Images → ViPE SLAM → COLMAP format → GSplat
```

**COLMAP mode**: Classic SfM, better for static scenes
```
Images → COLMAP reconstruction → GSplat
```

## 📹 Results / Demo Videos

### Pure COLMAP Mode
> Full COLMAP reconstruction → Gaussian Splatting

[![COLMAP Results](https://img.youtube.com/vi/GxzizEA2Eds/0.jpg)](https://www.youtube.com/watch?v=GxzizEA2Eds)

### ViPE Mode (Standard)
> ViPE SLAM → COLMAP format → Gaussian Splatting

[![ViPE Results](https://img.youtube.com/vi/ZvPsXBV6758/0.jpg)](https://www.youtube.com/watch?v=ZvPsXBV6758)

### ViPE Mode (Optimized)
> Trajectory generated on 1/4 frames, Gaussian Splatting trained on 1/2 frames

[![ViPE Optimized](https://img.youtube.com/vi/fHLGGDVGzsc/0.jpg)](https://www.youtube.com/watch?v=fHLGGDVGzsc)

## ⚙️ Requirements

- Conda
- CUDA-capable GPU
- Python 3.10
- Git

## 🛠️ Troubleshooting

```bash
# Check environments
conda env list

# Remove environment and recreate
conda env remove -n vipe
./setup_environments.sh

# Check GPU
nvidia-smi
```

---

Ready! Two files for complete pipeline 🚀
