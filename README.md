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

# Process using COLMAP
python pipeline.py data/input/dog-example.mp4 output/dog_test_colmap --max-size 640 --frame-skip 3 --mode colmap
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
  --opt-balanced          Use balanced optimization: 4x denser keyframes,
                          larger optimization windows. Good for handheld footage
                          and complex camera movements (ViPE mode only).
                          Trade-off: ~30-40% slower, 1.5x memory
  --opt-aggressive        Use aggressive optimization: 8x denser keyframes,
                          maximum tracking accuracy. Best for challenging scenes,
                          fast motion, and long sequences (ViPE mode only).
                          Trade-off: ~60-80% slower, 2-3x memory
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

## 🎓 When to Use Optimization Modes

### ✅ **Strongly Recommended For**:

1. **Handheld/POV Footage**
   - Shaky camera, unstable motion
   - Walking, running, or vehicle-mounted cameras
   - Use: `--opt-balanced --frame-skip 2`

2. **Complex Camera Movements**
   - Continuous rotation or panning
   - Combined movements (rotate + zoom + pan)
   - Use: `--opt-balanced --max-size 640`

3. **Fast Motion Scenes**
   - Quick camera movements
   - Fast-moving subjects in view
   - Use: `--opt-aggressive --frame-skip 3`

4. **Long Sequences**
   - Videos longer than 30 seconds
   - Where drift accumulates over time
   - Use: `--opt-aggressive --max-size 640 --frame-skip 2`

5. **Challenging Visual Conditions**
   - Low-texture scenes (walls, sky, uniform surfaces)
   - Repetitive patterns (tiles, windows, fences)
   - Motion blur or out-of-focus sections
   - Use: `--opt-aggressive`

### ⚠️ **May Not Need For**:

1. **Tripod/Stable Camera**
   - Static camera with minimal movement
   - Professional stabilized footage
   - Use: Standard mode is sufficient

2. **Short Clips**
   - Videos under 10 seconds
   - Limited camera movement
   - Use: Standard mode for faster processing

3. **Image Sequences** (not video)
   - Pre-captured photo sets
   - May benefit more from COLMAP mode
   - Use: `--mode colmap`

## 💡 Examples

```bash
# Basic usage
python pipeline.py video.mp4 output/result

# Balanced optimization (recommended for handheld footage)
python pipeline.py video.mp4 output/balanced --opt-balanced --max-size 640 --frame-skip 2

# Aggressive optimization (maximum tracking accuracy for challenging scenes)
python pipeline.py video.mp4 output/best --opt-aggressive --max-size 640 --frame-skip 2

# Fast processing (less accurate)
python pipeline.py video.mp4 output/fast --frame-skip 5 --max-size 480

# COLMAP mode for static camera/image sets
python pipeline.py images/ output/colmap --mode colmap

# Test with dog video
python pipeline.py data/input/dog-example.mp4 output/dog_test --frame-skip 2
```

## 🔍 Pipeline Modes

### **ViPE mode** (default)
Faster, better for video, more robust
```
Images → ViPE SLAM → COLMAP format → GSplat
```

### **ViPE Balanced Optimization mode** (`--opt-balanced`)
**Good balance: 4x denser keyframes - stable tracking with reasonable performance**
```
Images → ViPE SLAM (9 optimized parameters) → COLMAP format → GSplat
```

**Best for**: Handheld footage • Moderate camera movements • General use cases

**9 Parameter Changes**:

**📸 Keyframe Density** (capture more poses):
- `kf_gap_sec`: 2.0 → **0.5** (4x more keyframes per second)
- `keyframe_thresh`: 4.0 → **2.5** (easier threshold to create keyframes)
- `filter_thresh`: 2.4 → **1.5** (more sensitive to camera motion)

**🔗 Frontend Optimization** (better tracking):
- `frontend_window`: 25 → **40** (60% larger optimization window)
- `frontend_radius`: 2 → **3** (50% more neighbor connections)
- `frontend_thresh`: 16.0 → **20.0** (connect frames 25% further)

**⚙️ Backend Optimization** (reduce drift):
- `backend_iters`: 24 → **36** (50% more refinement iterations)
- `backend_thresh`: 22.0 → **28.0** (27% more frames in global optimization)
- `backend_radius`: 2 → **3** (50% more backend connections)

**🎯 Advanced Features**:
- `warmup`: 8 → **12** (50% better initialization)

**Trade-offs**:
- ✅ **Significantly better tracking** than default
- ✅ **Good stability** for handheld footage
- ✅ **Balanced performance** (reasonable speed/memory)
- ⚠️ **~30-40% slower** processing (4x more keyframes)
- ⚠️ **~1.5x memory** usage (50% more connections)

### **ViPE Aggressive Optimization mode** (`--opt-aggressive`)
**Maximum tracking accuracy - prevents tracking loss and camera teleportation**
```
Images → ViPE SLAM (16 optimized parameters) → COLMAP format → GSplat
```

**Best for**: Fast motion • Challenging scenes • Long sequences • Low-texture environments

**16 Parameter Changes**:

**📸 Keyframe Density** (capture maximum poses):
- `kf_gap_sec`: 2.0 → **0.25** (8x more keyframes per second)
- `keyframe_thresh`: 4.0 → **1.5** (much easier threshold to create keyframes)
- `filter_thresh`: 2.4 → **1.0** (highly sensitive to camera motion)

**🔗 Frontend Optimization** (prevent tracking loss):
- `frontend_window`: 25 → **50** (2x larger optimization window)
- `frontend_radius`: 2 → **4** (force connections with 2x more neighbors)
- `frontend_thresh`: 16.0 → **25.0** (connect frames at 56% greater distances)
- `frontend_nms`: 1 → **0** (disable non-maximum suppression - keep all edges)

**⚙️ Backend Optimization** (prevent camera teleportation):
- `backend_iters`: 24 → **48** (2x more refinement iterations)
- `backend_thresh`: 22.0 → **32.0** (include 45% more frames in global optimization)
- `backend_radius`: 2 → **4** (2x more forced backend connections)
- `backend_nms`: 3 → **1** (maximize edges in global optimization)

**🎯 Advanced Features**:
- `warmup`: 8 → **16** (2x better initialization before tracking starts)
- `adaptive_cross_view`: false → **true** (dynamically recompute cross-view connections)

**Trade-offs**:
- ✅ **Maximum tracking stability** (minimal tracking failures)
- ✅ **Prevents camera teleportation** (no sudden jumps)
- ✅ **Most accurate trajectory** for long sequences
- ✅ **Best for challenging scenes** (low texture, motion blur, etc.)
- ✅ **Loop closure detection** (corrects accumulated drift)
- ⚠️ **~60-80% slower** processing (8x more keyframes + 2x optimization)
- ⚠️ **~2-3x memory** usage (2x more connections stored)
- ⚠️ **Significantly more disk space** for SLAM map (8x denser keyframes)

### **COLMAP mode**
Classic SfM, better for static scenes
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
