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
  --optimized-trajectory  Use optimized ViPE config for maximum tracking stability
                          Changes 12 parameters to prevent tracking loss & teleportation:
                          
                          Keyframe density (capture more poses):
                            • kf_gap_sec: 2.0→0.5 (4x more keyframes/sec)
                            • keyframe_thresh: 4.0→2.5 (easier to create)
                            • filter_thresh: 2.4→1.5 (more motion sensitive)
                          
                          Frontend connections (prevent tracking loss):
                            • frontend_window: 25→35 (larger optimization)
                            • frontend_radius: 2→3 (more forced connections)
                            • frontend_thresh: 16.0→20.0 (connect distant frames)
                            • frontend_nms: 1→0 (keep all edges, no suppression)
                          
                          Backend optimization (prevent teleportation):
                            • backend_iters: 24→32 (more refinement iterations)
                            • backend_thresh: 22.0→28.0 (include more frames)
                            • backend_radius: 2→3 (more forced connections)
                            • backend_nms: 3→2 (allow more edges)
                          
                          Advanced features:
                            • warmup: 8→12 (better initialization)
                            • adaptive_cross_view: false→true (dynamic cross-view)
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

## 🎓 When to Use Optimized Trajectory Mode

### ✅ **Strongly Recommended For**:

1. **Handheld/POV Footage**
   - Shaky camera, unstable motion
   - Walking, running, or vehicle-mounted cameras
   - Example: `--optimized-trajectory --frame-skip 2`

2. **Complex Camera Movements**
   - Continuous rotation or panning
   - Combined movements (rotate + zoom + pan)
   - Example: `--optimized-trajectory --max-size 640`

3. **Fast Motion Scenes**
   - Quick camera movements
   - Fast-moving subjects in view
   - Example: `--optimized-trajectory --frame-skip 3`

4. **Long Sequences**
   - Videos longer than 30 seconds
   - Where drift accumulates over time
   - Example: `--optimized-trajectory --max-size 640 --frame-skip 2`

5. **Challenging Visual Conditions**
   - Low-texture scenes (walls, sky, uniform surfaces)
   - Repetitive patterns (tiles, windows, fences)
   - Motion blur or out-of-focus sections
   - Example: `--optimized-trajectory`

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

### 📊 **Comparison Table**:

| Scenario | Standard ViPE | Optimized Trajectory | COLMAP |
|----------|---------------|---------------------|--------|
| Handheld video | ⚠️ May lose tracking | ✅ Stable | ❌ Not designed for video |
| Complex movements | ⚠️ Risk of teleportation | ✅ Prevents jumps | ❌ May fail |
| Fast motion | ⚠️ Can lose frames | ✅ Maintains tracking | ❌ Poor results |
| Long sequences | ⚠️ Drift accumulates | ✅ Corrects drift | ⚠️ Slow |
| Static camera | ✅ Fast, good | ⚠️ Overkill (slower) | ✅ Best quality |
| Image set | ⚠️ Suboptimal | ⚠️ Suboptimal | ✅ Designed for this |
| Processing speed | ⚡ Fastest | 🐢 ~50% slower | 🐌 Slowest |
| Memory usage | 💾 Normal | 💾💾 ~1.5-2x | 💾 Normal |

## 💡 Examples

```bash
# Test example with dog video
python pipeline.py data/input/dog-example.mp4 output/dog_test --frame-skip 2

# Test with paper resolution
python pipeline.py data/input/dog-example.mp4 output/dog_640 --max-size 640x480 --frame-skip 3

# OPTIMIZED TRAJECTORY MODE - for preserving camera poses!
# Denser keyframes in ViPE SLAM for better trajectory
python pipeline.py video.mp4 output/optimized --optimized-trajectory

# Optimized trajectory with frame skip
python pipeline.py video.mp4 output/optimized --optimized-trajectory --frame-skip 2

# Optimized trajectory with resolution and frame skip
python pipeline.py video.mp4 output/best --optimized-trajectory --max-size 640 --frame-skip 2

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

## 🔍 Pipeline Modes

### **ViPE mode** (default)
Faster, better for video, more robust
```
Images → ViPE SLAM → COLMAP format → GSplat
```

### **ViPE Optimized Trajectory mode** (`--optimized-trajectory`)
**Maximum tracking stability - prevents tracking loss and camera teleportation**
```
Images → ViPE SLAM (12 optimized parameters) → COLMAP format → GSplat
```

**Use this when**: Camera tracking stability is critical!
- **Complex camera movements** (rotating, panning, zooming)
- **Fast motion** or **shaky footage**  
- **Cannot afford to lose tracking** or camera teleportation
- **Long sequences** where drift accumulates
- **Challenging scenes** (low texture, repetitive patterns)

---

**What it does** (12 parameter changes):

**📸 Keyframe Density** (capture more poses):
- `kf_gap_sec`: 2.0 → **0.5** (4x more keyframes per second)
- `keyframe_thresh`: 4.0 → **2.5** (easier threshold to create keyframes)
- `filter_thresh`: 2.4 → **1.5** (more sensitive to camera motion)

**🔗 Frontend Optimization** (prevent tracking loss):
- `frontend_window`: 25 → **35** (40% larger optimization window)
- `frontend_radius`: 2 → **3** (force connections with 50% more neighbors)
- `frontend_thresh`: 16.0 → **20.0** (connect frames at 25% greater distances)
- `frontend_nms`: 1 → **0** (disable non-maximum suppression - keep all edges)

**⚙️ Backend Optimization** (prevent camera teleportation):
- `backend_iters`: 24 → **32** (33% more refinement iterations)
- `backend_thresh`: 22.0 → **28.0** (include 27% more frames in global optimization)
- `backend_radius`: 2 → **3** (50% more forced backend connections)
- `backend_nms`: 3 → **2** (allow more edges in global optimization)

**🎯 Advanced Features**:
- `warmup`: 8 → **12** (50% better initialization before tracking starts)
- `adaptive_cross_view`: false → **true** (dynamically recompute cross-view connections)

---

**Why these parameters work together**:

1. **Denser keyframes** → Smaller gaps between tracked frames → Less chance to lose tracking
2. **More frontend connections** → Even if one connection fails, others maintain tracking
3. **Disabled NMS** → Keep all potential matches instead of filtering "redundant" ones
4. **Stronger backend** → Global optimization corrects accumulated errors and prevents drift
5. **Adaptive cross-view** → Automatically finds best cross-frame connections in complex scenes
6. **Better warmup** → More accurate initial map before camera starts moving

---

**Trade-offs**:
- ✅ **Much better tracking stability** (fewer tracking failures)
- ✅ **Prevents camera teleportation** (no sudden jumps)
- ✅ **More accurate trajectory** for long sequences
- ✅ **Handles challenging scenes** better
- ⚠️ **~40-50% slower** processing (more keyframes + optimization)
- ⚠️ **~1.5-2x memory** usage (more connections stored)
- ⚠️ **More disk space** for SLAM map (denser keyframes)

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
