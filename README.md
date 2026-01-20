# Gaussian Splatting Pipeline

Two simple files for complete pipeline: ViPE or COLMAP → GSplat

## 📦 Files

- `setup_environments.sh` - creates all conda environments
- `pipeline.py` - runs the full pipeline

## 🚀 Usage

### 1. Create environments (one time)

```bash
./setup_environments.sh
```

This will create 3 conda environments:
- `vipe` - for ViPE SLAM
- `colmap` - for COLMAP
- `gsplat` - for Gaussian Splatting

### 2. Run pipeline

```bash
# ViPE mode (default)
python pipeline.py /path/to/images /path/to/output

# COLMAP mode
python pipeline.py /path/to/images /path/to/output --mode colmap

# With parameters
python pipeline.py /path/to/images /path/to/output --mode vipe --scale 4 --gsplat-factor 1
```

## 📝 Parameters

```
pipeline.py INPUT_DIR OUTPUT_DIR [OPTIONS]

Arguments:
  INPUT_DIR                Directory with input images
  OUTPUT_DIR              Directory for results

Options:
  --mode {vipe,colmap}    Pipeline mode (default: vipe)
  --scale {1,2,4,8}       Processing scale (default: 4)
  --gsplat-factor N       Data factor for GSplat (default: 1)
```

## 📁 Results Structure

### ViPE mode:
```
output_dir/
├── processed/
│   └── images_4/
├── vipe_scale_4/           # ViPE SLAM results
├── vipe_colmap_scale_4/    # Converted to COLMAP
└── vipe_gsplat_scale_4/    # Gaussian Splatting
    └── point_cloud.ply     # 🎯 Result
```

### COLMAP mode:
```
output_dir/
├── processed/
│   └── images_4/
├── colmap_scale_4/         # COLMAP reconstruction
└── colmap_gsplat_scale_4/  # Gaussian Splatting
    └── point_cloud.ply     # 🎯 Result
```

## 💡 Examples

```bash
# Quick test (scale 8)
python pipeline.py data/input/test_images output/test1 --scale 2

# ViPE, standard quality
python pipeline.py ~/data/zavod70 output/vipe_run --mode vipe --scale 4

# COLMAP, high quality
python pipeline.py ~/data/zavod70 output/colmap_run --mode colmap --scale 2

# Quick GSplat (data_factor=4)
python pipeline.py ~/data/zavod70 output/quick --scale 8 --gsplat-factor 4
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
