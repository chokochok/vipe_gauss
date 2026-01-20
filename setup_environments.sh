#!/bin/bash
# Setup script for creating all conda environments

set -e

echo "=========================================="
echo "Setting up environments for pipeline"
echo "=========================================="

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y fonts-dejavu-core
fc-cache -fv

WORKSPACE=$(pwd)
EXTERNAL_DIR="$WORKSPACE/external"
mkdir -p "$EXTERNAL_DIR"

# ==========================================
# 1. Setup ViPE environment
# ==========================================
echo ""
echo "1/3: Setting up ViPE environment..."
cd "$EXTERNAL_DIR"
if [ ! -d "vipe" ]; then
    git clone https://github.com/nv-tlabs/vipe
fi
cd vipe

if conda env list | grep -q "^vipe "; then
    echo "ViPE environment already exists, skipping..."
else
    conda env create -f envs/base.yml
    conda run -n vipe pip install -r envs/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
    conda run -n vipe pip install --no-build-isolation -e .
fi

# ==========================================
# 2. Setup COLMAP environment
# ==========================================
echo ""
echo "2/3: Setting up COLMAP environment..."
if conda env list | grep -q "^colmap "; then
    echo "COLMAP environment already exists, skipping..."
else
    conda create -n colmap python=3.10 -y
    conda run -n colmap conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
    conda run -n colmap pip install Pillow tqdm pycolmap-cuda12
fi

# ==========================================
# 3. Setup GSplat environment
# ==========================================
echo ""
echo "3/3: Setting up GSplat environment..."
cd "$EXTERNAL_DIR"
if [ ! -d "gsplat" ]; then
    git clone https://github.com/nerfstudio-project/gsplat.git
fi

if conda env list | grep -q "^gsplat "; then
    echo "GSplat environment already exists, skipping..."
else
    conda create -n gsplat python=3.10 -y
    conda run -n gsplat pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
    conda run -n gsplat conda install -c nvidia/label/cuda-12.4.0 cuda-toolkit -y
    conda run -n gsplat conda install -c conda-forge gxx_linux-64=11 -y
    cd "$EXTERNAL_DIR/gsplat"
    conda run -n gsplat pip install git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation
    conda run -n gsplat pip install -r examples/requirements.txt --no-build-isolation
fi

echo ""
echo "=========================================="
echo "✓ All environments created successfully!"
echo "=========================================="
echo ""
echo "Available environments:"
echo "  - vipe   : ViPE SLAM"
echo "  - colmap : COLMAP reconstruction"
echo "  - gsplat : Gaussian Splatting"
