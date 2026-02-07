#!/bin/bash
# ===================================================
# Linux Server Setup Script for Multi-View Defect Synthesis
# Tested on: Ubuntu 20.04/22.04 with CUDA 11.8+
# ===================================================

set -e  # Exit on error

echo "======================================================"
echo "Multi-View Defect Synthesis - Server Setup"
echo "======================================================"

# 1. Create conda environment
echo "[Step 1/4] Creating conda environment..."
conda create -n defect_gen python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate defect_gen

# 2. Install PyTorch with CUDA
echo "[Step 2/4] Installing PyTorch (CUDA 11.8)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install minimal dependencies
echo "[Step 3/4] Installing project dependencies..."
pip install -r requirements_server.txt

# 4. Verify installation
echo "[Step 4/4] Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"

echo ""
echo "======================================================"
echo "Setup Complete!"
echo "======================================================"
echo ""
echo "To start training, run:"
echo "  conda activate defect_gen"
echo "  ./scripts/run_all_views.sh"
echo ""
echo "Or train a single view:"
echo "  CUDA_VISIBLE_DEVICES=0 python scripts/train_baseline.py --view Cam1 --epochs 50"
echo ""
