#!/bin/bash

# --- 1. Dọn dẹp môi trường hệ thống ---
echo "Cleaning up system modules..."
module --force purge

# --- 2. Load các module cần thiết cho JUPITER ---
echo "Loading Stages, GCC, and CUDA..."
module load Stages/2025
module load GCC/13.3.0
module load CUDA/12

# --- 3. Kích hoạt bộ máy Miniforge của bạn ---
# Đường dẫn này trỏ thẳng vào folder miniforge3 bạn đã cài trong project
echo "Initializing Miniforge..."
source /e/project1/reformo/nguyen38/3d-human-pose/miniforge3/bin/activate

# --- 4. Kích hoạt môi trường MotionBERT + YOLO ---
echo "Activating env_motion_final..."
conda activate /e/project1/reformo/nguyen38/3d-human-pose/env_motion_final

# --- 5. Kiểm tra nhanh tình trạng ---
echo "------------------------------------------------"
python -c "import torch; print('🚀 GPU H100 Status:', torch.cuda.is_available()); import numpy; print('🔢 NumPy Version:', numpy.__version__)"
echo "✅ Everything is READY! Let's work on MotionBERT."
echo "------------------------------------------------"