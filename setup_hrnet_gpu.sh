#!/bin/bash

# 1. Dọn dẹp module hệ thống
module --force purge

# 2. Load các module cần thiết cho H100 trên JUPITER
module load Stages/2025 GCC/13.3.0 CUDA/12

# 3. Kích hoạt Miniforge
source /e/project1/reformo/nguyen38/3d-human-pose/miniforge3/bin/activate

# 4. Kích hoạt môi trường HRNet GPU (MMPose 1.x)
conda activate /e/project1/reformo/nguyen38/3d-human-pose/env_hrnet_gpu

echo "------------------------------------------------"
echo "🚀 Phase 1: HRNet (GPU H100 Mode) is READY!"
echo "📍 Python: $(which python)"
echo "📍 CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "------------------------------------------------"