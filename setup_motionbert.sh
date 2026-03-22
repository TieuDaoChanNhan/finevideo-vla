#!/bin/bash

echo "Cleaning up system modules..."
module --force purge

echo "Loading Stages, GCC, and CUDA..."
module load Stages/2025
module load GCC/13.3.0
module load CUDA/12

echo "Initializing Miniforge..."
source /e/project1/reformo/nguyen38/3d-human-pose/miniforge3/bin/activate

echo "Activating env_motion_final..."
conda activate /e/project1/reformo/nguyen38/3d-human-pose/env_motion_final

echo "------------------------------------------------"
python -c "import torch; print('🚀 GPU H100 Status:', torch.cuda.is_available()); import numpy; print('🔢 NumPy Version:', numpy.__version__)"
echo "✅ Everything is READY! Let's work on MotionBERT."
echo "------------------------------------------------"