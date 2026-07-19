#!/bin/bash

module --force purge

module load Stages/2025 GCC/13.3.0 CUDA/12

source /e/project1/reformo/nguyen38/3d-human-pose/miniforge3/bin/activate

conda activate /e/data1/datasets/playground/mmlaion/shared/nguyen38/3d-human-pose/env_hrnet_datasets_v1

echo "------------------------------------------------"
echo "🚀 Phase 1: HRNet (GPU H100 Mode) is READY!"
echo "📍 Python: $(which python)"
echo "📍 CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "------------------------------------------------"