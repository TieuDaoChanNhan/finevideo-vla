#!/bin/bash
module --force purge
module load Stages/2025 GCC/13.3.0
source /e/project1/reformo/nguyen38/3d-human-pose/miniforge3/bin/activate
conda activate /e/project1/reformo/nguyen38/3d-human-pose/env_hrnet
echo "------------------------------------------------"
echo "🔍 HRNet (CPU Mode) is READY!"
echo "------------------------------------------------"