#!/bin/bash
#SBATCH --job-name=omni_p3_w24
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/omni_p3_w24_%j.log

# 2026-07-23: OmniVideo-100K Phase 3 (kinematics), window=24 to match
# FineVideo-VLA's pivot (was 8). CPU-only, reuses existing Phase 1/2
# (pose_2d_json, pose_3d_npy_30fps -- window-independent, not rerun).
# Sports subset only (1,126 videos).

source /e/project1/reformo/nguyen38/3d-human-pose/setup_motionbert.sh
cd /e/project1/reformo/nguyen38/3d-human-pose

srun python -u data_prep/omnivideo_100k/pose/phase3_kinematics_omnivideo.py

echo "OmniVideo-100K Phase 3 (window=24) done."
