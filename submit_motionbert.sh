#!/bin/bash
#SBATCH --job-name=motionbert_3d_lift
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH --output=logs/mb_array_%A_%a.log

# Kích hoạt môi trường MotionBERT
# (Đảm bảo bạn đã có script setup_jupiter.sh kích hoạt env_motion_final)
source setup_motionbert.sh

echo "🔥 3D Lifting - Task ID $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT"

python -u phase2_motionbert_gpu.py