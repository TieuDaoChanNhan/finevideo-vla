#!/bin/bash
#SBATCH --job-name=kinematics_batch
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH --array=0-39                 # "Tổng tấn công" với 40 nodes
#SBATCH --output=logs/kin_array_%A_%a.log

# Kích hoạt môi trường (có thể dùng chung env với Phase 2 vì đều cần NumPy/SciPy)
source setup_jupiter.sh

echo "🔥 Kinematics - Task ID $SLURM_ARRAY_TASK_ID"

# Chạy với flag -u để log in ra real-time
python -u phase3_kinematics_processor.py