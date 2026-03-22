#!/bin/bash
#SBATCH --job-name=kinematics_batch
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH --array=0-39           
#SBATCH --output=logs/kin_array_%A_%a.log

source setup_jupiter.sh

echo "🔥 Kinematics - Task ID $SLURM_ARRAY_TASK_ID"

python -u phase3_kinematics_processor.py