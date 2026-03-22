#!/bin/bash
#SBATCH --job-name=hrnet_2d_extraction
#SBATCH --partition=booster              
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                    
#SBATCH --cpus-per-task=12              
#SBATCH --time=1:00:00            
#SBATCH --array=0-39     
#SBATCH --output=logs/array_%A_%a.log

source setup_hrnet_gpu.sh
echo "🔥 I am Task ID $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT"

mkdir -p logs
mkdir -p outputs/2d_keypoints

export TORCH_CUDA_ARCH_LIST="9.0"
export FORCE_CUDA=1

echo "🚀 Starting Batch Processing for 100 videos..."
python -u phase1_hrnet_gpu.py