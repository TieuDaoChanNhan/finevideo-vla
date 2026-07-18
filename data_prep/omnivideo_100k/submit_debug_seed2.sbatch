#!/bin/bash
#SBATCH --job-name=debug_seed2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:1
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --time=00:10:00
#SBATCH --output=/e/project1/reformo/nguyen38/3d-human-pose/logs/%j_debug_seed2_out.log
#SBATCH --error=/e/project1/reformo/nguyen38/3d-human-pose/logs/%j_debug_seed2_err.log

module --force purge
module load Stages/2025
module load GCC/13.3.0
module load Python/3.12.3
module load CUDA/12
module load PyTorch/2.5.1
module load torchvision/0.20.1

source /e/project1/reformo/nguyen38/env_stable_vla/bin/activate

python -u /e/project1/reformo/nguyen38/3d-human-pose/data_prep/omnivideo_100k/debug_seed2_load.py
