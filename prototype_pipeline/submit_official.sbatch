#!/bin/bash
#SBATCH --job-name=VLA_BigScale_160GPU
#SBATCH --nodes=40
#SBATCH --ntasks-per-node=4       # 4 tasks = 4 GH200 GPUs per node
#SBATCH --cpus-per-task=18        # 72 cores split evenly across 4 tasks (18 cpus/task)
#SBATCH --gres=gpu:4              # Request all 4 GPUs per node
#SBATCH --partition=booster       # Highest-performance GPU partition on JUPITER
#SBATCH --account=reformo         # Project account
#SBATCH --time=9:00:00            # 9 hours to process the full dataset
#SBATCH --output=/e/project1/reformo/nguyen38/prototype/logs/%j_out.log
#SBATCH --error=/e/project1/reformo/nguyen38/prototype/logs/%j_err.log

# --- 1. Load HPC modules ---
module --force purge
module load Stages/2025
module load GCC/13.3.0
module load Python/3.12.3
module load CUDA/12
module load PyTorch/2.5.1
module load torchvision/0.20.1

# --- 2. Activate virtual environment ---
source /e/project1/reformo/nguyen38/env_stable_vla/bin/activate
export FFMPEG_PATH=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- 3. Launch with srun ---
srun python -u /e/project1/reformo/nguyen38/prototype/pipeline.py
