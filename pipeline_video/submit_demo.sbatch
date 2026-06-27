#!/bin/bash
#SBATCH --job-name=VLA_Demo_4GPU
#SBATCH --nodes=1                 # 1 node, eligible for immediate backfill scheduling
#SBATCH --ntasks=4                # 4 tasks to run all 4 GPUs simultaneously
#SBATCH --ntasks-per-node=4       # Pin all 4 tasks to this single node
#SBATCH --cpus-per-task=18        # 18 CPUs per GPU (18×4 = 72 cores = full node)
#SBATCH --gres=gpu:4              # Request all 4 GH200 GPUs on this node
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --time=00:10:00
#SBATCH --output=/e/project1/reformo/nguyen38/prototype/logs/demo_%j_out.log
#SBATCH --error=/e/project1/reformo/nguyen38/prototype/logs/demo_%j_err.log

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
