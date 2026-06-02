#!/bin/bash
#SBATCH --job-name=VLA_Demo_4GPU
#SBATCH --nodes=1                 # Vẫn dùng 1 node để được chạy ngay (Backfilling)
#SBATCH --ntasks=4                # Tăng lên 4 task để chạy 4 GPU cùng lúc
#SBATCH --ntasks-per-node=4       # Ép 4 task chạy trên cùng 1 node này
#SBATCH --cpus-per-task=18        # Giữ nguyên 18 CPU/GPU (18*4 = 72 nhân - Full node)
#SBATCH --gres=gpu:4              # Thuê trọn bộ 4 con GH200 của node này
#SBATCH --partition=booster       
#SBATCH --account=reformo
#SBATCH --time=00:10:00           
#SBATCH --output=/e/project1/reformo/nguyen38/prototype/logs/demo_%j_out.log
#SBATCH --error=/e/project1/reformo/nguyen38/prototype/logs/demo_%j_err.log

# --- 1. Load môi trường HPC ---
module --force purge
module load Stages/2025
module load GCC/13.3.0
module load Python/3.12.3
module load CUDA/12
module load PyTorch/2.5.1
module load torchvision/0.20.1

# --- 2. Kích hoạt môi trường ảo ---
source /e/project1/reformo/nguyen38/env_stable_vla/bin/activate
export FFMPEG_PATH=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- 3. Chạy code với srun ---
srun python -u /e/project1/reformo/nguyen38/prototype/pipeline.py