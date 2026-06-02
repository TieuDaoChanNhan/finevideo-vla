#!/bin/bash
#SBATCH --job-name=VLA_BigScale_160GPU
#SBATCH --nodes=40
#SBATCH --ntasks-per-node=4       # 4 Task ứng với 4 GPU Grace Hopper trên mỗi node
#SBATCH --cpus-per-task=18        # 72 nhân chia đều cho 4 task (18 cpus/task)
#SBATCH --gres=gpu:4              # Thuê trọn bộ 4 GPU mỗi node
#SBATCH --partition=booster       # Phân vùng GPU mạnh nhất của JUPITER
#SBATCH --account=reformo          # Tài khoản dự án của Khuê
#SBATCH --time=9:00:00           # Cho hẳn 9 tiếng để nhai hết dataset
#SBATCH --output=/e/project1/reformo/nguyen38/prototype/logs/%j_out.log
#SBATCH --error=/e/project1/reformo/nguyen38/prototype/logs/%j_err.log

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