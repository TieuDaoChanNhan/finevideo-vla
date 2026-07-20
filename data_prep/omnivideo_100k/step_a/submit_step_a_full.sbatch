#!/bin/bash
#SBATCH --job-name=omni100k_stepA_full
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4       # 4 tasks = 4 GH200 GPUs per node
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --time=05:00:00
#SBATCH --output=/e/project1/reformo/nguyen38/3d-human-pose/logs/%j_omni100k_stepA_full_out.log
#SBATCH --error=/e/project1/reformo/nguyen38/3d-human-pose/logs/%j_omni100k_stepA_full_err.log

module --force purge
module load Stages/2025
module load GCC/13.3.0
module load Python/3.12.3
module load CUDA/12
module load PyTorch/2.5.1
module load torchvision/0.20.1

source /e/project1/reformo/nguyen38/env_stable_vla/bin/activate
export FFMPEG_PATH=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Full-scale: 32 rank * ~163 video = 5,214 video. ~4.3h ước tính từ pilot
# (93.8s/video/GPU, xem PROGRESS_VI.md). Có resume — an toàn để submit lại
# nguyên lệnh này nếu job bị timeout/crash giữa chừng, video đã xong sẽ skip.
srun python -u /e/project1/reformo/nguyen38/3d-human-pose/data_prep/omnivideo_100k/step_a/step_a_tokenize_video.py
