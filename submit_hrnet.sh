#!/bin/bash
#SBATCH --job-name=hrnet_2d_extraction
#SBATCH --partition=booster              # Partition dành cho H100 trên JUPITER
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                    # Yêu cầu 1 card H100
#SBATCH --cpus-per-task=12              # Grace CPU hỗ trợ xử lý ảnh song song
#SBATCH --time=1:00:00                 # Đặt 12 tiếng cho dư dả (8h chạy + 4h dự phòng)
#SBATCH --output=logs/array_%A_%a.log

# 1. Load các biến môi trường từ file setup bạn đã làm
source setup_hrnet_gpu.sh
echo "🔥 I am Task ID $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT"

# 2. Tạo thư mục log nếu chưa có
mkdir -p logs
mkdir -p outputs/2d_keypoints

# 3. Ép xung biên dịch (phòng trường hợp mmcv cần compile lại)
export TORCH_CUDA_ARCH_LIST="9.0"
export FORCE_CUDA=1

# 4. Chạy script xử lý batch
echo "🚀 Starting Batch Processing for 100 videos..."
python -u phase1_hrnet_gpu.py