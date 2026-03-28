#!/bin/bash
#SBATCH --job-name=vjepa_filter
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=16             # Số lượng worker chạy song song (Tùy GPU)
#SBATCH --gpus=4                # Số lượng GPU A100 cậu được cấp
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00         # V-JEPA khá nặng nên cho hẳn 12 tiếng
#SBATCH --output=logs/phase6_%A_%a.log
#SBATCH --array=0-15            # Array id từ 0 đến 15 (tổng 16 workers)

# Load môi trường (Sửa lại tên môi trường của cậu nếu cần)
source setup_vla_env.sh

# Chạy script Python
# Task ID đã được SLURM tự động truyền qua biến môi trường
echo "🚀 Khởi chạy Worker $SLURM_ARRAY_TASK_ID"

python -u phase6_semantic_filter.py \
  --input-dir outputs/clean_pose_dataset \
  --video-dir videos \
  --skeleton-dir skeletons \
  --output-dir outputs/phase6_final_dataset \
  --threshold 0.70 \
  --min-yield-rate 0.40 \
  --batch-size 8 \
  --amp