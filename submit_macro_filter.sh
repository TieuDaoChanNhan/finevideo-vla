#!/bin/bash
#SBATCH --job-name=macro_filter
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=01:00:00        # Lọc text thường rất nhanh, 1h là dư
#SBATCH --output=logs/macro_filter_%j.log

source setup_motionbert.sh

# Nhồi nhiều worker hơn để tận dụng 288 cores của node
NUM_WORKERS=64 

for i in $(seq 1 $NUM_WORKERS); do
    # task_id bắt đầu từ 1 để khớp với công thức (task_id - 1) trong Python
    export SLURM_ARRAY_TASK_ID=$i 
    export SLURM_ARRAY_TASK_COUNT=$NUM_WORKERS
    
    echo "🚀 Khởi chạy Worker $i/$NUM_WORKERS"
    python -u phase5_macro_filter_dataset.py & 
done

wait
echo "🎉 Toàn bộ workers đã hoàn thành!"