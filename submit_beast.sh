#!/bin/bash
#SBATCH --job-name=phase4_tokenizer
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=04:00:00        # Nén token có thể lâu hơn lọc text một chút, cứ cho hẳn 4h cho an toàn
#SBATCH --output=logs/phase4_tokenize_%j.log

source setup_motionbert.sh

# Nhồi 64 workers để vắt kiệt công suất node
NUM_WORKERS=64 

for i in $(seq 1 $NUM_WORKERS); do
    export SLURM_ARRAY_TASK_ID=$i 
    export SLURM_ARRAY_TASK_COUNT=$NUM_WORKERS
    
    echo "🚀 Khởi chạy Worker $i/$NUM_WORKERS cho Phase 4 (Tokenizer)"
    
    # SỬA LẠI TÊN FILE PYTHON Ở ĐÂY CHUẨN PHASE 4
    python -u phase4_interpolation_tokenizer.py & 
done

wait
echo "🎉 Toàn bộ workers đã nén Token xong!"