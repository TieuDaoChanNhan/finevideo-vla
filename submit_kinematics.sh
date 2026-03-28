#!/bin/bash
#SBATCH --job-name=kin_packed
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1               # Chỉ xin 1 node duy nhất (nhưng dùng hết công suất)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288     # Chiếm trọn 288 cores của node
#SBATCH --time=02:00:00
#SBATCH --output=logs/kin_packed_%j.log

source setup_motionbert.sh

# Đồng bộ con số này
NUM_WORKERS=64 

# Dùng seq 1 để khớp với logic (task_id - 1) trong Python
for i in $(seq 1 $NUM_WORKERS); do
    export SLURM_ARRAY_TASK_ID=$i 
    export SLURM_ARRAY_TASK_COUNT=$NUM_WORKERS
    
    echo "🚀 Khởi chạy Worker $i trên $(hostname)"
    python -u phase3_kinematics_processor.py & 
done

wait
echo "🎉 Toàn bộ 32 workers đã quét xong 100% dữ liệu!"