#!/bin/bash
#SBATCH --job-name=p5_adapt
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=02:00:00
#SBATCH --output=logs/p5_adaptive_master_%j.log

source setup_motionbert.sh

DATA_ROOT="/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs"

mkdir -p logs/p5_adaptive_workers
mkdir -p "${DATA_ROOT}/agent_tokens_adaptive"

NUM_WORKERS=64
echo "Launching $NUM_WORKERS workers for Phase 5 (Adaptive PCHIP per-joint)..."

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    SLURM_ARRAY_TASK_ID=$i SLURM_ARRAY_TASK_COUNT=$NUM_WORKERS \
    python -u pipeline_pose/phase5_adaptive_pchip.py \
        --input-dir  "${DATA_ROOT}/yolo_cleaned_30fps" \
        --output-dir "${DATA_ROOT}/agent_tokens_adaptive" \
        --stride 8 \
        --tau-low  0.005 \
        --tau-high 0.05 > logs/p5_adaptive_workers/worker_${i}.log 2>&1 &
done

wait
echo "Phase 5 adaptive done — tokens in ${DATA_ROOT}/agent_tokens_adaptive/"
