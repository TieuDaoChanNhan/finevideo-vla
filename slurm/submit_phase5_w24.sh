#!/bin/bash
#SBATCH --job-name=p5_w24_full
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=02:00:00
#SBATCH --output=logs/p5_w24_full_%j.log

# 2026-07-23: full-scale Phase 5 rerun at window=24 (adaptive PCHIP, now
# with the 272 extended t_8..t_23 tokens since control points can land
# anywhere in the 24-frame window). Adapted from submit_phase5_adaptive.sh.

source setup_motionbert.sh

DATA_ROOT="/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs"

mkdir -p logs/p5_w24_workers
mkdir -p "${DATA_ROOT}/agent_tokens_adaptive_w24"

NUM_WORKERS=64
echo "Launching $NUM_WORKERS workers for Phase 5 (window=24)..."

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    SLURM_ARRAY_TASK_ID=$i SLURM_ARRAY_TASK_COUNT=$NUM_WORKERS \
    python -u pipeline_pose/phase5_adaptive_pchip.py \
        --input-dir  "${DATA_ROOT}/yolo_cleaned_w24" \
        --output-dir "${DATA_ROOT}/agent_tokens_adaptive_w24" \
        --window-frames 24 --stride 24 \
        --tau-low  0.005 \
        --tau-high 0.05 > logs/p5_w24_workers/worker_${i}.log 2>&1 &
done

wait
echo "Phase 5 (window=24) full-scale done -- tokens in ${DATA_ROOT}/agent_tokens_adaptive_w24/"
