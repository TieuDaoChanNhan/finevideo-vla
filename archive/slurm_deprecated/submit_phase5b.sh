#!/bin/bash
#SBATCH --job-name=p5b_xyzt
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=02:00:00
#SBATCH --output=logs/p5b_master_%j.log

source setup_motionbert.sh

DATA_ROOT="/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs"

mkdir -p logs/p5b_workers
mkdir -p "${DATA_ROOT}/agent_tokens_xyzt"
mkdir -p "${DATA_ROOT}/agent_xyzt_npy"

NUM_WORKERS=64
echo "🚀 Launching $NUM_WORKERS workers for Phase 5b (per-joint XYZ tokens)..."

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    SLURM_ARRAY_TASK_ID=$i SLURM_ARRAY_TASK_COUNT=$NUM_WORKERS \
    python -u pipeline/phase5b_xyzt_tokenizer.py \
        --input-dir  "${DATA_ROOT}/yolo_cleaned_30fps" \
        --output-dir "${DATA_ROOT}/agent_tokens_xyzt" \
        --npy-dir    "${DATA_ROOT}/agent_xyzt_npy" \
        --stride 8 > logs/p5b_workers/worker_${i}.log 2>&1 &
done

wait
echo "🎉 Phase 5b done — tokens in outputs/agent_tokens_xyzt/, numpy in outputs/agent_xyzt_npy/"
