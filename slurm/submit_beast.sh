#!/bin/bash
#SBATCH --job-name=p5_token
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --time=02:00:00
#SBATCH --output=logs/p5_master_%j.log

source setup_motionbert.sh

mkdir -p logs/p5_workers
mkdir -p outputs/agent_tokens

NUM_WORKERS=64
echo "🚀 Launching $NUM_WORKERS workers on 288 CPU cores..."

for i in $(seq 1 $NUM_WORKERS); do
    # Pass variables directly to avoid race conditions
    SLURM_ARRAY_TASK_ID=$i SLURM_ARRAY_TASK_COUNT=$NUM_WORKERS python -u pipeline/phase5_interpolation_tokenizer.py \
        --input-dir "outputs/yolo_cleaned" \
        --output-dir "outputs/agent_tokens" \
        --stride 1 > logs/p5_workers/worker_${i}.log 2>&1 &
done

wait
echo "🎉 DONE: All agent tokens are ready!"
