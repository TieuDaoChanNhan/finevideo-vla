#!/bin/bash
#SBATCH --job-name=kin_packed
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1               # Single node, fully utilised
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288     # Claim all 288 cores on the node
#SBATCH --time=02:00:00
#SBATCH --output=logs/kin_packed_%j.log

# Activate environment
source setup_motionbert.sh

# Directory configuration
DIR_INPUT="outputs/3d_npy"
DIR_OUTPUT="outputs/states_jsonl"
DIR_2D_JSON="outputs/2d_json"

NUM_WORKERS=64

# IDs run 0..N-1 to match the modulo arithmetic in the Python script
MAX_ID=$((NUM_WORKERS - 1))

echo "🚀 Launching $NUM_WORKERS workers..."

for i in $(seq 0 $MAX_ID); do
    export SLURM_ARRAY_TASK_ID=$i
    export SLURM_ARRAY_TASK_COUNT=$NUM_WORKERS

    python -u pipeline/phase3_kinematics_processor.py \
        --input-dir "$DIR_INPUT" \
        --output-dir "$DIR_OUTPUT" \
        --json-2d-dir "$DIR_2D_JSON" &
done

# Wait for all 64 background processes to finish
wait

echo "🎉 All $NUM_WORKERS workers finished — 100% of data processed!"
