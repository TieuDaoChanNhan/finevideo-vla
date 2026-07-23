#!/bin/bash
#SBATCH --job-name=kin_w24_full
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=02:00:00
#SBATCH --output=logs/kin_w24_full_%j.log

# 2026-07-23: full-scale (all ~40,804 FineVideo videos) Phase 3 rerun at
# window=24, adapted from submit_kinematics.sh (window=8 production). Phase
# 1/2 (HRNet 2D, MotionBERT 3D lift) outputs are untouched by the window
# change -- only Phase 3 onward needs a rerun. New output dir (states_jsonl_w24)
# so the existing window=8 production data stays intact.

source setup_motionbert.sh

DATA_ROOT="/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs"
DIR_INPUT="${DATA_ROOT}/3d_npy_30fps"
DIR_OUTPUT="${DATA_ROOT}/states_jsonl_w24"
DIR_2D_JSON="${DATA_ROOT}/2d_json"

NUM_WORKERS=64
MAX_ID=$((NUM_WORKERS - 1))

echo "Launching $NUM_WORKERS workers (window=24, stride=24)..."

for i in $(seq 0 $MAX_ID); do
    export SLURM_ARRAY_TASK_ID=$i
    export SLURM_ARRAY_TASK_COUNT=$NUM_WORKERS

    python -u pipeline_pose/phase3_kinematics_processor.py \
        --input-dir "$DIR_INPUT" \
        --output-dir "$DIR_OUTPUT" \
        --json-2d-dir "$DIR_2D_JSON" \
        --window-size 24 --stride 24 &
done

wait
echo "Phase 3 (window=24) full-scale done."
