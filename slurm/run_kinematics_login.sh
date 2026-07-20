#!/bin/bash
# Phase 3 (kinematics) for FineVideo, run directly on the JUPITER login node
# instead of via SLURM -- use only while the booster partition is under
# maintenance and submit_kinematics.sh's job stays PENDING.
#
# Same script/args/output dirs as slurm/submit_kinematics.sh (sharding via
# SLURM_ARRAY_TASK_ID/COUNT, per-video resume via output-file existence), just
# fewer workers + nice/ionice since this is a shared login node, not a
# dedicated compute node.

source setup_motionbert.sh

DATA_ROOT="/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs"
DIR_INPUT="${DATA_ROOT}/3d_npy_30fps"
DIR_OUTPUT="${DATA_ROOT}/states_jsonl_30fps"
DIR_2D_JSON="${DATA_ROOT}/2d_json"

NUM_WORKERS=32
MAX_ID=$((NUM_WORKERS - 1))

echo "Launching $NUM_WORKERS login-node workers (nice -n 15, ionice -c3)..."

for i in $(seq 0 $MAX_ID); do
    SLURM_ARRAY_TASK_ID=$i SLURM_ARRAY_TASK_COUNT=$NUM_WORKERS \
    nice -n 15 ionice -c3 \
    python -u pipeline_pose/phase3_kinematics_processor.py \
        --input-dir "$DIR_INPUT" \
        --output-dir "$DIR_OUTPUT" \
        --json-2d-dir "$DIR_2D_JSON" \
        --stride 8 \
        > "logs/kin_login_worker_${i}.log" 2>&1 &
done

wait
echo "All $NUM_WORKERS login-node workers finished."
