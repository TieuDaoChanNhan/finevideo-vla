#!/bin/bash
#SBATCH --job-name=w24_100test
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=logs/w24_100test_%j.log

# 2026-07-23: small-scale (100 video) rerun of Phase 3+4 at window=24, on a
# compute node instead of the login node (previous attempt died on the login
# node / an ephemeral background job with no durable log -- see PROGRESS_VI.md
# checkpoint). Deliberately scoped to a fixed 100-video staging dir (symlinks
# under outputs/_w24_100test_staging/), not the full 40K corpus.

# Phase 3/4 belong to the 3D pose pipeline env, NOT env_stable_vla -- these
# two envs must never be mixed in the same shell (CLAUDE.md). Use only
# setup_motionbert.sh's own module purge/load + conda activate.
source /e/project1/reformo/nguyen38/3d-human-pose/setup_motionbert.sh

cd /e/project1/reformo/nguyen38/3d-human-pose

BASE=/e/data1/datasets/playground/mmlaion/shared/nguyen38
STAGE=$BASE/outputs/_w24_100test_staging

echo "=== Phase 3 (window=24) on 100-video staging set ==="
python -u pipeline_pose/phase3_kinematics_processor.py \
    --input-dir "$STAGE/npy_in" \
    --json-2d-dir "$STAGE/2d_json" \
    --output-dir "$BASE/outputs/states_jsonl_w24_100test" \
    --window-size 24 --stride 24

echo "=== Phase 4 (window=24) on same 100-video set ==="
python -u pipeline_pose/phase4_yolo_cleaner.py \
    --videos-dir "$STAGE/videos" \
    --input-dir "$BASE/outputs/states_jsonl_w24_100test" \
    --resampled-npy-dir "$STAGE/npy_in" \
    --output-dir "$BASE/outputs/yolo_cleaned_w24_100test" \
    --model "/e/project1/reformo/nguyen38/3d-human-pose/yolo26n.pt" \
    --window-size 24 \
    --batch-size 128

echo "DONE"
