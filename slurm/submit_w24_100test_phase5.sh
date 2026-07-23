#!/bin/bash
#SBATCH --job-name=w24_100test_p5
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/w24_100test_phase5_%j.log

# Phase 5 (adaptive PCHIP, window=24) on the same 100-video test set as
# Phase 3/4 (job 1022278). CPU-only, no GPU needed for this phase.

source /e/project1/reformo/nguyen38/3d-human-pose/setup_motionbert.sh

cd /e/project1/reformo/nguyen38/3d-human-pose

BASE=/e/data1/datasets/playground/mmlaion/shared/nguyen38

python -u pipeline_pose/phase5_adaptive_pchip.py \
    --input-dir "$BASE/outputs/yolo_cleaned_w24_100test" \
    --output-dir "$BASE/outputs/agent_tokens_adaptive_w24_100test" \
    --window-frames 24 --stride 24

echo "DONE"
