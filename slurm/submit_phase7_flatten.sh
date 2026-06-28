#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --job-name=phase7_flatten_v2
#SBATCH --output=logs/phase7_flatten_%j.out
#SBATCH --error=logs/phase7_flatten_%j.err

module --force purge
module load Stages/2025 GCC/13.3.0 Python/3.12.3

source /e/project1/reformo/nguyen38/3d-human-pose/env_motion_final/bin/activate

INPUT_DATA=/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA
OUTPUT_DATA=/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA

cd /p/data1/mmlaion/nguyen38/3d-human-pose

python pipeline_pose/phase7_flatten.py \
    --input-glob "${INPUT_DATA}/final_dataset_adaptive/final_vla_adaptive_rank_*.jsonl" \
    --output-dir "${OUTPUT_DATA}/megatron_dataset_v2" \
    --drop_avc   1.0 \
    --drop_cosmos 0.5 \
    --drop_seed  0.0 \
    --workers    32 \
    --skip-existing
