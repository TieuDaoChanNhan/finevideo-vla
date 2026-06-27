#!/bin/bash
#SBATCH --job-name=phase2_5_resample
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/phase2_5_%A_%a.out
#SBATCH --error=logs/phase2_5_%A_%a.err

mkdir -p logs

source /e/project1/reformo/nguyen38/3d-human-pose/setup_motionbert.sh

cd /e/project1/reformo/nguyen38/3d-human-pose

python pipeline_pose/phase2_5_resample_30fps.py \
    --input-dir  outputs/3d_npy \
    --output-dir outputs/3d_npy_30fps \
    --fps-json   outputs/fps_lookup.json
