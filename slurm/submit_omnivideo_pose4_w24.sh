#!/bin/bash
#SBATCH --job-name=omni_p4_w24
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=logs/omni_p4_w24_%j.log

# 2026-07-23: OmniVideo-100K Phase 4 (YOLO person-presence cleaning),
# window=24 to match FineVideo-VLA's pivot (was 8). One task per GPU
# (SLURM_LOCALID binds device). Sports subset only (1,126 videos) --
# should finish comfortably inside 2h given FineVideo's 40K-video Phase 4
# took ~4h on a similar single-node setup.

source /e/project1/reformo/nguyen38/3d-human-pose/setup_motionbert.sh
cd /e/project1/reformo/nguyen38/3d-human-pose

srun python -u data_prep/omnivideo_100k/pose/phase4_yolo_cleaner_omnivideo.py

echo "OmniVideo-100K Phase 4 (window=24) done."
