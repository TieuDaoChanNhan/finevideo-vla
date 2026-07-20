#!/bin/bash
#SBATCH --job-name=omni100k_phase4_full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4       # 4 tasks = 4 GH200 GPUs -> 4 GPU total
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --time=01:00:00
#SBATCH --output=/e/project1/reformo/nguyen38/3d-human-pose/logs/%j_omni100k_phase4_full_out.log
#SBATCH --error=/e/project1/reformo/nguyen38/3d-human-pose/logs/%j_omni100k_phase4_full_err.log

cd /e/project1/reformo/nguyen38/3d-human-pose

source setup_motionbert.sh

# 928/1,126 videos have valid Phase 3 output (198 produced 0 windows -- too
# noisy/occluded, not an error, see Phase 3 investigation notes). This driver
# skips missing input gracefully, so passing the full 1,126-video list is safe.
# --time=01:00:00 is a placeholder based on the smoke-test rate (~5s/video on
# 1 warm GPU, so 1 node x 4 GPU over 928 videos should be well under an hour) --
# not yet confirmed at full scale, resubmit unchanged if it times out (resume
# is per-video, safe).
srun python -u data_prep/omnivideo_100k/pose/phase4_yolo_cleaner_omnivideo.py
