#!/bin/bash
#SBATCH --job-name=omni100k_phase2_full
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4       # 4 tasks = 4 GH200 GPUs per node -> 32 GPU total
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --time=06:00:00
#SBATCH --output=/e/project1/reformo/nguyen38/3d-human-pose/logs/%j_omni100k_phase2_full_out.log
#SBATCH --error=/e/project1/reformo/nguyen38/3d-human-pose/logs/%j_omni100k_phase2_full_err.log

cd /e/project1/reformo/nguyen38/3d-human-pose

source setup_motionbert.sh

# Full-scale: 32 GPU over 1,126 videos (sports_subset_video_ids_filtered.txt default).
# --time=06:00:00 is a conservative placeholder, NOT measured yet -- unlike Phase 1's
# 04:00:00 (backed by real pilot throughput, 263s/video/GPU), no Phase 2 pilot has been
# timed at the time this script was written. Run submit_phase2_pilot.sbatch first and
# check its real per-video wall time before trusting this budget on a first submit.
# Resume is per-video (skips any video_id whose outputs/3d_npy/{id}.npy already exists),
# safe to resubmit unchanged on timeout/crash.
srun python -u data_prep/omnivideo_100k/pose/phase2_motionbert_omnivideo.py
