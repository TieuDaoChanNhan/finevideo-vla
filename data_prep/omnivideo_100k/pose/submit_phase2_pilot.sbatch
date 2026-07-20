#!/bin/bash
#SBATCH --job-name=omni100k_phase2_pilot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4       # 4 tasks = 4 GH200 GPUs
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --time=00:45:00
#SBATCH --output=/e/project1/reformo/nguyen38/3d-human-pose/logs/%j_omni100k_phase2_pilot_out.log
#SBATCH --error=/e/project1/reformo/nguyen38/3d-human-pose/logs/%j_omni100k_phase2_pilot_err.log

cd /e/project1/reformo/nguyen38/3d-human-pose

source setup_motionbert.sh   # env_motion_final -- DIFFERENT from env_hrnet_datasets_v1 (Phase 1)
                              # and env_stable_vla (Step A)

# Pilot: 4 ranks * 6 videos = first 24 videos of sports_subset_video_ids_filtered.txt
# (same 24 the Phase 1 pilot, job 976467, already reported 78.7% frame-level detection
# on). Reads from the SHARED outputs/2d_json (production Phase 1 output) and writes into
# the SHARED outputs/3d_npy -- NOT an isolated pilot-only directory, unlike Phase 1's
# pilot, since Phase 2 only needs Phase 1's *output*, not raw video, and there is no
# equivalent "unverified new format" risk to isolate against here. Run this only after
# Phase 1 (job 976705 or its resubmit) has actually written 2D json for these 24 video_ids
# -- if it prints "WAIT (2D not ready)" for any of them, Phase 1 hasn't reached them yet.
srun python -u data_prep/omnivideo_100k/pose/phase2_motionbert_omnivideo.py --limit 24
