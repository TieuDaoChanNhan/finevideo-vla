#!/bin/bash
#SBATCH --job-name=omni100k_phase1_pilot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4       # 4 tasks = 4 GH200 GPUs
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --time=00:30:00
#SBATCH --output=/e/project1/reformo/nguyen38/3d-human-pose/logs/%j_omni100k_phase1_pilot_out.log
#SBATCH --error=/e/project1/reformo/nguyen38/3d-human-pose/logs/%j_omni100k_phase1_pilot_err.log

cd /e/project1/reformo/nguyen38/3d-human-pose

source setup_hrnet_gpu.sh   # env_hrnet_datasets_v1 -- MOI env dung cho Phase 1,
                             # KHAC env_stable_vla (Step A) va env_motion_final (Phase 2)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Pilot: 4 rank * 6 video = 24 video dau trong sports_subset_video_ids.txt,
# do throughput + ty le detect nguoi that truoc khi chay full 1,256 video.
srun python -u data_prep/omnivideo_100k/phase1_hrnet_omnivideo.py \
    --output-dir /e/data1/datasets/playground/mmlaion/shared/nguyen38/omnivideo_100k/pose_2d_json_pilot \
    --limit 24
