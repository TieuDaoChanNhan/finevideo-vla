#!/bin/bash
#SBATCH --job-name=omni100k_phase1_full
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4       # 4 tasks = 4 GH200 GPUs per node -> 32 GPU total
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --time=04:00:00
#SBATCH --output=/e/project1/reformo/nguyen38/3d-human-pose/logs/%j_omni100k_phase1_full_out.log
#SBATCH --error=/e/project1/reformo/nguyen38/3d-human-pose/logs/%j_omni100k_phase1_full_err.log

cd /e/project1/reformo/nguyen38/3d-human-pose

source setup_hrnet_gpu.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Full-scale: 32 GPU * ~35 videos/GPU (1,126 total, sports_subset_video_ids_filtered.txt
# default), measured pilot throughput ~263s/video/GPU -> ~2.6h estimated, --time=04:00:00
# for safety margin. Resume is global (skips video_ids whose output already exists across
# ALL step_a_rank_*.jsonl-equivalent 2d_json files), safe to resubmit unchanged on timeout/crash.
srun python -u data_prep/omnivideo_100k/pose/phase1_hrnet_omnivideo.py
