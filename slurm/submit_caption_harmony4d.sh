#!/bin/bash
#SBATCH --job-name=caption_h4d
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/caption_harmony4d_%j.log

# 2026-07-23: Harmony4D per-sequence captioning, same model/method as
# pipeline_pose/caption_finevideo.py (Qwen2.5-VL-3B-Instruct) but on
# JUPITER + env_stable_vla (the original caption_finevideo.py pipeline runs
# on JUWELS + a /p-only env, not accessible here) and on GPU (only 208
# sequences total, not FineVideo's 912,998 anchor points -- GPU finishes
# this in minutes, no need for the CPU-array-worker approach used there).

module --force purge
module load Stages/2025 GCC/13.3.0 Python/3.12.3 CUDA/12 PyTorch/2.5.1 torchvision/0.20.1
source /e/project1/reformo/nguyen38/env_stable_vla/bin/activate

# Compute nodes have no internet -- model must already be cached here from
# a login-node download (see logs/download_qwen25vl.log).
export HF_HOME=/e/project1/reformo/nguyen38/jupiter_cache/huggingface
export HF_HUB_OFFLINE=1

cd /e/project1/reformo/nguyen38/3d-human-pose
CUDA_LAUNCH_BLOCKING=1 python -u data_prep/harmony4d/caption_harmony4d.py

echo "DONE"
