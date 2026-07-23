#!/bin/bash
#SBATCH --job-name=snac_omni_w24
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=256
#SBATCH --time=02:00:00
#SBATCH --output=logs/snac_omnivideo_w24_%j.log

# 2026-07-23: SNAC (listen-format) tokenization for OmniVideo-100K,
# window=24 grid (data_prep/omnivideo_100k/snac_omnivideo.py). Only
# 5,213 videos (vs FineVideo's 40,798) -- 32 workers is plenty.

module --force purge
module load Stages/2025 GCC/13.3.0 Python/3.12.3 CUDA/12 PyTorch/2.5.1 torchvision/0.20.1
source /e/project1/reformo/nguyen38/env_stable_vla/bin/activate

export HF_HOME=/e/project1/reformo/nguyen38/jupiter_cache/huggingface
export HF_HUB_OFFLINE=1

mkdir -p logs/snac_omnivideo_workers

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d
echo "NVIDIA MPS activated for all 4 GPUs"

NUM_WORKERS=32
echo "Launching $NUM_WORKERS workers distributed across 4 GH200 GPUs (SNAC listen-format, OmniVideo-100K)..."

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    GPU_ID=$(( i % 4 ))
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    SLURM_ARRAY_TASK_ID=$i \
    SLURM_ARRAY_TASK_COUNT=$NUM_WORKERS \
    python -u data_prep/omnivideo_100k/snac_omnivideo.py > logs/snac_omnivideo_workers/worker_${i}.log 2>&1 &
done

wait
echo quit | nvidia-cuda-mps-control
echo "SNAC OmniVideo-100K (listen-format, window=24) full-scale done."
