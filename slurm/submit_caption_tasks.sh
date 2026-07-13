#!/bin/bash
#SBATCH --job-name=caption_tasks
#SBATCH --account=laionize
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --array=1-32
#SBATCH --time=00:30:00
#SBATCH --output=logs/caption_tasks_%a.out
#SBATCH --error=logs/caption_tasks_%a.err

cd /p/data1/mmlaion/nguyen38/3d-human-pose
source activate_env_caption_test.sh

DATA="/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA"
OUT_DIR="outputs/caption_tasks"

mkdir -p logs "${OUT_DIR}"

echo "[Worker ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}] Starting A1 caption task generation..."

python -u tools/analysis/generate_caption_tasks.py \
  --input-glob "${DATA}/final_dataset_adaptive_v3/*.jsonl" \
  --output-dir "${OUT_DIR}" \
  --skip-existing

echo "[Worker ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}] Done."
