#!/bin/bash
#SBATCH --job-name=merge_xyzt
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --array=1-32
#SBATCH --time=01:00:00
#SBATCH --output=logs/merge_xyzt_%a.out
#SBATCH --error=logs/merge_xyzt_%a.err

source setup_motionbert.sh

DATA="/e/data1/datasets/playground/mmlaion/shared/nguyen38/FineVideo-VLA"
AGENT_DIR="/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs/agent_tokens_xyzt"
OUT_DIR="${DATA}/final_dataset_xyzt"

mkdir -p logs "${OUT_DIR}"

echo "[Worker ${SLURM_ARRAY_TASK_ID}] Starting merge..."

python -u pipeline/merge_xyzt_tokens.py \
  --input-glob "${DATA}/training_ready_rank_*.jsonl" \
  --agent-tokens-dir "${AGENT_DIR}" \
  --output-dir "${OUT_DIR}" \
  --output-prefix "final_vla_xyzt" \
  --skip-existing

echo "[Worker ${SLURM_ARRAY_TASK_ID}] Done."
