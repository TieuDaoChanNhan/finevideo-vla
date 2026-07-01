#!/bin/bash
#SBATCH --job-name=merge_v2
#SBATCH --account=laionize
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --array=1-32
#SBATCH --time=02:00:00
#SBATCH --output=logs/merge_v2_%a.out
#SBATCH --error=logs/merge_v2_%a.err

cd /p/data1/mmlaion/nguyen38/3d-human-pose
source activate_env_tools.sh

DATA="/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA"
AGENT_DIR="/p/data1/mmlaion/shared/nguyen38/data/outputs/agent_tokens_adaptive"
SNAC_DIR="${DATA}/snac_tokens"
OUT_DIR="${DATA}/final_dataset_adaptive_v2"

mkdir -p logs "${OUT_DIR}"

echo "[Worker ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}] Starting merge v2 (agent + snac)..."

python -u pipeline_pose/phase6_merge_adaptive.py \
  --input-glob "${DATA}/training_ready_rank_*.jsonl" \
  --agent-tokens-dir "${AGENT_DIR}" \
  --snac-tokens-dir  "${SNAC_DIR}" \
  --output-dir       "${OUT_DIR}" \
  --output-prefix    "final_vla_adaptive_v2" \
  --skip-existing

echo "[Worker ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}] Done."
