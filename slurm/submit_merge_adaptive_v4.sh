#!/bin/bash
#SBATCH --job-name=merge_v4
#SBATCH --account=laionize
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --array=1-32
#SBATCH --time=03:00:00
#SBATCH --output=logs/merge_v4_%a.out
#SBATCH --error=logs/merge_v4_%a.err

cd /p/data1/mmlaion/nguyen38/3d-human-pose
source activate_env_tools.sh

DATA="/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA"
AGENT_DIR="/p/data1/mmlaion/shared/nguyen38/data/outputs/agent_tokens_adaptive"
SNAC_DIR="${DATA}/snac_tokens"
CAPTIONS_DIR="${DATA}/captions_dict"
SPEECH_DIR="${DATA}/speech_segments"
OUT_DIR="${DATA}/final_dataset_adaptive_v4"

mkdir -p logs "${OUT_DIR}"

echo "[Worker ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}] Starting merge v4 (inject caption+speech language anchors on top of v3)..."

# Input is v3 (already has <agent>/<snac> injected). agent-tokens-dir/snac-tokens-dir
# are still passed so chunk_timing's has_agent/has_snac stay accurate, but
# phase6_merge_adaptive.py's idempotency guard (checks for existing tags in
# video_tokens) prevents re-injecting them a second time -- only captions-dir
# and speech-segments-dir add new content this run.
python -u pipeline_pose/phase6_merge_adaptive.py \
  --input-glob "${DATA}/final_dataset_adaptive_v3/final_vla_adaptive_v3_rank_*.jsonl" \
  --agent-tokens-dir    "${AGENT_DIR}" \
  --snac-tokens-dir     "${SNAC_DIR}" \
  --captions-dir        "${CAPTIONS_DIR}" \
  --speech-segments-dir "${SPEECH_DIR}" \
  --output-dir          "${OUT_DIR}" \
  --output-prefix       "final_vla_adaptive" \
  --skip-existing

echo "[Worker ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}] Done."
