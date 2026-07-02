#!/bin/bash
#SBATCH --job-name=phase7_v4
#SBATCH --account=laionize
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/phase7_v4.out
#SBATCH --error=logs/phase7_v4.err

# Phase 7 v4 — Per-chunk temporal flatten
# Changes from v3:
#   - Per-chunk ordering: [seed2?][cosmos?][agent?][snac?] per 8-frame chunk
#   - Speech in ### Speech: header block, not scattered into token sequence
#   - Text headers shuffled, token sequence always last

set -euo pipefail

cd /p/data1/mmlaion/nguyen38/3d-human-pose
source activate_env_tools.sh

DATA="/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA"
INPUT_DIR="${DATA}/final_dataset_adaptive_v2"
OUT_DIR="${DATA}/megatron_dataset_v4"

mkdir -p logs "${OUT_DIR}"

echo "[Phase7 v4] $(date)"
echo "[Phase7 v4] Input:   ${INPUT_DIR}/"
echo "[Phase7 v4] Output:  ${OUT_DIR}/"
echo "[Phase7 v4] Workers: 32"

python -u pipeline_pose/phase7_flatten.py \
  --input-glob  "${INPUT_DIR}/final_vla_adaptive_v2_rank_*.jsonl" \
  --output-dir  "${OUT_DIR}" \
  --drop_avc    1.0 \
  --drop_cosmos 0.5 \
  --drop_seed   0.0 \
  --drop_snac   0.0 \
  --workers     32 \
  --skip-existing

echo "[Phase7 v4] Done at $(date)"
