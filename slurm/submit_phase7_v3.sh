#!/bin/bash
#SBATCH --job-name=phase7_v3
#SBATCH --account=laionize
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=logs/phase7_v3.out
#SBATCH --error=logs/phase7_v3.err

cd /p/data1/mmlaion/nguyen38/3d-human-pose
source activate_env_tools.sh

DATA="/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA"
OUT_DIR="${DATA}/megatron_dataset_v3"

mkdir -p logs "${OUT_DIR}"

echo "[Phase7 v3] Starting flatten with 32 workers..."
echo "[Phase7 v3] Input:  ${DATA}/final_dataset_adaptive_v2/"
echo "[Phase7 v3] Output: ${OUT_DIR}/"

python -u pipeline_pose/phase7_flatten.py \
  --input-glob  "${DATA}/final_dataset_adaptive_v2/final_vla_adaptive_v2_rank_*.jsonl" \
  --output-dir  "${OUT_DIR}" \
  --drop_avc    1.0 \
  --drop_cosmos 0.5 \
  --drop_seed   0.0 \
  --drop_snac   0.0 \
  --workers     32 \
  --skip-existing

echo "[Phase7 v3] Done."
