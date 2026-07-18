#!/bin/bash
#SBATCH --job-name=mint1t_text
#SBATCH --account=laionize
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --array=1-32
#SBATCH --time=06:00:00
#SBATCH --output=logs/mint1t_text_%a.out
#SBATCH --error=logs/mint1t_text_%a.err

cd /p/data1/mmlaion/nguyen38/3d-human-pose
source activate_env_tools.sh

mkdir -p logs

echo "[Worker ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}] Converting MINT-1T-HTML parquet -> flat text JSONL..."

python -u tools/extract/convert_mint1t_text_jsonl.py

echo "[Worker ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}] Done."
