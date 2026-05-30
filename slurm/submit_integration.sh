#!/bin/bash
#SBATCH --job-name=vla_merge
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2       # 2 cores: single-threaded code doesn't use more than 2
#SBATCH --array=1-32            # Increase to 1-64 or 1-128 if actual file count exceeds 160
#SBATCH --time=01:00:00         # 1 hour to guard against JSONL I/O stalls on the cluster
#SBATCH --output=logs/merge_%a.out
#SBATCH --error=logs/merge_%a.err

# Activate environment
source setup_motionbert.sh

# Create log and output directories
mkdir -p logs
mkdir -p ../prototype/FineVideo-VLA/final_dataset

echo "🎬 Starting worker $SLURM_ARRAY_TASK_ID..."

# --skip-existing avoids reprocessing if the job crashes mid-run
python -u pipeline/merge_agent_tokens.py \
  --input-glob "../prototype/FineVideo-VLA/training_ready_rank_*.jsonl" \
  --agent-tokens-dir "outputs/agent_tokens" \
  --output-prefix "final_dataset/final_vla" \
  --skip-existing

echo "🎉 Worker $SLURM_ARRAY_TASK_ID finished!"
