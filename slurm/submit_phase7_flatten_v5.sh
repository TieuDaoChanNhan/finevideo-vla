#!/bin/bash
#SBATCH --job-name=phase7_v5
#SBATCH --account=laionize
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=logs/phase7_v5_%j.out
#SBATCH --error=logs/phase7_v5_%j.err

cd /p/data1/mmlaion/nguyen38/3d-human-pose
source activate_env_tools.sh

DATA="/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA"

mkdir -p logs

echo "=== Phase 7 v5 Flatten (adds <caption>/<speech> event handling) ==="
echo "Input:   ${DATA}/final_dataset_adaptive_v4/final_vla_adaptive_rank_*.jsonl"
echo "Output:  ${DATA}/megatron_dataset_v5/"

# Uses phase7_flatten.py's own v5 defaults for --input-glob/--output-dir
# (already point at final_dataset_adaptive_v4 / megatron_dataset_v5) --
# only overriding --workers to match this node's cpus-per-task.
python -u pipeline_pose/phase7_flatten.py \
    --workers 32 \
    --skip-existing

echo "=== Phase 7 v5 done ==="
