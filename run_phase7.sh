#!/bin/bash
# run_phase7.sh — Re-flatten FineVideo-VLA with v2 dropout settings
#
# Dropout v2: AVC-LM 100% drop, Cosmos 50% drop, Seed2 0% drop
# Input:  /p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/final_dataset_adaptive/
# Output: /p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/megatron_dataset_v2/
#
# Usage (in tmux on JUWELS login node):
#   bash run_phase7.sh
#
# If wn/oewn:2024 not yet downloaded, run once first:
#   bash run_phase7.sh --setup
#
# Env: env_tools at /p/data1/mmlaion/nguyen38/env_tools
#   (Python 3.12.3 via Stages/2025 modules + venv with wn, transformers, torch, etc.)

set -e

REPO=/p/data1/mmlaion/nguyen38/3d-human-pose
ENV=/p/data1/mmlaion/nguyen38/env_tools
WN_DATA=/p/data1/mmlaion/nguyen38/wn_data

# --- Load modules (needed to resolve env_tools Python symlink) ---
module --force purge
module load Stages/2025 GCC/13.3.0 Python/3.12.3

source "$ENV/bin/activate"

# Keep WordNet data on /p/ to avoid home quota issues
export WN_HOME="$WN_DATA"

# --- One-time setup: download WordNet corpus ---
if [ "$1" = "--setup" ]; then
    echo "Downloading oewn:2024 WordNet corpus → $WN_DATA ..."
    mkdir -p "$WN_DATA"
    python -c "import wn; wn.download('oewn:2024')"
    echo "Done. Re-run without --setup to flatten."
    exit 0
fi

# --- Run flatten ---
cd "$REPO"
echo "=== Phase 7 Flatten v2 ==="
echo "Input:   /p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/final_dataset_adaptive/"
echo "Output:  /p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/megatron_dataset_v2/"
echo "Dropout: AVC-LM=100% drop | Cosmos=50% drop | Seed2=keep all"
echo "Workers: 16"
echo ""

python pipeline_pose/phase7_flatten.py \
    --workers 16 \
    --skip-existing
