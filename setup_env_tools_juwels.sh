#!/bin/bash
# setup_env_tools_juwels.sh — Create env_tools from scratch on JUWELS.
#
# Run ONCE after deleting the old broken env:
#   rm -rf /p/data1/mmlaion/nguyen38/env_tools
#   bash setup_env_tools_juwels.sh
#
# After setup, activate in any new shell with 3 lines:
#   module --force purge
#   module load Stages/2025 GCC/13.3.0 Python/3.12.3
#   source /p/data1/mmlaion/nguyen38/env_tools/bin/activate

set -e

ENV=/p/data1/mmlaion/nguyen38/env_tools
WN_DATA=/p/data1/mmlaion/nguyen38/wn_data

echo "=== setup_env_tools_juwels ==="

# --- Step 1: Load modules (sets LD_LIBRARY_PATH for libpython3.12) ---
echo "[1/4] Loading modules..."
module --force purge
module load Stages/2025 GCC/13.3.0 Python/3.12.3
echo "Python: $(python3 --version) at $(which python3)"

# --- Step 2: Create fresh venv ---
if [ -d "$ENV" ]; then
    echo "[2/4] $ENV already exists — delete it first if you want a clean reinstall"
    echo "      rm -rf $ENV"
    exit 1
fi
echo "[2/4] Creating venv at $ENV ..."
python3 -m venv "$ENV"

# --- Step 3: Install packages ---
echo "[3/4] Installing packages..."
source "$ENV/bin/activate"
pip install --upgrade pip --quiet

pip install --no-cache-dir \
    wn \
    transformers \
    accelerate \
    huggingface-hub \
    hf-xet \
    datasets \
    scipy \
    numpy \
    tqdm \
    rich \
    typer \
    pandas \
    imageio \
    imageio-ffmpeg \
    requests \
    matplotlib

# --- Step 4: Download WordNet corpus ---
echo "[4/4] Downloading WordNet corpus (oewn:2024) → $WN_DATA ..."
mkdir -p "$WN_DATA"
WN_HOME="$WN_DATA" python -c "import wn; wn.download('oewn:2024')"

echo ""
echo "=== Done ==="
echo ""
echo "To activate in a new shell:"
echo "  module --force purge"
echo "  module load Stages/2025 GCC/13.3.0 Python/3.12.3"
echo "  source $ENV/bin/activate"
echo "  export WN_HOME=$WN_DATA"
