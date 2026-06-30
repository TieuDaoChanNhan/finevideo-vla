#!/bin/bash
# Create a minimal venv for data_inventory.py on JUWELS.
# Run once: bash tools/setup_env_inventory.sh

set -e
ENV=/p/data1/mmlaion/nguyen38/env_inventory

echo "Creating venv at $ENV ..."
python3 -m venv "$ENV"
source "$ENV/bin/activate"

echo "Installing packages..."
pip install --upgrade pip --quiet
pip install requests matplotlib --quiet

echo ""
echo "Verifying..."
python -c "import requests, matplotlib; print('requests', requests.__version__); print('matplotlib', matplotlib.__version__)"
echo ""
echo "Done. Activate with:"
echo "  source $ENV/bin/activate"
