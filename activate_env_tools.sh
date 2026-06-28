module --force purge
module load Stages/2025 GCC/13.3.0 Python/3.12.3
source /p/data1/mmlaion/nguyen38/env_tools/bin/activate
export WN_HOME=/p/data1/mmlaion/nguyen38/wn_data
echo "env_tools ready | $(python --version)"
