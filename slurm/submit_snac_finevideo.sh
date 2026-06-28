#!/bin/bash
# submit_snac_finevideo.sh — SNAC tokenization for FineVideo-VLA activities
#
# TWO SUBMISSION MODES:
#
#   GPU mode  (fast, ~12h):  submit from juwels-booster.fz-juelich.de
#     bash slurm/submit_snac_finevideo.sh
#     Partition: booster, Account: laionize
#     Env: /p/project1/laionize/nguyen38/my_env_clean (Python 3.11, torch 2.8, snac)
#
#   CPU mode  (slow, ~24h):  submit from jwlogin*.juwels (current node)
#     bash slurm/submit_snac_finevideo.sh --cpu
#     Partition: batch, Account: laionize
#     Env: /p/data1/mmlaion/nguyen38/env_tools (Python 3.12, snac)
#     Note: GPU unavailable → SNAC runs on CPU (~5-10x realtime, ~20-24h total)
#
# STEP 1 — Build task list (run once on login node, ~5-15 min):
#   bash slurm/submit_snac_finevideo.sh --build-tasks
#
# Output: /p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/snac_tokens/
#   {video_id}_snac.jsonl — one file per video, one line per activity

set -e

REPO=/p/data1/mmlaion/nguyen38/3d-human-pose
ENV_TOOLS=/p/data1/mmlaion/nguyen38/env_tools         # x86 Stages/2025 (login + batch nodes)
ENV_BOOSTER=/p/project1/laionize/nguyen38/my_env_clean # ppc64le Stages/2024 (booster nodes)
TASK_CACHE=/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/snac_task_list.json
OUTPUT_DIR=/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/snac_tokens
VIDEO_DIR=/p/data1/mmlaion/shared/nguyen38/data/videos_staging
HF_CACHE=/p/scratch/laionize/nguyen38/hf_cache
LOG_DIR=$REPO/logs/snac_finevideo

mkdir -p "$LOG_DIR"

# ── Step 1: build task list on login node ────────────────────────────────────
if [ "$1" = "--build-tasks" ]; then
    echo "=== Building SNAC task list (scanning final_dataset_adaptive) ==="
    echo "This reads ~657 GB of JSONL — expect 5-15 min..."
    module --force purge
    module load Stages/2025 GCC/13.3.0 Python/3.12.3
    source "$ENV_TOOLS/bin/activate"
    cd "$REPO"
    python pipeline_pose/snac_finevideo.py \
        --build-tasks \
        --scan-workers 8
    echo "Task list written to: $TASK_CACHE"
    echo "Now run:  bash slurm/submit_snac_finevideo.sh        (GPU, from juwels-booster)"
    echo "      or: bash slurm/submit_snac_finevideo.sh --cpu  (CPU, from jwlogin)"
    exit 0
fi

# ── Check task list exists ────────────────────────────────────────────────────
if [ ! -f "$TASK_CACHE" ]; then
    echo "ERROR: task list not found at $TASK_CACHE"
    echo "Run first:  bash slurm/submit_snac_finevideo.sh --build-tasks"
    exit 1
fi

# ── Step 2a: GPU mode (booster) ───────────────────────────────────────────────
if [ "$1" != "--cpu" ]; then
    NUM_TASKS=16   # 4 nodes × 4 GPUs

    echo "=== Submitting SNAC GPU array job (booster) ==="
    echo "  NOTE: Submit this from juwels-booster.fz-juelich.de if it fails here."
    echo "  Tasks: 0-$((NUM_TASKS-1)) | Logs: $LOG_DIR/"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=snac_fv_gpu
#SBATCH --account=laionize
#SBATCH --partition=booster
#SBATCH --array=0-$((NUM_TASKS-1))
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=$LOG_DIR/snac_%A_%a.out
#SBATCH --error=$LOG_DIR/snac_%A_%a.err

# ── Env setup (JUWELS Booster: ppc64le, Stages/2024) ─────────────────────────
module purge
module load Stages/2024
module load GCCcore/.12.3.0
module load Python/3.11.3
source $ENV_BOOSTER/bin/activate

export HF_HOME=$HF_CACHE
export HF_HUB_OFFLINE=1   # no internet on compute nodes; use local cache only
mkdir -p "\$HF_HOME"

cd $REPO
export SLURM_ARRAY_TASK_COUNT=$NUM_TASKS

python pipeline_pose/snac_finevideo.py \
    --task-cache  $TASK_CACHE \
    --output-dir  $OUTPUT_DIR \
    --video-dir   $VIDEO_DIR \
    --hf-cache    \$HF_HOME
EOF

    exit 0
fi

# ── Step 2b: CPU mode (batch, x86, laionize account) ─────────────────────────
NUM_TASKS=32   # 32 CPU workers, ~20-24h each

echo "=== Submitting SNAC CPU array job (batch partition) ==="
echo "  Tasks: 0-$((NUM_TASKS-1)) | Logs: $LOG_DIR/"
echo "  WARNING: CPU mode is ~10x slower than GPU. Expect ~20-24h per worker."

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=snac_fv_cpu
#SBATCH --account=laionize
#SBATCH --partition=batch
#SBATCH --array=0-$((NUM_TASKS-1))
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=$LOG_DIR/snac_cpu_%A_%a.out
#SBATCH --error=$LOG_DIR/snac_cpu_%A_%a.err

# ── Env setup (JUWELS Cluster: x86, Stages/2025) ─────────────────────────────
module --force purge
module load Stages/2025 GCC/13.3.0 Python/3.12.3
source $ENV_TOOLS/bin/activate

export HF_HOME=$HF_CACHE
export HF_HUB_OFFLINE=1   # no internet on compute nodes; use local cache only
mkdir -p "\$HF_HOME"

cd $REPO
export SLURM_ARRAY_TASK_COUNT=$NUM_TASKS

python pipeline_pose/snac_finevideo.py \
    --task-cache  $TASK_CACHE \
    --output-dir  $OUTPUT_DIR \
    --video-dir   $VIDEO_DIR \
    --hf-cache    \$HF_HOME
EOF
