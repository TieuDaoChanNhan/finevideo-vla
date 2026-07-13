#!/bin/bash
# submit_caption_finevideo.sh — A2: caption generation for FineVideo-VLA
#
# CPU-only (Van Khue decision, 12/07/2026): many CPU cores > the 2x4090 test
# machine for this workload. Mirrors slurm/submit_snac_finevideo.sh's CPU mode.
#
#   bash slurm/submit_caption_finevideo.sh
#     Partition: batch, Account: laionize
#     Env: /p/data1/mmlaion/nguyen38/env_caption_test (Python 3.12, transformers, cv2)
#
# Prerequisite: A1 task list already built at outputs/caption_tasks/*.jsonl
#   (tools/analysis/generate_caption_tasks.py -- already run, 912,998 task points
#   across 372,385 activities as of 12/07/2026).
#
# Output: outputs/captions/{video_id}_captions.jsonl — one file per video,
#   one line per anchor point (see pipeline_pose/caption_finevideo.py header).
#
# Cost note: measured ~10-15s/caption on CPU (single request, prototype test).
# 912,998 total task points / 32 workers ~= 28,500/worker * ~12s =~ 95h/worker
# in the worst case -- almost certainly will NOT finish within one --time
# window. Safe to re-submit: per-video output file + --skip-existing (default)
# means a re-run only picks up unfinished videos.
#
# Chaining: optional $1 = jobid to depend on (afterany -- starts once that
# job's whole array exits, success or failure/timeout, so the chain always
# keeps moving). Prints the new job id on its own line for easy chaining, e.g.:
#   j=$(bash slurm/submit_caption_finevideo.sh | tail -1)
#   j=$(bash slurm/submit_caption_finevideo.sh "$j" | tail -1)

set -e

DEPENDENCY="${1:-}"

REPO=/p/data1/mmlaion/nguyen38/3d-human-pose
ENV_CAPTION=/p/data1/mmlaion/nguyen38/env_caption_test
HF_CACHE=/p/data1/mmlaion/nguyen38/hf_cache   # NOT /p/scratch/laionize/... -- that dir lacks the cached Qwen2.5-VL model
OUTPUT_DIR=/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/captions
LOG_DIR=$REPO/logs/caption_finevideo

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

NUM_TASKS=32   # 32 CPU workers, see cost note above -- expect multiple re-submits

echo "=== Submitting A2 caption CPU array job (batch partition) ===" >&2
echo "  Tasks: 0-$((NUM_TASKS-1)) | Logs: $LOG_DIR/" >&2
echo "  Output: $OUTPUT_DIR" >&2
if [ -n "$DEPENDENCY" ]; then echo "  Dependency: afterany:$DEPENDENCY" >&2; fi
echo "  WARNING: unlikely to finish in one 24h window -- re-submit this script to resume." >&2

DEP_ARG=""
if [ -n "$DEPENDENCY" ]; then DEP_ARG="--dependency=afterany:$DEPENDENCY"; fi

JOBID=$(sbatch --parsable $DEP_ARG <<EOF
#!/bin/bash
#SBATCH --job-name=caption_fv
#SBATCH --account=laionize
#SBATCH --partition=batch
#SBATCH --array=0-$((NUM_TASKS-1))
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=$LOG_DIR/caption_%A_%a.out
#SBATCH --error=$LOG_DIR/caption_%A_%a.err

module --force purge
module load Stages/2025 GCC/13.3.0 Python/3.12.3
source $ENV_CAPTION/bin/activate

export HF_HOME=$HF_CACHE
export HF_HUB_OFFLINE=1   # no internet on compute nodes; use local cache only
mkdir -p "\$HF_HOME"

cd $REPO
export SLURM_ARRAY_TASK_COUNT=$NUM_TASKS

python pipeline_pose/caption_finevideo.py \
    --output-dir "$OUTPUT_DIR" \
    --hf-cache   "\$HF_HOME"
EOF
)

echo "Submitted batch job $JOBID" >&2
echo "$JOBID"
