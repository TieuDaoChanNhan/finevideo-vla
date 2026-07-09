#!/bin/bash
#SBATCH --job-name=p5_resume
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --time=04:00:00
#SBATCH --output=logs/p5_resume_%j.log

source setup_motionbert.sh

mkdir -p logs/p5_workers
mkdir -p outputs/agent_tokens

INPUT_DIR="outputs/yolo_cleaned"
OUTPUT_DIR="outputs/agent_tokens"
MISSING_LIST="missing_phase5.txt"

# ── Step 1: build list of non-empty yolo_cleaned files with no agent_tokens output ──
echo "🔍 Scanning for missing agent token files..."
python3 - <<'PYEOF'
import glob, os, sys

input_dir  = "outputs/yolo_cleaned"
output_dir = "outputs/agent_tokens"
missing    = []

for src in sorted(glob.glob(os.path.join(input_dir, "*_cleaned.jsonl"))):
    if os.path.getsize(src) == 0:          # empty = YOLO filtered everything, skip
        continue
    vid = os.path.basename(src).replace("_cleaned.jsonl", "")
    out = os.path.join(output_dir, f"{vid}_tokens.jsonl")
    if not os.path.exists(out):
        missing.append(src)

with open("missing_phase5.txt", "w") as f:
    f.write("\n".join(missing) + ("\n" if missing else ""))

print(f"✅ Found {len(missing)} files that still need Phase 5.")
sys.exit(0 if missing else 99)
PYEOF

EXIT_CODE=$?
if [ $EXIT_CODE -eq 99 ]; then
    echo "🎉 Nothing to do — all videos already have agent tokens!"
    exit 0
fi

TOTAL=$(wc -l < "$MISSING_LIST")
echo "📋 $TOTAL files queued. Launching workers..."

# ── Step 2: fan out to workers ────────────────────────────────────────────────
NUM_WORKERS=64

for i in $(seq 1 $NUM_WORKERS); do
    SLURM_ARRAY_TASK_ID=$i \
    SLURM_ARRAY_TASK_COUNT=$NUM_WORKERS \
    python -u phase5_interpolation_tokenizer_fixed.py \
        --input-dir  "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --file-list  "$MISSING_LIST" \
        --stride 1 \
        > logs/p5_workers/resume_worker_${i}.log 2>&1 &
done

wait
echo "🎉 All workers finished. Check logs/p5_workers/ for per-worker output."
