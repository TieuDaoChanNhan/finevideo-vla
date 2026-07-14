#!/bin/bash
# Login-node multi-process runner for extract_speech_segments.py.
#
# Why login node, not SLURM: the script fetches per-shard parquet from the HF
# Hub (hf_hub_download), which needs internet -- JUWELS compute nodes have
# none (HF_HUB_OFFLINE=1 there). This workload is I/O-bound (network fetch +
# text parsing), not CPU/GPU heavy, so running it directly on the shared
# login node with a modest worker count is the practical option, following
# the same pattern already used for A1's remaining 147 shards.
#
# Politeness: nice -n 15 + ionice -c3 (idle I/O class) on every worker, and a
# conservative default worker count -- this is a SHARED login node used by
# other people, be a good citizen or an admin will kill it. Reduce
# NUM_WORKERS further (or kill this script) if anyone reports the node feels
# slow.
#
# Resume: per-video output files (tools/analysis/extract_speech_segments.py
# writes {video_id}_speech.jsonl) + --skip-existing means killing this script
# and rerunning it later picks up only unfinished videos, no lost work and no
# duplicate work.
#
# Usage:
#   bash tools/analysis/run_speech_extraction_login.sh            # 8 workers (default)
#   bash tools/analysis/run_speech_extraction_login.sh 12         # override worker count

set -e

REPO=/p/data1/mmlaion/nguyen38/3d-human-pose
cd "$REPO"
source activate_env_tools.sh

# Home dir (~/.cache) has a much smaller quota than /p/data1 -- without this,
# hf_hub_download() defaults to ~/.cache/huggingface and 8 parallel workers
# downloading parquet shards blow through it in minutes ("Disk quota
# exceeded", observed in practice on the first launch of this script).
export HF_HOME=/p/data1/mmlaion/nguyen38/hf_cache
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HOME"

NUM_WORKERS="${1:-8}"
OUTPUT_DIR=/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/speech_segments
LOG_DIR=$REPO/logs/speech_extraction_login

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "=== Speech extraction (login-node run) ==="
echo "  Workers:    $NUM_WORKERS (nice -n 15, ionice -c3)"
echo "  Output:     $OUTPUT_DIR"
echo "  Logs:       $LOG_DIR/worker_*.log"
echo "  Resume:     safe to Ctrl-C / kill and rerun this script -- --skip-existing"
echo

for i in $(seq 1 "$NUM_WORKERS"); do
    SLURM_ARRAY_TASK_ID="$i" SLURM_ARRAY_TASK_COUNT="$NUM_WORKERS" \
    nice -n 15 ionice -c3 python3 -u tools/analysis/extract_speech_segments.py \
        --output-dir "$OUTPUT_DIR" \
        --skip-existing \
        > "$LOG_DIR/worker_${i}.log" 2>&1 &
    echo "  Launched worker $i/$NUM_WORKERS (PID $!) -> $LOG_DIR/worker_${i}.log"
done

echo
echo "All $NUM_WORKERS workers launched. Waiting for completion..."
wait
echo "All workers finished."
