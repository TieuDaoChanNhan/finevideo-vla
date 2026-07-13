#!/usr/bin/env python3
"""
A2 -- caption generation for FineVideo-VLA (CPU captioning with Qwen2.5-VL-3B).

Reads A1's task list (tools/analysis/generate_caption_tasks.py output,
outputs/caption_tasks/*.jsonl -- one task per anchor point: video_id,
video_path, scene_id, activity_id, chunk_idx, start_sec, has_agent),
extracts the frame at start_sec from videos_staging/{video_id}.mp4, and
captions it with Qwen2.5-VL-3B-Instruct (chosen model, see
tools/analysis/caption_prototype.py / caption_model_compare.py).

Output: {output_dir}/{video_id}_captions.jsonl -- one line per task point:
    {"video_id":..., "scene_id":..., "activity_id":..., "chunk_idx":...,
     "start_sec":..., "has_agent":..., "caption": "..."}

Follows the same worker-split / resume pattern as pipeline_pose/snac_finevideo.py:
  - Model loaded once per SLURM array worker (not per video)
  - Videos striped across workers: all_vids[task_id::num_tasks]
  - Per-video output file checked for skip/resume (safe to re-submit)
  - Progress logged periodically with rate/ETA

SLURM usage:
    SLURM_ARRAY_TASK_ID    = task index (0-based)
    SLURM_ARRAY_TASK_COUNT = total number of tasks in the array

Local smoke test (no SLURM env vars -> task_id=0, num_tasks=1):
    python pipeline_pose/caption_finevideo.py --max-videos 2
"""

import argparse
import glob
import json
import logging
import os
import sys
import time

# Must run before torch/transformers are imported (lazily, inside load_model()
# in caption_prototype.py) -- otherwise OMP/MKL thread pools default to using
# every core visible on the node. On a shared SLURM allocation with
# --cpus-per-task=N, an unconstrained worker will still see the *node's*
# full core count and oversubscribe against the other array workers sharing
# that node, causing severe contention (observed directly: two concurrent
# unconstrained CPU inference runs on an 80-core login node were ~4x slower
# per-caption than a single unconstrained run).
_NUM_THREADS = os.environ.get("SLURM_CPUS_PER_TASK", "4")
os.environ.setdefault("OMP_NUM_THREADS", _NUM_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _NUM_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NUM_THREADS)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "analysis"))

TASKS_GLOB_DEFAULT = os.path.join(REPO_ROOT, "outputs", "caption_tasks", "*.jsonl")
VIDEO_DIR_DEFAULT = "/p/data1/mmlaion/shared/nguyen38/data/videos_staging"
OUTPUT_DIR_DEFAULT = os.path.join(REPO_ROOT, "outputs", "captions")
HF_CACHE_DEFAULT = "/p/scratch/laionize/nguyen38/hf_cache"


def load_all_tasks(tasks_glob: str) -> dict:
    """Read all A1 shard task files, group by video_id -> list of task dicts."""
    tasks_by_video: dict = {}
    files = sorted(glob.glob(tasks_glob))
    if not files:
        raise FileNotFoundError(f"No task files matched: {tasks_glob!r}")
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = json.loads(line)
                tasks_by_video.setdefault(t["video_id"], []).append(t)
    return tasks_by_video


def process_video(video_id, tasks, model, processor, output_dir, skip_existing):
    """Caption every anchor point for one video. Returns stats dict."""
    out_path = os.path.join(output_dir, f"{video_id}_captions.jsonl")
    if skip_existing and os.path.exists(out_path):
        return {"ok": 0, "skipped_vid": len(tasks), "failed": 0}

    video_path = tasks[0]["video_path"]
    if not os.path.exists(video_path):
        return {"ok": 0, "skipped_vid": 0, "failed": len(tasks)}

    from caption_prototype import extract_frame, caption_frame

    stats = {"ok": 0, "skipped_vid": 0, "failed": 0}
    rows = []
    for t in tasks:
        try:
            frame = extract_frame(video_path, t["start_sec"])
            caption, _gen_time = caption_frame(model, processor, frame)
        except Exception as e:
            log.warning(f"Caption failed {video_id}@{t['start_sec']}s: {e}")
            stats["failed"] += 1
            continue
        rows.append({
            "video_id": video_id,
            "scene_id": t["scene_id"],
            "activity_id": t["activity_id"],
            "chunk_idx": t["chunk_idx"],
            "start_sec": t["start_sec"],
            "has_agent": t["has_agent"],
            "caption": caption,
        })
        stats["ok"] += 1

    if rows:
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A2: caption FineVideo keyframes with Qwen2.5-VL-3B (CPU).")
    p.add_argument("--tasks-glob", default=TASKS_GLOB_DEFAULT,
                    help="Glob for A1 task-list shard files.")
    p.add_argument("--video-dir", default=VIDEO_DIR_DEFAULT,
                    help="Dir with <video_id>.mp4 source videos.")
    p.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT,
                    help="Output directory for per-video caption JSONL.")
    p.add_argument("--hf-cache", default=HF_CACHE_DEFAULT,
                    help="HF_HOME for model weights cache.")
    p.add_argument("--no-skip", action="store_true",
                    help="Reprocess videos even if output already exists.")
    p.add_argument("--max-videos", type=int, default=0,
                    help="Limit videos processed this run (0 = no limit; for local smoke tests).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    skip_existing = not args.no_skip

    os.environ.setdefault("HF_HOME", args.hf_cache)
    os.makedirs(args.hf_cache, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))

    log.info("Loading A1 task list...")
    t0 = time.time()
    tasks_by_video = load_all_tasks(args.tasks_glob)
    all_vids = sorted(tasks_by_video.keys())
    my_vids = all_vids[task_id::num_tasks]
    if args.max_videos:
        my_vids = my_vids[: args.max_videos]
    log.info(f"Task {task_id}/{num_tasks}: {len(my_vids)}/{len(all_vids)} videos "
              f"(loaded task list in {time.time()-t0:.1f}s), skip_existing={skip_existing}")

    import torch
    torch.set_num_threads(int(_NUM_THREADS))
    from caption_prototype import load_model
    log.info(f"Loading Qwen2.5-VL-3B-Instruct (CPU, {_NUM_THREADS} threads)...")
    t_load = time.time()
    model, processor = load_model()
    log.info(f"Model loaded ({time.time()-t_load:.1f}s)")

    cumul = {"ok": 0, "skipped_vid": 0, "failed": 0}
    t_start = time.time()
    for idx, vid in enumerate(my_vids, 1):
        s = process_video(vid, tasks_by_video[vid], model, processor, args.output_dir, skip_existing)
        for k in cumul:
            cumul[k] += s[k]

        if idx % 10 == 0 or idx == len(my_vids):
            elapsed = time.time() - t_start
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (len(my_vids) - idx) / rate if rate > 0 else 0
            log.info(f"[{idx:5d}/{len(my_vids)}] vid={vid} "
                      f"ok={cumul['ok']} skip={cumul['skipped_vid']} failed={cumul['failed']} "
                      f"rate={rate:.2f}vid/s ETA={eta/60:.0f}m")

    elapsed = time.time() - t_start
    log.info(f"DONE task {task_id}: ok={cumul['ok']} skipped={cumul['skipped_vid']} "
              f"failed={cumul['failed']} wall={elapsed:.0f}s")


if __name__ == "__main__":
    main()
