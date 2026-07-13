#!/usr/bin/env python3
"""
A1 -- caption task list generation (CPU-only, first stage of the captioning
pipeline described in PROGRESS_VI.md "Cap nhat phien lam viec - 12/07/2026").

Scans final_dataset_adaptive_v3/ shards, computes caption anchor points for
every activity via select_anchor_points() (agent-transition + periodic
supplement, see caption_prototype.py), and writes one task per anchor point
to a per-shard task-list JSONL under --output-dir. Activities whose video_id
has no local mp4 in --videos-dir are skipped (counted, not silently dropped).

Downstream (not yet built): A2 (SLURM array captioning job) reads these task
lists, extracts the frame at start_sec from video_path, and captions it with
Qwen2.5-VL; B1 extends phase6_merge_adaptive.py to inject the resulting
captions back into video_tokens keyed by (video_id, activity_id, chunk_idx).

Follows the same SLURM_ARRAY_TASK_ID/TASK_COUNT worker-split convention as
pipeline_pose/phase6_merge_adaptive.py.

Usage:
    python tools/analysis/generate_caption_tasks.py
    python tools/analysis/generate_caption_tasks.py --input-glob "/path/*.jsonl" --skip-existing
"""

import argparse
import glob
import json
import math
import os
import re

from caption_prototype import select_anchor_points

INPUT_GLOB_DEFAULT = (
    "/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/"
    "final_dataset_adaptive_v3/*.jsonl"
)
VIDEOS_DIR_DEFAULT = "/p/data1/mmlaion/shared/nguyen38/data/videos_staging"
RANK_RE = re.compile(r"_rank_(\d+)\.jsonl$")


def process_file(in_path: str, videos_dir: str, min_gap_sec: float, target_count: int):
    """Returns (tasks: list[dict], stats: dict) for one shard file."""
    tasks = []
    stats = {"videos": 0, "videos_missing_mp4": 0, "activities": 0, "task_points": 0}

    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            video_id = d.get("video_id", "")
            stats["videos"] += 1

            video_path = os.path.join(videos_dir, f"{video_id}.mp4")
            if not os.path.exists(video_path):
                stats["videos_missing_mp4"] += 1
                continue

            for scene in d.get("scenes", []):
                scene_id = scene.get("scene_id", "")
                for act in scene.get("activities", []):
                    ct = act.get("chunk_timing") or []
                    if not ct:
                        continue
                    stats["activities"] += 1
                    pts = select_anchor_points(ct, min_gap_sec=min_gap_sec, target_count=target_count)
                    for p in pts:
                        tasks.append({
                            "video_id": video_id,
                            "video_path": video_path,
                            "scene_id": scene_id,
                            "activity_id": act.get("activity_id", ""),
                            "chunk_idx": p["chunk_idx"],
                            "start_sec": p["start_sec"],
                            "has_agent": p["has_agent"],
                        })
                        stats["task_points"] += 1

    return tasks, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A1: generate caption anchor-point task list from chunk_timing.")
    p.add_argument("--input-glob", default=INPUT_GLOB_DEFAULT,
                    help="Glob for final_dataset_adaptive_v3 shard files.")
    p.add_argument("--videos-dir", default=VIDEOS_DIR_DEFAULT,
                    help="Dir with <video_id>.mp4 source videos.")
    p.add_argument("--output-dir", default=os.path.join("outputs", "caption_tasks"),
                    help="Output directory for per-shard task-list JSONL.")
    p.add_argument("--min-gap-sec", type=float, default=5.0,
                    help="Debounce gap for select_anchor_points (final design uses 5.0).")
    p.add_argument("--target-count", type=int, default=4,
                    help="Target anchor points per activity (periodic supplement target).")
    p.add_argument("--skip-existing", action="store_true",
                    help="Skip shards whose output file already exists.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = sorted(glob.glob(args.input_glob))
    if not input_paths:
        raise FileNotFoundError(f"No files matched: {args.input_glob!r}")

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    chunk = math.ceil(len(input_paths) / num_tasks)
    start = (task_id - 1) * chunk
    end = min(start + chunk, len(input_paths))
    my_paths = input_paths[start:end]

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[Worker {task_id}/{num_tasks}] {len(my_paths)}/{len(input_paths)} files")

    grand = {"files": 0, "videos": 0, "videos_missing_mp4": 0, "activities": 0, "task_points": 0}

    for in_path in my_paths:
        base = os.path.basename(in_path)
        m = RANK_RE.search(base)
        out_name = f"caption_tasks_rank_{m.group(1)}.jsonl" if m else f"caption_tasks_{os.path.splitext(base)[0]}.jsonl"
        out_path = os.path.join(args.output_dir, out_name)

        if args.skip_existing and os.path.exists(out_path):
            print(f"  skip (exists): {out_name}")
            continue

        tasks, stats = process_file(in_path, args.videos_dir, args.min_gap_sec, args.target_count)
        with open(out_path, "w", encoding="utf-8") as f:
            for t in tasks:
                f.write(json.dumps(t) + "\n")

        grand["files"] += 1
        for k in ("videos", "videos_missing_mp4", "activities", "task_points"):
            grand[k] += stats[k]
        avg = stats["task_points"] / stats["activities"] if stats["activities"] else 0.0
        print(f"  {base}: {stats['videos']} videos ({stats['videos_missing_mp4']} missing mp4), "
              f"{stats['activities']} activities -> {stats['task_points']} tasks (avg {avg:.2f}/activity)")

    avg_total = grand["task_points"] / grand["activities"] if grand["activities"] else 0.0
    print(f"[Worker {task_id}/{num_tasks}] DONE: {grand['files']} files, {grand['videos']} videos "
          f"({grand['videos_missing_mp4']} missing mp4), {grand['activities']} activities, "
          f"{grand['task_points']} task points (avg {avg_total:.2f}/activity)")


if __name__ == "__main__":
    main()
