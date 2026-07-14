#!/usr/bin/env python3
"""
Adapter: reshape caption_finevideo.py's (A2) flat per-anchor-point output into
the {activity_id: {chunk_idx_str: "<caption>...</caption>"}}-per-video shape
phase6_merge_adaptive.py's SNAC-style loader convention expects.

caption_finevideo.py is left untouched (it's an active, multi-day SLURM job) --
this is a cheap, idempotent, standalone re-grouping pass over its output, safe
to re-run in full every time (no incremental-merge complexity needed: no model
inference here, just JSON reshaping over small per-video files).

Input:  {captions_dir}/{video_id}_captions.jsonl   (one line per anchor point)
            {"video_id","scene_id","activity_id","chunk_idx","start_sec","has_agent","caption"}
Output: {output_dir}/{video_id}_captions_dict.jsonl (one line per video)
            {"video_id", "captions_by_activity": {activity_id: {chunk_idx_str: "<caption>...</caption>"}}}

Usage:
    python tools/analysis/build_caption_dict.py
    python tools/analysis/build_caption_dict.py --skip-existing   # incremental (still cheap without it)
"""

import argparse
import glob
import json
import os

CAPTIONS_DIR_DEFAULT = "/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/captions"
OUTPUT_DIR_DEFAULT = "/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/captions_dict"


def build_one_video(in_path: str) -> dict:
    """Returns (video_id, captions_by_activity, stats) for one {video_id}_captions.jsonl file."""
    video_id = None
    captions_by_activity = {}
    stats = {"lines": 0, "collisions": 0}

    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            stats["lines"] += 1
            video_id = video_id or t.get("video_id")
            act_id = t.get("activity_id", "")
            chunk_key = str(t.get("chunk_idx", ""))
            caption = t.get("caption", "").strip()
            if not act_id or not caption:
                continue

            by_chunk = captions_by_activity.setdefault(act_id, {})
            if chunk_key in by_chunk:
                stats["collisions"] += 1
                continue  # first caption at this chunk wins, defensive only -- select_anchor_points already dedups
            by_chunk[chunk_key] = f"<caption> {caption} </caption>"

    return video_id, captions_by_activity, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reshape A2 flat caption output into phase6-loader-ready per-video dicts.")
    p.add_argument("--captions-glob", default=os.path.join(CAPTIONS_DIR_DEFAULT, "*_captions.jsonl"))
    p.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_paths = sorted(glob.glob(args.captions_glob))
    if not in_paths:
        raise FileNotFoundError(f"No files matched: {args.captions_glob!r}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{len(in_paths)} caption files found")

    grand = {"videos": 0, "lines": 0, "activities": 0, "collisions": 0, "skipped": 0}

    for in_path in in_paths:
        base = os.path.basename(in_path)  # {video_id}_captions.jsonl
        video_id_guess = base[: -len("_captions.jsonl")]
        out_path = os.path.join(args.output_dir, f"{video_id_guess}_captions_dict.jsonl")

        if args.skip_existing and os.path.exists(out_path):
            grand["skipped"] += 1
            continue

        video_id, captions_by_activity, stats = build_one_video(in_path)
        if video_id is None:
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "video_id": video_id,
                "captions_by_activity": captions_by_activity,
            }) + "\n")

        grand["videos"] += 1
        grand["lines"] += stats["lines"]
        grand["activities"] += len(captions_by_activity)
        grand["collisions"] += stats["collisions"]

    print(f"DONE: {grand['videos']} videos ({grand['skipped']} skipped), "
          f"{grand['lines']} caption lines -> {grand['activities']} activities "
          f"({grand['collisions']} chunk collisions, first-wins)")


if __name__ == "__main__":
    main()
