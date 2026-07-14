#!/usr/bin/env python3
"""
One-time manifest: FineVideo video_id -> HF parquet shard_index.

Scans all 1357 train-*.parquet shards of HuggingFaceFV/finevideo (json column
only -- no mp4 blobs downloaded) to record which shard each video_id lives in.
extract_speech_segments.py uses this manifest to fetch timecoded_text_to_speech
for a given video without re-scanning the whole 1357-shard dataset each time.

Shard assignment is static for a given dataset revision, so this only needs to
run once. Reproduces the exact video_id derivation used by
pipeline_video/pipeline.py's parse_video_metadata() so video_ids match what's
already in training_ready_rank_*.jsonl / final_dataset_adaptive_v3.

Usage:
    python tools/analysis/build_video_shard_manifest.py
    python tools/analysis/build_video_shard_manifest.py --start 0 --end 100   # partial run
"""

import argparse
import json
import os

from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq

NUM_SHARDS = 1357
REPO = "HuggingFaceFV/finevideo"
OUTPUT_DEFAULT = os.path.join("outputs", "speech_extraction", "video_id_to_shard.json")


def get_token():
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_path):
        with open(token_path) as f:
            return f.read().strip()
    return None


def derive_video_id(row: dict) -> str:
    """Mirror pipeline_video/pipeline.py's parse_video_metadata() id logic exactly."""
    video_id = (row.get("original_video_filename") or "unknown").replace(".mp4", "")
    if video_id == "unknown":
        video_id = (row.get("youtube_title") or "video").replace(" ", "_").lower()
    return video_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build video_id -> shard_index manifest from HF FineVideo parquet.")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=NUM_SHARDS)
    p.add_argument("--output", default=OUTPUT_DEFAULT)
    p.add_argument("--checkpoint-every", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fs = HfFileSystem(token=get_token())

    manifest = {}
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        print(f"Loaded existing manifest: {len(manifest)} video_ids")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    errors = 0
    for i in range(args.start, args.end):
        path = f"hf://datasets/{REPO}/data/train-{i:05d}-of-{NUM_SHARDS:05d}.parquet"
        try:
            tbl = pq.read_table(path, columns=["json"], filesystem=fs)
        except Exception as e:
            print(f"  shard {i}: ERROR {e}")
            errors += 1
            continue

        rows = tbl.column("json").to_pylist()
        for row in rows:
            video_id = derive_video_id(row)
            if video_id:
                manifest[video_id] = i

        if (i - args.start) % args.checkpoint_every == 0:
            print(f"  shard {i}/{args.end}: +{len(rows)} rows, manifest size {len(manifest)}")
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(manifest, f)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    print(f"DONE: {len(manifest)} video_ids mapped ({errors} shard errors) -> {args.output}")


if __name__ == "__main__":
    main()
