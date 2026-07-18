#!/usr/bin/env python3
"""
Extract per-episode JPEG frame sequences from RoboVQA's tfrecord shards
(train: 175 shards, val: 9 shards) using the dependency-free parser in
tfrecord_lite.py -- see that file's docstring for why no tensorflow.

Verified before writing this (18/07/2026, 500-episode sample from shard 0):
  - context always has exactly {unique_id, video_filename}
  - feature_lists always has exactly {images, raw_texts, texts, texts_start,
    texts_end, timestamps}
  - every episode has exactly 16 image frames, all valid JPEG (magic bytes
    checked, 1000/1000)
  - `texts` feature (single blob) decodes byte-identical to the `text`
    field of the matching record in json/train/*.json when joined on
    tfrecord's `video_filename` == json's `video` (NOT `unique_id`/`uid` --
    those are a different, unrelated ID space)

Since data_prep/robovqa/flatten_text.py already extracted `text` from the
json/ shards, this script does NOT re-extract text -- only images +
timestamps, written per-episode so they can be joined back to the flat text
JSONL by video_filename stem.

Output layout:
    robovqa_frames/<video_stem>/frame_00.jpg .. frame_15.jpg
    robovqa_frames_manifest.jsonl  -- one line per episode:
        {"video_filename": "...", "num_frames": 16,
         "timestamps": [...], "frame_dir": "robovqa_frames/<video_stem>"}

Resumable: skips a shard's episodes entirely if that shard's manifest
entries already exist (checked via a per-shard "done" marker file), safe to
re-run / kill and restart.

Usage:
    python3 data_prep/robovqa/extract_frames.py [--limit-shards N]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tfrecord_lite import iter_tfrecords, parse_sequence_example, context_str

SRC_DIRS = [
    ("/p/data1/mmlaion/shared/vla/robovqa/tfrecord/train", "train"),
    ("/p/data1/mmlaion/shared/vla/robovqa/tfrecord/val", "val"),
]
OUT_ROOT = "/p/data1/mmlaion/shared/vla/robovqa_flat"
FRAMES_DIR = os.path.join(OUT_ROOT, "robovqa_frames")
MANIFEST_PATH = os.path.join(OUT_ROOT, "robovqa_frames_manifest.jsonl")
DONE_MARKERS_DIR = os.path.join(OUT_ROOT, "robovqa_frames_shard_done")


def extract_shard(path, split, manifest_f):
    n_episodes = 0
    n_frames = 0
    for data in iter_tfrecords(path):
        ctx, fls = parse_sequence_example(data)
        video_filename = context_str(ctx, "video_filename")
        if not video_filename:
            continue
        stem = os.path.splitext(video_filename)[0]

        # fls["images"] is a list of (kind, values) -- one entry PER TIMESTEP
        # (FeatureList = repeated Feature), each Feature holding a 1-element
        # BytesList. Flatten to one JPEG bytes object per timestep.
        image_steps = fls.get("images", [])
        if not image_steps:
            continue
        image_bytes_list = [values[0] for kind, values in image_steps if values]

        timestamp_steps = fls.get("timestamps", [])
        timestamps = [values[0] for kind, values in timestamp_steps if values]

        frame_dir = os.path.join(FRAMES_DIR, stem)
        os.makedirs(frame_dir, exist_ok=True)
        for i, jpeg_bytes in enumerate(image_bytes_list):
            with open(os.path.join(frame_dir, f"frame_{i:02d}.jpg"), "wb") as out:
                out.write(jpeg_bytes)

        manifest_f.write(json.dumps({
            "video_filename": video_filename,
            "split": split,
            "num_frames": len(image_bytes_list),
            "timestamps": timestamps,
            "frame_dir": frame_dir,
        }) + "\n")

        n_episodes += 1
        n_frames += len(image_bytes_list)
    return n_episodes, n_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-shards", type=int, default=None,
                         help="only process the first N shards per split (for testing)")
    args = parser.parse_args()

    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(DONE_MARKERS_DIR, exist_ok=True)

    total_episodes = 0
    total_frames = 0

    with open(MANIFEST_PATH, "a") as manifest_f:
        for src_dir, split in SRC_DIRS:
            shard_files = sorted(os.listdir(src_dir))
            if args.limit_shards:
                shard_files = shard_files[:args.limit_shards]
            print(f"{split}: {len(shard_files)} shards")
            for fname in shard_files:
                marker = os.path.join(DONE_MARKERS_DIR, f"{split}_{fname}.done")
                if os.path.exists(marker):
                    continue
                path = os.path.join(src_dir, fname)
                n_ep, n_fr = extract_shard(path, split, manifest_f)
                manifest_f.flush()
                total_episodes += n_ep
                total_frames += n_fr
                with open(marker, "w") as m:
                    m.write(f"{n_ep} episodes, {n_fr} frames\n")
                print(f"  {fname}: {n_ep} episodes, {n_fr} frames")

    print(f"\nTotal this run: {total_episodes} episodes, {total_frames} frames")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Frames: {FRAMES_DIR}")


if __name__ == "__main__":
    main()
