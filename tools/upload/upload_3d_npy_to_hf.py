#!/usr/bin/env python3
"""
Convert 3d_npy/*.npy files to Parquet shards and upload to HuggingFace.

Dataset: EmpathicRobotics/FineVideo-Phase2-3DPose
Schema per row:
  video_id   : str   — YouTube video ID (filename without .npy)
  num_frames : int32 — number of frames T
  joints_3d  : bytes — raw float32 bytes, reshape to (num_frames, 17, 3)

Usage:
  export HF_TOKEN=hf_...
  python upload_3d_npy_to_hf.py [--dry-run] [--num-shards 100] [--out-dir /tmp/parquet_shards]
"""

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, DatasetCard

NPY_DIR = Path("/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs/3d_npy")
REPO_ID = "EmpathicRobotics/FineVideo-Phase2-3DPose"
REPO_TYPE = "dataset"

SCHEMA = pa.schema([
    pa.field("video_id",   pa.string()),
    pa.field("num_frames", pa.int32()),
    # raw little-endian float32 bytes; reshape: (num_frames, 17, 3)
    pa.field("joints_3d",  pa.large_binary()),
])


def build_shards(npy_files: list[Path], out_dir: Path, num_shards: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_size = max(1, len(npy_files) // num_shards)
    shard_paths = []

    for shard_idx in range(num_shards):
        shard_path = out_dir / f"train-{shard_idx:05d}-of-{num_shards:05d}.parquet"
        if shard_path.exists():
            print(f"  shard {shard_idx} already exists, skipping")
            shard_paths.append(shard_path)
            continue

        start = shard_idx * shard_size
        end = start + shard_size if shard_idx < num_shards - 1 else len(npy_files)
        batch = npy_files[start:end]
        if not batch:
            continue

        video_ids, num_frames_list, joints_list = [], [], []
        for f in batch:
            arr = np.load(f)                         # (T, 17, 3) float32
            video_ids.append(f.stem)
            num_frames_list.append(arr.shape[0])
            joints_list.append(arr.astype(np.float32).tobytes())

        table = pa.table(
            {
                "video_id":   pa.array(video_ids,       type=pa.string()),
                "num_frames": pa.array(num_frames_list, type=pa.int32()),
                "joints_3d":  pa.array(joints_list,     type=pa.large_binary()),
            },
            schema=SCHEMA,
        )
        pq.write_table(table, shard_path, compression="zstd", compression_level=3)
        print(f"  shard {shard_idx:4d}/{num_shards}: {len(batch)} videos → {shard_path.name} "
              f"({shard_path.stat().st_size / 1e6:.1f} MB)")
        shard_paths.append(shard_path)

    return shard_paths


def upload_shards(shard_paths: list[Path], token: str) -> None:
    api = HfApi(token=token)

    # Create repo if it doesn't exist
    api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, exist_ok=True, private=False)

    for shard_path in shard_paths:
        path_in_repo = f"data/{shard_path.name}"
        print(f"  uploading {shard_path.name} ({shard_path.stat().st_size / 1e6:.1f} MB) ...")
        api.upload_file(
            path_or_fileobj=str(shard_path),
            path_in_repo=path_in_repo,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message=f"Add {shard_path.name}",
        )
    print("All shards uploaded.")


def push_dataset_card(token: str) -> None:
    card_content = """\
---
license: cc-by-4.0
task_categories:
- robotics
- video-understanding
language:
- en
tags:
- pose-estimation
- 3d-human-pose
- finevideo
- vla
- motionbert
---

# FineVideo 3D Human Pose Dataset

3D joint coordinates extracted from [HuggingFace FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo)
using HRNet (2D) + MotionBERT (3D lift).

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `video_id` | string | YouTube video ID |
| `num_frames` | int32 | Number of frames T |
| `joints_3d` | bytes | Raw float32 bytes — reshape to `(num_frames, 17, 3)` |

17 joints follow the Human3.6M / COCO keypoint order.
Coordinates are in metres, camera-relative.

## Usage

```python
from datasets import load_dataset
import numpy as np

ds = load_dataset("EmpathicRobotics/FineVideo-Phase2-3DPose", split="train", streaming=True)
for row in ds:
    poses = np.frombuffer(row["joints_3d"], dtype=np.float32).reshape(row["num_frames"], 17, 3)
    print(row["video_id"], poses.shape)
    break
```

## Part of FineVideo-VLA

This data feeds into the `<agent>` token stream in the
[FineVideo-VLA](https://github.com/EmpathicRobotics/FineVideo-VLA) pretraining pipeline.
"""
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=card_content.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="Add dataset card",
    )
    print("Dataset card pushed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="build shards but skip upload")
    parser.add_argument("--num-shards", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="/tmp/finevideo_3d_parquet")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    if not token and not args.dry_run:
        sys.exit("Set HF_TOKEN env var before running (export HF_TOKEN=hf_...)")

    npy_files = sorted(NPY_DIR.glob("*.npy"))
    print(f"Found {len(npy_files)} npy files in {NPY_DIR}")
    print(f"Building {args.num_shards} parquet shards in {args.out_dir} ...")

    shard_paths = build_shards(npy_files, Path(args.out_dir), args.num_shards)

    if args.dry_run:
        print("Dry run — skipping upload.")
        return

    print(f"\nUploading {len(shard_paths)} shards to {REPO_ID} ...")
    upload_shards(shard_paths, token)
    push_dataset_card(token)
    print(f"\nDone: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
