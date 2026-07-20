#!/usr/bin/env python3
"""
Upload the finalized OmniVideo-100K dataset to HF.

Source: 32 shards at
  /p/data1/mmlaion/shared/vla/omnivideo_100k_final/step_a_rank_{0..31}.jsonl
  (5,214 rows -- verified 2026-07-20: 0 malformed, 799/5,214 videos have
  <agent> blocks matching Phase 5's 62,631 pose windows exactly, 5,214/5,214
  have QA joined. See PROGRESS_VI.md for the full merge pipeline + checks.)

Each row: {"video_id": ..., "text": <flattened training record>}
  <seed2_N>... <cosmos_N>... [<agent> <fps_30> <pelvis>...</agent>]...
  Q: <question>\nA: <answer>\n...  (all QA pairs for this video_id, appended)

Usage:
    export HF_TOKEN='hf_...'
    python data_prep/omnivideo_100k/upload_hf.py --repo-id EmpathicRobotics/<name-you-pick>
    python data_prep/omnivideo_100k/upload_hf.py --repo-id ... --skip-compress   # reuse existing .gz
    python data_prep/omnivideo_100k/upload_hf.py --repo-id ... --skip-upload     # compress only
"""
import argparse
import gzip
import os
import random
import shutil

from huggingface_hub import HfApi, login

SOURCE_DIR = "/p/data1/mmlaion/shared/vla/omnivideo_100k_final"
UPLOAD_DIR = "/p/data1/mmlaion/shared/vla/omnivideo_100k_final/hf_upload"
SHARD_PREFIX = "step_a_rank"
TOTAL_SHARDS = 32
TEST_RATIO = 0.06  # ~2/32 shards held out
SEED = 42

DATASET_CARD = """---
license: apache-2.0
---

# {repo_name}

Finalized [MiG-NJU/OmniVideo-100K](https://huggingface.co/datasets/MiG-NJU/OmniVideo-100K)
for the PAB-Spline / omni-modal VLA project: video tokens + pose agent tokens
+ QA, merged into one record per video.

- **5,214 rows** (one per source video)
- **799/5,214 videos have `<agent>` blocks** (3D pose, 62,631 windows total) --
  pose pipeline only ran on a sports-subset of 1,126 videos, and only videos
  with clean 8-frame windows after hallucination/YOLO filtering produced
  output; the rest are video+QA only, no pose (same partial-coverage pattern
  as the FineVideo-VLA flagship dataset)
- **5,214/5,214 videos have QA appended** (99,983 QA pairs total: 70,017
  open-ended + 29,966 multiple-choice, from the source `train_oe_70k.jsonl`/
  `train_mcq_30k.jsonl`)
- 17,229,664 `<seed2_N>` tokens, 201,736,400 `<cosmos_N>` tokens total

## Format

Each row is `{{"video_id": ..., "text": ...}}` with `text`, per 8-frame/30fps
chunk in order:

```
<seed2_N>... [<caption> ... </caption>] <cosmos_N>... [<agent> <fps_30> <pelvis> ... </agent>] [<speech> ... </speech>]
...(repeated per chunk)...
Q: <question>
A: <answer>
Reasoning: <cross-modal reasoning hint, from source `analysis.connections`>
...(repeated per QA pair for this video)...
```

`<agent>` (pose, adaptive-PCHIP 17-joint xyz, same scheme as FineVideo-VLA)
only appears on chunks where the pose pipeline produced a clean window for
that video; most videos have none. `avc_lm` payload is always discarded,
`<cosmos_N>` is kept with 50% per-chunk dropout, `<seed2_N>` always kept --
same convention as FineVideo's `pipeline_pose/phase7_flatten.py`.

## Pipeline

1. `step_a/step_a_tokenize_video.py` -- video -> raw seed2/cosmos/avc_lm (+ scripts.jsonl caption/speech anchors)
2. `step_a/flatten_step_a_video.py` -- intermediate flatten (not the final artifact, kept for reference)
3. `pose/phase1..4_*_omnivideo.py` + shared `pipeline_pose/phase5_adaptive_pchip.py` -- 3D pose -> agent tokens (sports subset only)
4. `phase6_merge_omnivideo.py` -- inject `<agent>` into the Step A stream, chunk-aligned (window_id == chunk_idx * 8, verified exact match, no time interpolation needed)
5. `phase7_finalize_omnivideo.py` -- append QA (grouped by video_id) after the video token stream -- **this repo's content**

## Split

Train/test split by shard (not by row), seed 42, ~{test_ratio:.0%} held out as test.
"""


def compress_shard(old_path: str, new_path: str) -> str:
    if os.path.exists(new_path):
        return f"  Skipped (exists): {os.path.basename(new_path)}"
    with open(old_path, "rb") as f_in, gzip.open(new_path, "wb", compresslevel=5) as f_out:
        shutil.copyfileobj(f_in, f_out)
    return f"  Done: {os.path.basename(new_path)}"


def main():
    ap = argparse.ArgumentParser(description="Upload finalized OmniVideo-100K dataset to HuggingFace.")
    ap.add_argument("--repo-id", required=True, help="e.g. EmpathicRobotics/omnivideo-100k-final")
    ap.add_argument("--source-dir", default=SOURCE_DIR)
    ap.add_argument("--upload-dir", default=UPLOAD_DIR)
    ap.add_argument("--skip-compress", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()

    train_dir = os.path.join(args.upload_dir, "train")
    test_dir = os.path.join(args.upload_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    all_shards = [f"{SHARD_PREFIX}_{i}.jsonl" for i in range(TOTAL_SHARDS)]
    missing = [f for f in all_shards if not os.path.exists(os.path.join(args.source_dir, f))]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} shards, first: {missing[0]}")
    print(f"All {TOTAL_SHARDS} shards found.")

    random.seed(SEED)
    shuffled = all_shards[:]
    random.shuffle(shuffled)
    test_count = max(1, round(TOTAL_SHARDS * TEST_RATIO))
    test_files = shuffled[:test_count]
    train_files = shuffled[test_count:]
    print(f"Train: {len(train_files)} shards | Test: {len(test_files)} shards")

    if not args.skip_compress:
        for prefix, files, target_dir in (("train", train_files, train_dir), ("test", test_files, test_dir)):
            print(f"Compressing {prefix} ({len(files)} files)...")
            for i, name in enumerate(files):
                old_path = os.path.join(args.source_dir, name)
                new_name = f"{prefix}-{i:05d}-of-{len(files):05d}.jsonl.gz"
                print(compress_shard(old_path, os.path.join(target_dir, new_name)))

    actual_train = len([f for f in os.listdir(train_dir) if f.endswith(".jsonl.gz")])
    actual_test = len([f for f in os.listdir(test_dir) if f.endswith(".jsonl.gz")])
    if actual_train != len(train_files):
        raise ValueError(f"Expected {len(train_files)} train shards, found {actual_train}")
    if actual_test != len(test_files):
        raise ValueError(f"Expected {len(test_files)} test shards, found {actual_test}")

    if args.skip_upload:
        print("Skipping upload (--skip-upload). Files in:", args.upload_dir)
        return

    if "HF_TOKEN" not in os.environ:
        raise EnvironmentError("HF_TOKEN not set. Run: export HF_TOKEN='hf_...'")

    login(token=os.environ["HF_TOKEN"])
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)

    repo_name = args.repo_id.split("/")[-1]
    readme_path = os.path.join(args.upload_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(DATASET_CARD.format(repo_name=repo_name, test_ratio=TEST_RATIO))
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add dataset card",
    )

    print(f"Uploading to {args.repo_id} ...")
    api.upload_folder(
        folder_path=args.upload_dir,
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Upload finalized OmniVideo-100K (5,214 rows, 799 with agent tokens, "
                        "5,214 with QA)",
        allow_patterns=["train/*.jsonl.gz", "test/*.jsonl.gz", "README.md"],
    )

    print(f"Done! https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
