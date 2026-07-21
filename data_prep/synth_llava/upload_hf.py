#!/usr/bin/env python3
"""
Upload the seed2-tokenized synth_llava/synth_llava2 dataset to HF.

Source: 151 shards at
  /p/data1/mmlaion/shared/vla/synth_llava_flat/{synth_llava,synth_llava2}_shard-*.jsonl
  (603,999 rows, 19,327,968 seed2 tokens -- verified 2026-07-21: 0 duplicate ids,
  0 bad-json lines, 0 missing keys, 0 rows without a seed2 token. See
  tokenize_seed2.py for the tokenize step and PROGRESS_VI.md 21/07 entry for
  the check.)

Each row: {"id": <source row id, prefixed synth_llava(2)_>, "text": <flattened
training record>}, already fully flat/training-ready straight out of
tokenize_seed2.py (single modality -- image to seed2 -- so there is no
separate agent/pose stream to merge in, unlike the FineVideo/OmniVideo
pipelines):
  <caption><seed2_N> <seed2_N> ... <caption text></caption>

Unlike laion_emotional_roleplay's uniform 14-shard numbering, the two source
prefixes here (synth_llava: 56 shards, synth_llava2: 95 shards) do not share
a contiguous shard index, so shards are discovered by glob rather than
assumed by count.

Usage:
    export HF_TOKEN='hf_...'
    python data_prep/synth_llava/upload_hf.py --repo-id EmpathicRobotics/<name-you-pick>
    python data_prep/synth_llava/upload_hf.py --repo-id ... --skip-compress   # reuse existing .gz
    python data_prep/synth_llava/upload_hf.py --repo-id ... --skip-upload     # compress only
"""
import argparse
import glob
import gzip
import os
import random
import shutil

from huggingface_hub import HfApi, login

SOURCE_DIR = "/p/data1/mmlaion/shared/vla/synth_llava_flat"
UPLOAD_DIR = "/p/data1/mmlaion/shared/vla/synth_llava_hf_upload"
TEST_RATIO = 0.07  # ~1/14 shards held out, same ratio as laion_emotional_roleplay
SEED = 42

DATASET_CARD = """---
license: other
---

# {repo_name}

Seed2-tokenized version of `synth_llava`/`synth_llava2`
(`mixture-vitae-backup/MixtureVitae-Backup/data/multimodal`), Huu's own
synthetic (AI-generated) image+caption dataset, prepared for the PAB-Spline /
omni-modal VLA project (image<->text modality pair).

- **603,999 rows** (56 `synth_llava` shards + 95 `synth_llava2` shards)
- **19,327,968 seed2 tokens** total (32 tokens/image, `Seed2Tokenizer`)
- Source images: 256x256 PNG, WebDataset format, synthetic (llava_pretrain-style)
- **License:** Huu's own dataset (project lead), confirmed permissive by him
  directly (2026-07-21) -- exact license terms/name not documented in the
  source, hence the generic `license: other` tag above rather than a specific
  SPDX id.

## Format

Each row is `{{"id": ..., "text": ...}}` with `text`:

```
<caption><seed2_N> <seed2_N> ... (32 seed2 tokens)<caption text></caption>
```

`id` is prefixed `synth_llava_` or `synth_llava2_` by source shard.

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
    ap = argparse.ArgumentParser(description="Upload seed2-tokenized synth_llava dataset to HuggingFace.")
    ap.add_argument("--repo-id", required=True, help="e.g. EmpathicRobotics/synth-llava-seed2")
    ap.add_argument("--source-dir", default=SOURCE_DIR)
    ap.add_argument("--upload-dir", default=UPLOAD_DIR)
    ap.add_argument("--skip-compress", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()

    train_dir = os.path.join(args.upload_dir, "train")
    test_dir = os.path.join(args.upload_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    all_shards = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(args.source_dir, "*.jsonl"))
    )
    if not all_shards:
        raise FileNotFoundError(f"No .jsonl shards found in {args.source_dir}")
    print(f"Found {len(all_shards)} shards.")

    random.seed(SEED)
    shuffled = all_shards[:]
    random.shuffle(shuffled)
    test_count = max(1, round(len(all_shards) * TEST_RATIO))
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
        commit_message="Upload seed2-tokenized synth_llava/synth_llava2 "
                        "(603,999 rows, 19.33M seed2 tokens)",
        allow_patterns=["train/*.jsonl.gz", "test/*.jsonl.gz", "README.md"],
    )

    print(f"Done! https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
