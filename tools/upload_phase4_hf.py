#!/usr/bin/env python3
"""
Upload Phase 4 YOLO-cleaned 3D pose data to HuggingFace.

Shards 40K per-video JSONL files into combined JSONL.GZ shards,
then uploads to EmpathicRobotics/FineVideo-Phase4-Pose.

Each output line: {"video_id": str, "window_id": int, "states": [8][17][3]}
  - states[f][j] = [x, y, z] in metres, root-centred, bone-normalised
  - 8 frames at 30fps per window, 17 joints (H36M skeleton)

Usage:
    export HF_TOKEN='hf_...'
    python tools/upload_phase4_hf.py [--source-dir PATH] [--upload-dir PATH]
"""

import argparse
import gzip
import multiprocessing as mp
import os
import random
import shutil
import glob

from huggingface_hub import HfApi, login


REPO_ID = "EmpathicRobotics/FineVideo-Phase4-Pose"
NUM_SHARDS = 64
TEST_RATIO = 0.05
SEED = 42


def build_shard(args):
    shard_idx, total_shards, file_list, output_path, split_name = args
    try:
        with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=5) as gz:
            for fpath in file_list:
                video_id = os.path.basename(fpath).replace("_cleaned.jsonl", "")
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # Inject video_id into each line if not present
                        if '"video_id"' not in line:
                            line = '{"video_id":"' + video_id + '",' + line[1:]
                        gz.write(line + "\n")
        return f"  {split_name} shard {shard_idx}/{total_shards}: {len(file_list)} videos -> {output_path}"
    except Exception as e:
        return f"  ERROR shard {shard_idx}: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Upload Phase 4 pose data to HuggingFace."
    )
    parser.add_argument(
        "--source-dir",
        default="/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs/yolo_cleaned_30fps",
    )
    parser.add_argument(
        "--upload-dir",
        default="/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs/hf_upload_phase4",
    )
    parser.add_argument(
        "--num-shards", type=int, default=NUM_SHARDS,
    )
    parser.add_argument(
        "--skip-compress", action="store_true",
        help="Skip sharding/compression, reuse existing files.",
    )
    parser.add_argument(
        "--skip-upload", action="store_true",
        help="Only shard and compress, don't upload.",
    )
    args = parser.parse_args()

    all_files = sorted(glob.glob(os.path.join(args.source_dir, "*_cleaned.jsonl")))
    print(f"Found {len(all_files)} Phase 4 files in {args.source_dir}")

    if not all_files:
        raise FileNotFoundError(f"No *_cleaned.jsonl files in {args.source_dir}")

    train_dir = os.path.join(args.upload_dir, "data")
    os.makedirs(train_dir, exist_ok=True)

    # Split into train/test by video
    random.seed(SEED)
    shuffled = list(all_files)
    random.shuffle(shuffled)
    test_count = max(1, int(len(shuffled) * TEST_RATIO))
    test_files = shuffled[:test_count]
    train_files = shuffled[test_count:]
    print(f"Train: {len(train_files)} videos | Test: {len(test_files)} videos")

    if not args.skip_compress:
        tasks = []

        # Build train shards
        n_train_shards = args.num_shards
        chunk_size = (len(train_files) + n_train_shards - 1) // n_train_shards
        for i in range(n_train_shards):
            chunk = train_files[i * chunk_size : (i + 1) * chunk_size]
            if not chunk:
                continue
            out = os.path.join(
                train_dir,
                f"train-{i:05d}-of-{n_train_shards:05d}.jsonl.gz",
            )
            tasks.append((i, n_train_shards, chunk, out, "train"))

        # Build test shards (fewer shards)
        n_test_shards = max(1, args.num_shards // 16)
        chunk_size = (len(test_files) + n_test_shards - 1) // n_test_shards
        for i in range(n_test_shards):
            chunk = test_files[i * chunk_size : (i + 1) * chunk_size]
            if not chunk:
                continue
            out = os.path.join(
                train_dir,
                f"test-{i:05d}-of-{n_test_shards:05d}.jsonl.gz",
            )
            tasks.append((i, n_test_shards, chunk, out, "test"))

        print(f"Building {len(tasks)} shards with {min(mp.cpu_count(), 16)} workers...")
        with mp.Pool(min(mp.cpu_count(), 16)) as pool:
            for res in pool.imap_unordered(build_shard, tasks):
                print(res)

        print("Sharding complete.")

    if args.skip_upload:
        print("Skipping upload (--skip-upload). Files ready in:", args.upload_dir)
        return

    if "HF_TOKEN" not in os.environ:
        raise EnvironmentError("HF_TOKEN not set. Run: export HF_TOKEN='hf_...'")

    login(token=os.environ["HF_TOKEN"])

    print(f"Uploading to {REPO_ID} ...")
    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

    # Upload dataset card first
    readme_path = os.path.join(os.path.dirname(__file__), "phase4_dataset_card.md")
    if os.path.exists(readme_path):
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="Add dataset card",
        )
        print("Uploaded dataset card.")

    api.upload_folder(
        folder_path=args.upload_dir,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Upload Phase 4 YOLO-cleaned 3D pose data (30fps, 40K videos)",
    )

    print(f"Done! https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
