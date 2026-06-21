#!/usr/bin/env python3
"""
Delete old root-level files from EmpathicRobotics/FineVideo-Phase7-Flattened.

Removes the previous upload's files (train-*_flattened.jsonl.gz,
test-*_flattened.jsonl.gz, nohup.out) that sit in the repo root,
keeping only .gitattributes and the new train/ + test/ subdirectories.

Usage:
    export HF_TOKEN='hf_...'
    python tools/cleanup_flattened_hf.py
"""

import os
from huggingface_hub import CommitOperationDelete, HfApi, login

REPO_ID = "EmpathicRobotics/FineVideo-Phase7-Flattened"


def main():
    if "HF_TOKEN" not in os.environ:
        raise EnvironmentError("HF_TOKEN not set. Run: export HF_TOKEN='hf_...'")

    login(token=os.environ["HF_TOKEN"])
    api = HfApi()

    all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")

    keep = {".gitattributes"}
    old_files = [
        f for f in all_files
        if f not in keep and not f.startswith("train/") and not f.startswith("test/")
    ]

    if not old_files:
        print("No old files to delete. Repo is clean.")
        return

    print(f"Deleting {len(old_files)} old root-level files:")
    for f in sorted(old_files):
        print(f"  {f}")

    operations = [CommitOperationDelete(path_in_repo=f) for f in old_files]

    api.create_commit(
        repo_id=REPO_ID,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Remove {len(old_files)} old root-level files from previous upload",
    )

    print(f"\nDone! Deleted {len(old_files)} files from {REPO_ID}")


if __name__ == "__main__":
    main()
