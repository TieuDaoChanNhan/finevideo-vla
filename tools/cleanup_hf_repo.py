#!/usr/bin/env python3
"""
Delete leftover train/ and test/ folders from EmpathicRobotics/FineVideo-VLA-Agent on HuggingFace.

Usage:
    export HF_TOKEN='hf_...'
    python tools/cleanup_hf_repo.py
"""

import os

from huggingface_hub import CommitOperationDelete, HfApi, login


REPO_ID = "EmpathicRobotics/FineVideo-VLA-Agent"


def main():
    if "HF_TOKEN" not in os.environ:
        raise EnvironmentError("HF_TOKEN not set. Run: export HF_TOKEN='hf_...'")

    login(token=os.environ["HF_TOKEN"])
    api = HfApi()

    all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    to_delete = [f for f in all_files if f.startswith("train/") or f.startswith("test/")]

    if not to_delete:
        print("No train/ or test/ files found. Nothing to delete.")
        return

    print(f"Found {len(to_delete)} files to delete:")
    for f in to_delete:
        print(f"  {f}")

    operations = [CommitOperationDelete(path_in_repo=f) for f in to_delete]

    api.create_commit(
        repo_id=REPO_ID,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Remove leftover train/ and test/ folders ({len(to_delete)} files)",
    )

    print(f"Deleted {len(to_delete)} files from {REPO_ID}.")


if __name__ == "__main__":
    main()
