#!/usr/bin/env python3
"""
Push just the updated README.md to EmpathicRobotics/vla-1.7b-qwen3-v2 --
does NOT re-upload the ~7GB model weights (unlike upload_vla_v2_model.py,
which does a full folder upload and is meant for the initial publish only).

Written 2026-07-23 to ship the fixed decode instructions (portable
cosmos/agent/snac decoders, previously-missing SNAC section) without
re-pushing the model itself.

Usage:
    export HF_TOKEN='hf_...'
    python tools/upload/update_vla_v2_card.py
"""
import os

from huggingface_hub import HfApi

from upload_vla_v2_model import README, REPO_ID


def main():
    if "HF_TOKEN" not in os.environ:
        raise EnvironmentError("HF_TOKEN not set. Run: export HF_TOKEN='hf_...'")

    api = HfApi()
    tmp_path = "/tmp/vla_v2_README.md"
    with open(tmp_path, "w") as f:
        f.write(README)

    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Fix decode instructions: portable cosmos/agent decoders, add missing SNAC section (2026-07-23)",
    )
    os.remove(tmp_path)
    print(f"Done: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
