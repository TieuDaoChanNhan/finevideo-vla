#!/usr/bin/env python3
"""
Download MiG-NJU/OmniVideo-100K from HuggingFace Hub.

Verified via HF tree API 18/07/2026:
  - videos.tar.part_aa .. videos.tar.part_ae  (~52.4GB, 5 parts, real video bytes
    -- cat parts together then `tar xf` to get the actual video files)
  - scripts.jsonl        (149MB)  -- structured script/evidence-chain per video,
    usable as a ready-made caption/language-anchor (no need to run our own
    captioning pipeline like we did for FineVideo)
  - train_oe_70k.jsonl / train_mcq_30k.jsonl (+ *_formatted variants) -- QA pairs

Total real size: ~52.9GB. License: apache-2.0 (verified from HF cardData tag).

Real video (not URLs) reusing the existing Step A pipeline (Seed2/Cosmos/AVC-LM)
with zero new token modality needed -- see datasets.md section 5.

Must run from a JUWELS login node (compute nodes have no internet).
Resumable: snapshot_download skips files already complete in local_dir, and
this script retries on transient network errors until the full download
succeeds -- safe to re-run / leave in a tmux session.

Usage:
    export HF_TOKEN='hf_...'
    python3 tools/extract/download_omnivideo_100k.py
"""
import os
import time

os.environ.setdefault("HF_HOME", "/p/data1/mmlaion/nguyen38/hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # Xet backend flaky on this cluster, see project memory

from huggingface_hub import login, snapshot_download

REPO_ID = "MiG-NJU/OmniVideo-100K"
LOCAL_DIR = "/p/data1/mmlaion/shared/vla/omnivideo_100k"
MAX_WORKERS = 16
RETRY_DELAY_SEC = 30
ALLOW_PATTERNS = ["videos.tar.part_*", "*.jsonl"]
EXPECTED_PARTS = 5


def report_progress(local_dir):
    if not os.path.isdir(local_dir):
        print("  (nothing downloaded yet)", flush=True)
        return
    parts_done = sorted(f for f in os.listdir(local_dir) if f.startswith("videos.tar.part_"))
    jsonl_done = sorted(f for f in os.listdir(local_dir) if f.endswith(".jsonl"))
    size_gb = sum(os.path.getsize(os.path.join(local_dir, f)) for f in os.listdir(local_dir)
                  if os.path.isfile(os.path.join(local_dir, f))) / 1e9
    print(f"  progress: {len(parts_done)}/{EXPECTED_PARTS} video parts, "
          f"{len(jsonl_done)} jsonl files, {size_gb:.1f} GB on disk so far", flush=True)


def main():
    if "HF_TOKEN" not in os.environ:
        raise EnvironmentError("HF_TOKEN not set. Run: export HF_TOKEN='hf_...'")
    login(token=os.environ["HF_TOKEN"])

    os.makedirs(LOCAL_DIR, exist_ok=True)
    print(f"Target: {LOCAL_DIR}  (~52.9GB total, 5 video-tar parts + jsonl QA/scripts)", flush=True)

    attempt = 0
    while True:
        attempt += 1
        print(f"\n[attempt {attempt}] snapshot_download starting -> {LOCAL_DIR}", flush=True)
        report_progress(LOCAL_DIR)
        try:
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                allow_patterns=ALLOW_PATTERNS,
                local_dir=LOCAL_DIR,
                max_workers=MAX_WORKERS,
            )
            print("snapshot_download completed successfully.", flush=True)
            break
        except Exception as e:
            print(f"[attempt {attempt}] failed: {e!r} -- retrying in {RETRY_DELAY_SEC}s", flush=True)
            report_progress(LOCAL_DIR)
            time.sleep(RETRY_DELAY_SEC)

    report_progress(LOCAL_DIR)
    print(f"\nDone. Next step (not run by this script): "
          f"cat videos.tar.part_* > videos.tar && tar xf videos.tar")


if __name__ == "__main__":
    main()
