#!/usr/bin/env python3
"""
Download sensenova/SenseNova-SI-8M (full config) from HuggingFace Hub.

Real repo layout (verified via HF tree API 18/07/2026 -- NOT the same as the
`full` dataset-viewer config, which points at a single auto-converted
parquet that does not exist as a raw file the same way):
  - SenseNova-SI-8M.parquet (851MB)  -- conversations + image `path` refs,
    does NOT embed image bytes (unlike the small *_1000samples.parquet
    preview, which is a separate auto-converted convenience file)
  - images_part_001.zip .. images_part_053.zip (~21.5GB each, ~1.10TB total)
    -- the actual image bytes, referenced by `path` from the parquet/jsonl

Total real size: ~1.13TB (53 zips + metadata). Images are real bytes, not
URLs -- no dead-link/license-via-crawl risk like MINT-1T-HTML had.

Must run from a JUWELS login node (compute nodes have no internet).
Resumable: snapshot_download skips files already complete in local_dir, and
this script retries on transient network errors until the full download
succeeds -- safe to re-run / leave in a tmux session.

Usage:
    export HF_TOKEN='hf_...'
    python3 tools/extract/download_sensenova_si8m.py
"""
import os
import time

os.environ.setdefault("HF_HOME", "/p/data1/mmlaion/nguyen38/hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # Xet backend flaky on this cluster, see project memory

from huggingface_hub import HfApi, login, snapshot_download

REPO_ID = "sensenova/SenseNova-SI-8M"
LOCAL_DIR = "/p/data1/mmlaion/shared/vla/sensenova_si8m"
MAX_WORKERS = 16
RETRY_DELAY_SEC = 30
ALLOW_PATTERNS = ["*.zip", "SenseNova-SI-8M.parquet"]


def report_progress(local_dir):
    if not os.path.isdir(local_dir):
        print("  (nothing downloaded yet)", flush=True)
        return
    zips_done = sorted(f for f in os.listdir(local_dir) if f.endswith(".zip"))
    parquet_done = os.path.exists(os.path.join(local_dir, "SenseNova-SI-8M.parquet"))
    size_gb = sum(os.path.getsize(os.path.join(local_dir, f)) for f in os.listdir(local_dir)
                  if os.path.isfile(os.path.join(local_dir, f))) / 1e9
    print(f"  progress: {len(zips_done)}/53 zip parts, parquet={'yes' if parquet_done else 'no'}, "
          f"{size_gb:.1f} GB on disk so far", flush=True)


def main():
    if "HF_TOKEN" not in os.environ:
        raise EnvironmentError("HF_TOKEN not set. Run: export HF_TOKEN='hf_...'")
    login(token=os.environ["HF_TOKEN"])

    os.makedirs(LOCAL_DIR, exist_ok=True)
    print(f"Target: {LOCAL_DIR}  (~1.13TB total, 53 image zips + 1 metadata parquet)", flush=True)

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
    print(f"\nDone. Next step (not run by this script): extract the 53 zips into an images/ dir.")


if __name__ == "__main__":
    main()
