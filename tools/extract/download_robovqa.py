#!/usr/bin/env python3
"""
Download Tianli/robovqa from HuggingFace Hub -- unofficial mirror of Google
DeepMind's RoboVQA (official repo only ships a Colab notebook pulling from a
GCS bucket, no direct HF snapshot; this mirror is the practical way in).

Verified via HF tree API 18/07/2026:
  - LICENSE.txt (real Apache-2.0 text, matches official github.com/google-deepmind/robovqa
    dual-license statement: "software: Apache 2.0, other materials: CC-BY-4.0")
  - instructions/*.txt        -- long/medium-horizon instruction text
  - json/train/data-*-of-00160.json  (181 files, ~0.24GB) -- QA/instruction records
  - *.mp4                     (9,999 files, ~3.4GB)  -- real video clips
  - tfrecord-*-of-00175 (+ a second -of-00009 shard set)  (~67GB) -- RLDS-style
    packed episodes (video+text), the bulk of the real data

Total real size: ~70.8GB. This is a third-party mirror, not the official
google-deepmind release -- the LICENSE.txt file matches the official Apache-2.0
statement, but treat this as "verified good-faith mirror", not "official
source", when reporting provenance upstream.

Must run from a JUWELS login node (compute nodes have no internet).
Resumable: snapshot_download skips files already complete in local_dir, and
this script retries on transient network errors until the full download
succeeds -- safe to re-run / leave in a tmux session.

Usage:
    export HF_TOKEN='hf_...'
    python3 tools/extract/download_robovqa.py
"""
import os
import time

os.environ.setdefault("HF_HOME", "/p/data1/mmlaion/nguyen38/hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # Xet backend flaky on this cluster, see project memory

from huggingface_hub import login, snapshot_download

REPO_ID = "Tianli/robovqa"
LOCAL_DIR = "/p/data1/mmlaion/shared/vla/robovqa"
MAX_WORKERS = 16
RETRY_DELAY_SEC = 30
# Pull everything except .gitattributes -- data is a mix of mp4/tfrecord/json/txt,
# no single glob captures it all cleanly.
IGNORE_PATTERNS = [".gitattributes"]


def report_progress(local_dir):
    if not os.path.isdir(local_dir):
        print("  (nothing downloaded yet)", flush=True)
        return
    n_files = 0
    size_bytes = 0
    for root, _, files in os.walk(local_dir):
        for f in files:
            n_files += 1
            size_bytes += os.path.getsize(os.path.join(root, f))
    print(f"  progress: {n_files} files, {size_bytes/1e9:.1f} GB on disk so far", flush=True)


def main():
    if "HF_TOKEN" not in os.environ:
        raise EnvironmentError("HF_TOKEN not set. Run: export HF_TOKEN='hf_...'")
    login(token=os.environ["HF_TOKEN"])

    os.makedirs(LOCAL_DIR, exist_ok=True)
    print(f"Target: {LOCAL_DIR}  (~70.8GB total: mp4 clips + tfrecord shards + json/txt instructions)", flush=True)

    attempt = 0
    while True:
        attempt += 1
        print(f"\n[attempt {attempt}] snapshot_download starting -> {LOCAL_DIR}", flush=True)
        report_progress(LOCAL_DIR)
        try:
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                ignore_patterns=IGNORE_PATTERNS,
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
    print(f"\nDone. Next step (not run by this script): inspect tfrecord schema "
          f"(RLDS-style, needs tensorflow to read) before deciding how to feed Step A.")


if __name__ == "__main__":
    main()
