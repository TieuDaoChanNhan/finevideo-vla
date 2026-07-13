#!/usr/bin/env python3
"""
Download mlfoundations/MINT-1T-HTML (data_v1_1 config) from HuggingFace Hub.

6,159 parquet shards, ~2.89TB total (measured via HF tree API 13/07/2026,
not the 5.91TB figure quoted on the dataset page -- that includes PDF/ArXiv
splits not present in this HTML-only repo).

Must run from a JUWELS login node (compute nodes have no internet).
Resumable: snapshot_download skips files already complete in local_dir,
and this script retries on transient network errors until the full
snapshot succeeds, so it's safe to re-run / leave in a tmux session.

Usage:
    python3 tools/extract/download_mint1t_html.py
"""
import os
import time

os.environ.setdefault("HF_HOME", "/p/data1/mmlaion/nguyen38/hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # Xet backend flaky on this cluster, see project memory

from huggingface_hub import snapshot_download

REPO_ID = "mlfoundations/MINT-1T-HTML"
LOCAL_DIR = "/p/data1/mmlaion/shared/vla/mint1t_html"
MAX_WORKERS = 16
RETRY_DELAY_SEC = 30

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    attempt = 0
    while True:
        attempt += 1
        print(f"[attempt {attempt}] snapshot_download starting -> {LOCAL_DIR}", flush=True)
        try:
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                allow_patterns=["data_v1_1/*.parquet"],
                local_dir=LOCAL_DIR,
                max_workers=MAX_WORKERS,
            )
            print("snapshot_download completed successfully.", flush=True)
            break
        except Exception as e:
            print(f"[attempt {attempt}] failed: {e!r} -- retrying in {RETRY_DELAY_SEC}s", flush=True)
            time.sleep(RETRY_DELAY_SEC)

if __name__ == "__main__":
    main()
