#!/usr/bin/env python3
"""
Download synth_llava.tar.gz + synth_llava2.tar.gz from
mixture-vitae-backup/MixtureVitae-Backup/data/multimodal (HF).

Context: this path was investigated Jul 9, 2026 (see
investigations/mixturevitae_multimodal/README.md and PROGRESS.md's
"MixtureVitae-Backup Multimodal Investigation" section) via streaming peek,
no full download -- 13/15 files turned out to be plain text/caption corpora,
2 files had raw SNAC audio tokens. That investigation stopped just short of
these 2 files. Huu flagged them directly on Discord (Jul 19-20, 2026),
noting the .wds members inside are WebDataset shards made for streaming.

Peek result (tools/inventory/peek_multimodal.py --only synth_llava2.tar.gz,
2026-07-20): outer tar contains per-shard *.jsonl caption/metadata files
(`{"text": "<caption><image_0>...</caption>", "metadata": {"source":
"llava_pretrain|shard-NNNNNNN|create_multimodal_data.generate_images_then_captions...`)
plus the paired *.wds shards (~382MB each, not peeked -- binary webdataset,
presumably tar-packed images). Synthetic LLaVA-style image+caption data, no
VLA tokens in the outer text. synth_llava.tar.gz (not yet peeked -- its
result was overwritten by the synth_llava2.tar.gz peek run, see peek script's
note about non-additive reports) is presumed to be the same format/source
given the matching name.

Total: ~56.2GB (20.7GB + 35.5GB per HF file listing).

Must run from a login node with internet access (confirmed reachable from
this JUPITER login node, jpbl-s01-02 -- unlike the JUWELS-only assumption in
older download scripts in this file, no need to switch clusters here).
Resumable: hf_hub_download skips a file already complete in local_dir.

Usage:
    export HF_TOKEN='hf_...'
    python3 tools/extract/download_synth_llava.py
"""
import os
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # Xet backend flaky on this cluster, see project memory

from huggingface_hub import hf_hub_download

REPO_ID = "mixture-vitae-backup/MixtureVitae-Backup"
REPO_TYPE = "dataset"
FILES = [
    "data/multimodal/synth_llava.tar.gz",
    "data/multimodal/synth_llava2.tar.gz",
]
LOCAL_DIR = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/mixturevitae_multimodal/synth_llava"
MAX_RETRIES = 20
RETRY_DELAY_SEC = 30


def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None

    for rel_path in FILES:
        fname = os.path.basename(rel_path)
        dest = os.path.join(LOCAL_DIR, fname)
        if os.path.exists(dest):
            size_gb = os.path.getsize(dest) / 1e9
            print(f"[skip] {fname} already present locally ({size_gb:.1f}GB) -- "
                  f"delete it first if you suspect a partial/corrupt download.")
            continue

        print(f"[download] {rel_path} -> {dest}")
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type=REPO_TYPE,
                    filename=rel_path,
                    local_dir=LOCAL_DIR,
                    token=token,
                )
                print(f"[done] {fname}")
                break
            except Exception as e:
                print(f"  attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(RETRY_DELAY_SEC)

    print("\nAll files present:")
    for rel_path in FILES:
        fname = os.path.basename(rel_path)
        dest = os.path.join(LOCAL_DIR, fname)
        if os.path.exists(dest):
            print(f"  {fname}: {os.path.getsize(dest)/1e9:.1f}GB")


if __name__ == "__main__":
    main()
