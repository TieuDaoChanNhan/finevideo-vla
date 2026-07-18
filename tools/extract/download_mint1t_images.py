#!/usr/bin/env python3
"""
Download images referenced in MINT-1T-HTML manifest JSONL (see
extract_mint1t_manifest.py) into bucketed folders, with per-domain rate
limiting (the corpus is dominated by a handful of blogspot/blogger hosts --
hammering one domain risks getting the whole crawl blocked) and full resume
via a per-shard status file.

Folder layout (avoids one-folder-per-record: 850M records would blow up
filesystem inode/dir-entry limits):
    {image_dir}/{shard_stem}/{record_idx // BUCKET_SIZE}/{record_idx}_{img_pos}.{ext}

Status file (resume + audit trail), one line per attempted image:
    {image_dir}/_status/{shard_stem}.jsonl
    {"record_id":..., "img_pos":..., "url":..., "status":"ok"|"fail",
     "path":..., "size":..., "error":...}

Must run from a JUWELS login node (compute nodes have no internet). Requires
network access, so run in tmux, e.g.:
    tmux new -s mint_images
    source activate_env_tools.sh
    python3 tools/extract/download_mint1t_images.py --num-shards 20

Re-running skips (record_id, img_pos) pairs already present in the status
file, whether they succeeded or failed -- pass --retry-failed to re-attempt
only the failed ones.
"""
import argparse
import glob
import json
import os
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

MANIFEST_DIR_DEFAULT = "/p/data1/mmlaion/shared/vla/mint1t_html/manifest"
IMAGE_DIR_DEFAULT = "/p/data1/mmlaion/shared/vla/mint1t_html/images"
BUCKET_SIZE = 1000
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"


class DomainLimiter:
    """Caps concurrent in-flight requests per domain (default 8) instead of
    serializing with a fixed delay. Pilot run (18/07) measured ~10 img/s
    aggregate with the old serialize-per-domain design regardless of
    --max-workers=64, because MINT-1T-HTML's images are dominated by a
    handful of shared blogspot CDN hosts (1-4.bp.blogspot.com) -- serializing
    on exact domain effectively capped the whole download to ~4 hosts x 2/s.
    A per-domain semaphore still protects small/fragile blogs from being
    hammered by all 64 workers at once, but lets big shared hosts run at
    real concurrency."""

    def __init__(self, per_domain_concurrency):
        self.per_domain_concurrency = per_domain_concurrency
        self.semaphores = {}
        self._guard = threading.Lock()

    def _sem_for(self, domain):
        with self._guard:
            sem = self.semaphores.get(domain)
            if sem is None:
                sem = threading.Semaphore(self.per_domain_concurrency)
                self.semaphores[domain] = sem
            return sem

    def acquire(self, domain):
        sem = self._sem_for(domain)
        sem.acquire()
        return sem


def guess_ext(url):
    path = urllib.parse.urlparse(url).path
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"):
        return ext
    return ".jpg"


def load_done_set(status_path, retry_failed):
    done = set()
    if not os.path.exists(status_path):
        return done
    with open(status_path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if retry_failed and d.get("status") == "fail":
                continue
            done.add((d["record_id"], d["img_pos"]))
    return done


def download_one(record_id, img_pos, url, local_path, limiter, timeout, session):
    domain = urllib.parse.urlparse(url).netloc
    sem = limiter.acquire(domain)
    try:
        resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout, stream=True)
        resp.raise_for_status()
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        size = 0
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
                size += len(chunk)
        return {"record_id": record_id, "img_pos": img_pos, "url": url, "status": "ok",
                "path": local_path, "size": size}
    except Exception as e:
        return {"record_id": record_id, "img_pos": img_pos, "url": url, "status": "fail",
                "error": repr(e)[:200]}
    finally:
        sem.release()


def process_shard(manifest_path, image_dir, status_dir, limiter, max_workers, timeout, retry_failed):
    stem = os.path.splitext(os.path.basename(manifest_path))[0]
    status_path = os.path.join(status_dir, f"{stem}.jsonl")
    done = load_done_set(status_path, retry_failed)

    tasks = []
    with open(manifest_path) as f:
        for line in f:
            rec = json.loads(line)
            record_id = rec["record_id"]
            row_idx = int(record_id.rsplit("_", 1)[-1])
            bucket = row_idx // BUCKET_SIZE
            for pos, url in enumerate(rec["images"]):
                if not url:
                    continue
                if (record_id, pos) in done:
                    continue
                ext = guess_ext(url)
                local_path = os.path.join(image_dir, stem, str(bucket), f"{row_idx}_{pos}{ext}")
                tasks.append((record_id, pos, url, local_path))

    if not tasks:
        print(f"{stem}: nothing to do (already complete or empty)", flush=True)
        return 0, 0

    ok = 0
    fail = 0
    t0 = time.time()
    session = requests.Session()
    with open(status_path, "a") as status_f, ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(download_one, rid, pos, url, path, limiter, timeout, session)
                   for rid, pos, url, path in tasks]
        for i, fut in enumerate(as_completed(futures)):
            result = fut.result()
            status_f.write(json.dumps(result) + "\n")
            if result["status"] == "ok":
                ok += 1
            else:
                fail += 1
            if (i + 1) % 5000 == 0:
                status_f.flush()
                rate = (i + 1) / (time.time() - t0)
                print(f"{stem}: {i+1}/{len(tasks)} ({rate:.1f} img/s, ok={ok} fail={fail})", flush=True)

    elapsed = time.time() - t0
    print(f"{stem}: DONE {len(tasks)} images in {elapsed:.0f}s -- ok={ok} fail={fail} "
          f"({ok/len(tasks)*100:.1f}% success)", flush=True)
    return ok, fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", default=MANIFEST_DIR_DEFAULT)
    ap.add_argument("--image-dir", default=IMAGE_DIR_DEFAULT)
    ap.add_argument("--num-shards", type=int, default=None, help="pilot: only first N manifest shards")
    ap.add_argument("--shard-list", default=None, help="file with one manifest filename per line")
    ap.add_argument("--max-workers", type=int, default=64)
    ap.add_argument("--per-domain-concurrency", type=int, default=8,
                     help="max simultaneous in-flight requests to the same domain")
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()

    status_dir = os.path.join(args.image_dir, "_status")
    os.makedirs(status_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.manifest_dir, "*.jsonl")))
    if args.shard_list:
        wanted = set(open(args.shard_list).read().split())
        files = [f for f in files if os.path.basename(f) in wanted]
    elif args.num_shards:
        files = files[:args.num_shards]

    print(f"Downloading images for {len(files)} manifest shard(s) -> {args.image_dir}", flush=True)
    limiter = DomainLimiter(args.per_domain_concurrency)

    total_ok = 0
    total_fail = 0
    for i, path in enumerate(files):
        print(f"\n--- shard {i+1}/{len(files)}: {os.path.basename(path)} ---", flush=True)
        ok, fail = process_shard(path, args.image_dir, status_dir, limiter,
                                  args.max_workers, args.timeout, args.retry_failed)
        total_ok += ok
        total_fail += fail

    total = total_ok + total_fail
    rate = (total_ok / total * 100) if total else 0.0
    print(f"\n=== All shards done: {total_ok:,} ok / {total_fail:,} fail ({rate:.1f}% success) ===")


if __name__ == "__main__":
    main()
