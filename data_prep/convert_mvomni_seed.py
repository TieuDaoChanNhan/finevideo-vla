#!/usr/bin/env python3
"""
convert_mvomni_seed.py — Convert MV-Omni <seed_N> tokens to <seed2_N>.

MV-Omni uses <seed_N> (N: 0–8191) for image tokens, identical vocab to our
<seed2_N> but different prefix. This script does a streaming in-place rename
so the data is compatible with our existing tokenizer without vocab expansion.

Input:  /p/data1/mmlaion/nguyen38/inventory_cache/hf_snac/valid_snac_*.jsonl.gz
Output: /p/data1/mmlaion/shared/vla/mv_omni_converted/mv_omni_snac_*.jsonl.gz

Usage:
    python data_prep/convert_mvomni_seed.py               # all 6 files
    python data_prep/convert_mvomni_seed.py --shard 0     # single file (testing)
    python data_prep/convert_mvomni_seed.py --workers 6   # parallel (default: 6)
"""

import argparse
import gzip
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

INPUT_DIR  = "/p/data1/mmlaion/nguyen38/inventory_cache/hf_snac"
OUTPUT_DIR = "/p/data1/mmlaion/shared/vla/mv_omni_converted"

# Matches <seed_0> through <seed_8191>
_SEED_RE = re.compile(r'<seed_(\d+)>')


def convert_text(text: str) -> tuple[str, int]:
    """Replace <seed_N> → <seed2_N>. Returns (converted_text, count_replaced)."""
    count = 0

    def _replace(m):
        nonlocal count
        count += 1
        return f"<seed2_{m.group(1)}>"

    return _SEED_RE.sub(_replace, text), count


def process_shard(shard_idx: int) -> dict:
    in_path  = Path(INPUT_DIR)  / f"valid_snac_{shard_idx}.jsonl.gz"
    out_path = Path(OUTPUT_DIR) / f"mv_omni_snac_{shard_idx}.jsonl.gz"

    if not in_path.exists():
        return {"shard": shard_idx, "error": f"not found: {in_path}"}

    out_path.parent.mkdir(parents=True, exist_ok=True)

    records     = 0
    seeds_total = 0
    t0          = time.time()

    with gzip.open(in_path, "rt", encoding="utf-8") as fin, \
         gzip.open(out_path, "wt", encoding="utf-8", compresslevel=5) as fout:

        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue

            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = d.get("text", "")
            if "<seed_" in text:
                d["text"], n = convert_text(text)
                seeds_total += n

            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
            records += 1

            if records % 50_000 == 0:
                elapsed = time.time() - t0
                print(f"  shard {shard_idx}: {records:,} records, "
                      f"{seeds_total:,} seeds converted, {elapsed:.0f}s",
                      flush=True)

    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / 1e6
    return {
        "shard":   shard_idx,
        "records": records,
        "seeds":   seeds_total,
        "size_mb": size_mb,
        "elapsed": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard",   type=int, default=None,
                    help="Process only this shard index (0–5)")
    ap.add_argument("--workers", type=int, default=6,
                    help="Parallel workers (default: 6, one per file)")
    args = ap.parse_args()

    shards = [args.shard] if args.shard is not None else list(range(6))

    print(f"Converting {len(shards)} shard(s): {shards}")
    print(f"  input:  {INPUT_DIR}")
    print(f"  output: {OUTPUT_DIR}")
    print()

    results = []

    if len(shards) == 1 or args.workers == 1:
        for s in shards:
            print(f"Processing shard {s}...")
            r = process_shard(s)
            results.append(r)
            if "error" in r:
                print(f"  ERROR: {r['error']}")
            else:
                print(f"  done: {r['records']:,} records, "
                      f"{r['seeds']:,} seed tokens converted, "
                      f"{r['size_mb']:.0f} MB, {r['elapsed']:.0f}s")
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(shards))) as ex:
            futures = {ex.submit(process_shard, s): s for s in shards}
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                if "error" in r:
                    print(f"shard {r['shard']} ERROR: {r['error']}")
                else:
                    print(f"shard {r['shard']} done: {r['records']:,} records, "
                          f"{r['seeds']:,} seed tokens converted, "
                          f"{r['size_mb']:.0f} MB, {r['elapsed']:.0f}s",
                          flush=True)

    print()
    total_records = sum(r.get("records", 0) for r in results)
    total_seeds   = sum(r.get("seeds",   0) for r in results)
    total_mb      = sum(r.get("size_mb", 0) for r in results)
    print(f"TOTAL: {total_records:,} records | "
          f"{total_seeds:,} seed→seed2 conversions | "
          f"{total_mb/1000:.1f} GB output")
    print(f"Output: {OUTPUT_DIR}/mv_omni_snac_*.jsonl.gz")


if __name__ == "__main__":
    main()
