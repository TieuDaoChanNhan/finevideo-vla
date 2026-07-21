#!/usr/bin/env python3
"""
Count VLA token types (seed2/cosmos/avclm/snac/agent) + approx text-word
count across already-flattened, training-ready datasets. Uses the exact
same PATTERNS/count_tokens regex convention as data_inventory.py in this
same directory, so results are directly comparable to that script's
figures (and to numbers already recorded in PROGRESS_VI.md).

Verified 20/07/2026 against 4 datasets (see PROGRESS_VI.md entry for the
full comparison table):
  - mv-omni (valid_snac converted): 6.933B tokens -- matches the "+6.93B
    token" figure documented from the June data-inventory investigation.
  - finevideo-vla v5: 5.226B tokens -- matches the documented "5.256B
    tokens" within ~0.03B (the gap is because `text` here is an approximate
    whitespace-split word count, not a real tokenizer pass -- see below).
  - avclm always counts 0 across every dataset checked so far: this is
    expected, not a bug -- every flatten script in this project (FineVideo's
    pipeline_pose/phase7_flatten.py, OmniVideo's flatten_step_a_video.py/
    phase6_merge_omnivideo.py) always discards the avc_lm payload in its
    final output; avc_lm tokens only ever exist in pre-flatten raw Step A
    data (see tools/decode/decode_avclm.py's docstring).

Caveat: `text` is counted via simple whitespace-split (excluding pieces that
look like a single `<...>` tag), same approximation data_inventory.py uses
-- it is NOT a real BPE token count, so it under/over-counts vs. what the
actual tokenizer produces for prose (captions, QA, speech transcripts).
Only useful as a rough cross-dataset comparison, not as an exact total for
Megatron sequence-length planning.

Usage:
    # Default: count the 4 datasets this was verified against
    python tools/inventory/count_flattened_tokens.py

    # Custom dataset(s): name=glob-pattern (repeatable)
    python tools/inventory/count_flattened_tokens.py \
        --dataset "my-dataset=/path/to/shards/*.jsonl" --workers 16
"""
import argparse
import glob
import gzip
import json
import multiprocessing as mp
import os
import re
import sys
import time

PATTERNS = {
    'seed2':  re.compile(r'<seed2_\d+>'),
    'seed':   re.compile(r'<seed_\d+>'),        # MV-Omni uses <seed_N>, not <seed2_N>
    'cosmos': re.compile(r'<cosmos_\d+>'),
    'avclm':  re.compile(r'<avclm_\d+>|<avc_lm_\d+>'),
    'snac':   re.compile(r'<snac_\d+>'),
    'agent':  re.compile(r'<fps_\d+>|<[a-z_]+_[txyz]_\d+>'),
}
TOKEN_TYPES = list(PATTERNS.keys()) + ['text']
_ANGLE_RE = re.compile(r'^<[^>]+>$')

DEFAULT_DATASETS = {
    "emotional-roleplay": "/p/data1/mmlaion/shared/vla/laion_emotional_roleplay/flattened/*.jsonl",
    "omnivideo-100k-final": "/p/data1/mmlaion/shared/vla/omnivideo_100k_final/*.jsonl",
    "mv-omni-snac": "/p/data1/mmlaion/shared/vla/mv_omni_converted/*.jsonl.gz",
    "finevideo-vla-v5": "/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/megatron_dataset_v5/*.jsonl",
}


def count_tokens(text: str) -> dict:
    c = {}
    for k, pat in PATTERNS.items():
        c[k] = len(pat.findall(text))
    c['text'] = sum(1 for w in text.split() if w and not _ANGLE_RE.match(w))
    return c


def zero_counts() -> dict:
    return {k: 0 for k in TOKEN_TYPES}


def add_counts(dst: dict, src: dict) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + v


def open_any(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def count_file(path: str):
    c = zero_counts()
    n_records = 0
    try:
        with open_any(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                add_counts(c, count_tokens(d.get("text", "")))
                n_records += 1
    except Exception as e:
        print(f"ERROR {path}: {e}", file=sys.stderr)
    return path, c, n_records


def count_dataset(name: str, files: list, num_workers: int = 24):
    t0 = time.time()
    total = zero_counts()
    total_records = 0
    with mp.Pool(num_workers) as pool:
        for i, (path, c, n) in enumerate(pool.imap_unordered(count_file, files), 1):
            add_counts(total, c)
            total_records += n
            if i % 20 == 0 or i == len(files):
                print(f"  [{name}] {i}/{len(files)} files done...", flush=True)
    elapsed = time.time() - t0
    print(f"\n=== {name} === ({elapsed:.0f}s, {len(files)} files, {total_records:,} records)")
    grand_total = sum(total.values())
    for k in TOKEN_TYPES:
        print(f"  {k:8s}: {total[k]:>15,}")
    print(f"  {'TOTAL':8s}: {grand_total:>15,}")
    return total, total_records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", default=[],
                     help="name=glob-pattern, repeatable. Omit to use the 4 default datasets.")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    if args.dataset:
        datasets = {}
        for spec in args.dataset:
            name, _, pattern = spec.partition("=")
            datasets[name] = pattern
    else:
        datasets = DEFAULT_DATASETS

    results = {}
    for name, pattern in datasets.items():
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"=== {name} === NO FILES FOUND ({pattern}), skipping")
            continue
        results[name] = count_dataset(name, files, num_workers=args.workers)

    print("\n\n### SUMMARY (grand total tokens per dataset) ###")
    for name, (total, n_records) in results.items():
        gt = sum(total.values())
        print(f"{name}: {gt:,} tokens ({gt/1e9:.3f}B), {n_records:,} records")


if __name__ == "__main__":
    main()
