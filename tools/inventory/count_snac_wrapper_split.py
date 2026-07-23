#!/usr/bin/env python3
"""
Split <snac_N> token counts by which wrapper tag they appear inside:
<listen>...</listen> (ambient/scene audio the model perceives -- FineVideo,
OmniVideo-100K, MV-Omni) vs <speak>...</speak> (a generated reply --
emotional-roleplay). Also flags any <snac_N> found outside both wrappers
("orphan"), which should not happen in any current dataset and would
indicate a flatten bug if found.

Usage:
    python tools/inventory/count_snac_wrapper_split.py \
        --dataset "name=glob-pattern" [--dataset ...] --workers 24
"""
import argparse
import glob
import gzip
import json
import multiprocessing as mp
import re
import sys

SNAC = re.compile(r'<snac_\d+>')
LISTEN_BLOCK = re.compile(r'<listen>(.*?)</listen>', re.S)
SPEAK_BLOCK = re.compile(r'<speak>(.*?)</speak>', re.S)


def open_any(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def count_file(path):
    listen_n = speak_n = total_n = 0
    try:
        with open_any(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    text = json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    continue
                total_n += len(SNAC.findall(text))
                for m in LISTEN_BLOCK.finditer(text):
                    listen_n += len(SNAC.findall(m.group(1)))
                for m in SPEAK_BLOCK.finditer(text):
                    speak_n += len(SNAC.findall(m.group(1)))
    except Exception as e:
        print(f"ERROR {path}: {e}", file=sys.stderr)
    return listen_n, speak_n, total_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True, help="name=glob-pattern, repeatable")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    grand_listen = grand_speak = grand_total = grand_orphan = 0
    for spec in args.dataset:
        name, _, pattern = spec.partition("=")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"{name}: NO FILES ({pattern})")
            continue
        listen_n = speak_n = total_n = 0
        with mp.Pool(args.workers) as pool:
            for l, s, t in pool.imap_unordered(count_file, files):
                listen_n += l
                speak_n += s
                total_n += t
        orphan = total_n - listen_n - speak_n
        print(f"{name}: listen={listen_n:,} speak={speak_n:,} orphan={orphan:,} (raw <snac_N> total={total_n:,})")
        grand_listen += listen_n
        grand_speak += speak_n
        grand_total += total_n
        grand_orphan += orphan

    print("\n=== GRAND TOTAL across all --dataset args ===")
    print(f"listen={grand_listen:,}  speak={grand_speak:,}  orphan={grand_orphan:,}  total={grand_total:,}")


if __name__ == "__main__":
    main()
