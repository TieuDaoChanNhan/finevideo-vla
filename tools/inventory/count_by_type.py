#!/usr/bin/env python3
"""
Per-dataset token count broken down by exactly 5 types: agent, cosmos, seed2,
listen (<snac_N> inside <listen>...</listen>), speak (<snac_N> inside
<speak>...</speak>). <listen>/<speak> are wrapper tags, not counted
themselves -- only the <snac_N> tokens inside them are counted, split by
which wrapper they're in.

Handles two record shapes:
  - flat: {"text": "..."}  (OmniVideo-100K, roleplay, harmony4d, synth_llava, mv-omni)
  - FineVideo hierarchical: {"scenes": [{"activities": [{"video_tokens": "..."}]}]}
    (Phase 6 merge output, pre-Phase7-flatten -- video_tokens per activity
    concatenated as the text to scan)

Usage:
    python tools/inventory/count_by_type.py --dataset "name=glob[:hier]" ...
    Append ":hier" to a dataset spec to use the FineVideo hierarchical reader.
"""
import argparse
import glob
import gzip
import json
import multiprocessing as mp
import re
import sys

AGENT = re.compile(r'<fps_\d+>|<[a-z_]+_[txyz]_\d+>')
COSMOS = re.compile(r'<cosmos_\d+>')
SEED2 = re.compile(r'<seed2_\d+>')
SNAC = re.compile(r'<snac_\d+>')
LISTEN_BLOCK = re.compile(r'<listen>(.*?)</listen>', re.S)
SPEAK_BLOCK = re.compile(r'<speak>(.*?)</speak>', re.S)

# Pre-Phase7-flatten FineVideo: seed2/cosmos are still raw space-separated
# integers inside container tags, not yet atomic <seed2_N>/<cosmos_N> --
# Phase 7 (phase7_flatten.py) does that conversion. Count raw ints here so
# Phase-6-only output still gives a real (pre-dropout) cosmos/seed2 total.
RAW_SEED2_BLOCK = re.compile(r'<seed2>\s*(.*?)\s*</seed2>', re.S)
RAW_COSMOS_BLOCK = re.compile(r'<cosmos>\s*(.*?)\s*</cosmos>', re.S)


def open_any(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def count_text(text, c):
    c["agent"] += len(AGENT.findall(text))
    c["cosmos"] += len(COSMOS.findall(text))
    c["seed2"] += len(SEED2.findall(text))
    for m in LISTEN_BLOCK.finditer(text):
        c["listen"] += len(SNAC.findall(m.group(1)))
    for m in SPEAK_BLOCK.finditer(text):
        c["speak"] += len(SNAC.findall(m.group(1)))


def zero():
    return {"agent": 0, "cosmos": 0, "seed2": 0, "listen": 0, "speak": 0}


def count_file_flat(path):
    c = zero()
    n = 0
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
                count_text(d.get("text", ""), c)
                n += 1
    except Exception as e:
        print(f"ERROR {path}: {e}", file=sys.stderr)
    return c, n


def count_text_raw_pre_flatten(text, c):
    for m in RAW_SEED2_BLOCK.finditer(text):
        c["seed2"] += len(m.group(1).split())
    for m in RAW_COSMOS_BLOCK.finditer(text):
        c["cosmos"] += len(m.group(1).split())


def count_file_hier(path):
    c = zero()
    n = 0
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
                n += 1
                for scene in d.get("scenes", []):
                    for act in scene.get("activities", []):
                        vt = act.get("video_tokens", "")
                        count_text(vt, c)  # agent, listen, speak (already atomic pre-Phase7)
                        count_text_raw_pre_flatten(vt, c)  # seed2, cosmos (still raw ints pre-Phase7)
    except Exception as e:
        print(f"ERROR {path}: {e}", file=sys.stderr)
    return c, n


def add(dst, src):
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True,
                     help="name=glob-pattern, optionally suffixed :hier for FineVideo-style hierarchical records")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    header = f"{'dataset':28s} {'agent':>14s} {'cosmos':>14s} {'seed2':>14s} {'listen':>14s} {'speak':>14s} {'TOTAL':>15s}"
    print(header)
    print("-" * len(header))

    grand = zero()
    for spec in args.dataset:
        name, _, rest = spec.partition("=")
        hier = rest.endswith(":hier")
        pattern = rest[:-5] if hier else rest
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"{name}: NO FILES ({pattern})")
            continue
        fn = count_file_hier if hier else count_file_flat
        total = zero()
        n_records = 0
        with mp.Pool(args.workers) as pool:
            for c, n in pool.imap_unordered(fn, files):
                add(total, c)
                n_records += n
        add(grand, total)
        tot = sum(total.values())
        print(f"{name:28s} {total['agent']:>14,} {total['cosmos']:>14,} {total['seed2']:>14,} "
              f"{total['listen']:>14,} {total['speak']:>14,} {tot:>15,}   ({n_records:,} records)")

    print("-" * len(header))
    gtot = sum(grand.values())
    print(f"{'GRAND TOTAL':28s} {grand['agent']:>14,} {grand['cosmos']:>14,} {grand['seed2']:>14,} "
          f"{grand['listen']:>14,} {grand['speak']:>14,} {gtot:>15,}")


if __name__ == "__main__":
    main()
