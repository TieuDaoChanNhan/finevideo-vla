#!/usr/bin/env python3
"""
peek_multimodal.py — Quick structural probe of the 15 files in
mixture-vitae-backup/MixtureVitae-Backup/data/multimodal (HF).

Streams just the first few records/members of each file over HTTP — no full
download, no local temp file — to discover: jsonl.gz vs tar.gz, field/member
names, and whether VLA tokens (<seed2_>, <cosmos_>, <snac_>, <avclm_>, ...)
are present. Run this before count_multimodal_tokens.py so the per-file
parsing logic there can be adjusted to whatever structure is actually found.

Usage:
    python peek_multimodal.py                        # all 15 files
    python peek_multimodal.py --only youtube.tar.gz   # a single file
"""
import argparse
import gzip
import json
import os
import sys
import tarfile
import time
import zlib

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_inventory import PATTERNS, _hf_token, hf_url  # reuse existing regex + auth

REPO = "mixture-vitae-backup/MixtureVitae-Backup"
PREFIX = "data/multimodal"

# Ordered smallest -> largest so cheap files finish (and get flagged) first.
FILES = [
    "coco.tar.gz",
    "finevideo_transcripts.jsonl.gz",
    "clappa.tar.gz",
    "low_nemo_maga.tar.gz",
    "valid_data_snac.jsonl.gz",
    "europarl.tar.gz",
    "magalith-10m-florence2.jsonl.gz",
    "emo.jsonl.gz",
    "stack_maga.tar.gz",
    "youtube.tar.gz",
    "valid_text_only.tar.gz",
    "high_stack.tar.gz",
    "train_data_snac.jsonl.gz",
    "synth_llava.tar.gz",
    "synth_llava2.tar.gz",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(SCRIPT_DIR, "multimodal_peek_report.json")

MAX_JSONL_LINES = 5
MAX_TAR_MEMBERS = 5
TEXT_EXTENSIONS = {".json", ".jsonl", ".txt"}
BINARY_EXTENSIONS = {".png", ".jpg", ".jpeg", ".ogg", ".wav", ".mp4", ".webp", ".mp3", ".flac"}


def _open_stream(filename: str):
    url = hf_url(REPO, f"{PREFIX}/{filename}")
    token = _hf_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(url, stream=True, headers=headers, timeout=60)
    r.raise_for_status()
    r.raw.decode_content = True
    return r


def _flag_tokens(text: str):
    return [name for name, pat in PATTERNS.items() if pat.search(text)]


PEEK_RAW_BYTES_CAP = 3_000_000  # compressed bytes read for peek — bounds pathological huge-line files


def peek_jsonl_gz(filename: str) -> dict:
    """
    Reads at most PEEK_RAW_BYTES_CAP *compressed* bytes and decompresses
    incrementally (zlib, gzip window) rather than iterating gzip.GzipFile
    line-by-line — a single oversized JSONL line (e.g. one record holding a
    long SNAC token sequence) would otherwise force GzipFile to decompress
    an unbounded amount of the stream before yielding a line.
    """
    result = {"format": "jsonl.gz", "records": [], "flag": "unknown", "vla_types_seen": [],
              "raw_bytes_read": 0, "truncated": False}
    r = _open_stream(filename)
    try:
        raw = r.raw.read(PEEK_RAW_BYTES_CAP)
        result["raw_bytes_read"] = len(raw)
        result["truncated"] = len(raw) >= PEEK_RAW_BYTES_CAP
    finally:
        r.close()

    d = zlib.decompressobj(zlib.MAX_WBITS | 16)  # 16+MAX_WBITS = gzip container
    try:
        decompressed = d.decompress(raw)
    except zlib.error as e:
        result["flag"] = "error"
        result["error"] = f"decompress failed within {len(raw)} raw bytes: {e}"
        return result

    text = decompressed.decode("utf-8", errors="replace")
    lines = text.split("\n")
    if result["truncated"] and lines:
        lines = lines[:-1]  # last line is likely mid-record when we cut the stream short

    vla_hits = set()
    n = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            result["records"].append({"unparsed_preview": line[:500]})
            n += 1
            if n >= MAX_JSONL_LINES:
                break
            continue
        if isinstance(rec, dict):
            sample = {k: str(v)[:200] for k, v in rec.items()}
            result["records"].append({"keys": list(rec.keys()), "sample": sample})
            blob = json.dumps(rec)[:5000]
            vla_hits.update(_flag_tokens(blob))
        n += 1
        if n >= MAX_JSONL_LINES:
            break

    if not result["records"] and result["truncated"]:
        # Not even one full line within the byte cap — surface a raw preview instead of "text-only".
        result["flag"] = "unresolved-line-too-large"
        result["raw_preview"] = text[:1000]
    else:
        result["flag"] = "has-vla-tokens" if vla_hits else "text-only"
    result["vla_types_seen"] = sorted(vla_hits)
    return result


def peek_tar_gz(filename: str) -> dict:
    result = {"format": "tar.gz", "members": [], "flag": "unknown", "vla_types_seen": []}
    r = _open_stream(filename)
    try:
        vla_hits = set()
        tf = tarfile.open(fileobj=r.raw, mode="r|gz")
        count = 0
        for member in tf:
            if not member.isfile():
                continue
            ext = os.path.splitext(member.name)[1].lower()
            entry = {"name": member.name, "size": member.size, "ext": ext}
            if ext in TEXT_EXTENSIONS and 0 < member.size < 5_000_000:
                fobj = tf.extractfile(member)
                if fobj is not None:
                    text = fobj.read().decode("utf-8", errors="replace")
                    entry["preview"] = text[:500]
                    vla_hits.update(_flag_tokens(text[:20000]))
            result["members"].append(entry)
            count += 1
            if count >= MAX_TAR_MEMBERS:
                break
        tf.close()
        if vla_hits:
            result["flag"] = "has-vla-tokens"
        elif result["members"] and all(m["ext"] in BINARY_EXTENSIONS for m in result["members"]):
            result["flag"] = "unknown/binary"
        else:
            result["flag"] = "text-only"
        result["vla_types_seen"] = sorted(vla_hits)
    finally:
        r.close()
    return result


def peek_file(filename: str) -> dict:
    print(f"\n{'=' * 72}\n{filename}\n{'=' * 72}", flush=True)
    t0 = time.time()
    try:
        if filename.endswith(".jsonl.gz"):
            res = peek_jsonl_gz(filename)
        elif filename.endswith(".tar.gz"):
            res = peek_tar_gz(filename)
        else:
            res = {"format": "unknown", "flag": "unknown"}
    except Exception as e:
        res = {"format": "error", "flag": "error", "error": str(e)}
    res["elapsed_s"] = round(time.time() - t0, 1)
    print(json.dumps(res, indent=2, ensure_ascii=False)[:4000], flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="peek a single file by name")
    args = ap.parse_args()

    targets = [args.only] if args.only else FILES
    report = {}
    for fn in targets:
        report[fn] = peek_file(fn)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nSaved report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
