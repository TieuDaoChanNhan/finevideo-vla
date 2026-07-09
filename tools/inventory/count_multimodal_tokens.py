#!/usr/bin/env python3
"""
count_multimodal_tokens.py — Sample-based token inventory for
mixture-vitae-backup/MixtureVitae-Backup/data/multimodal (HF, 15 files, ~103GB).

True HTTP streaming — no full download to disk. Caps each file to
--sample-mb compressed MB (default 75), counts:
  - VLA-style tagged tokens (<seed2_N>, <cosmos_N>, <snac_N>, <avclm_N>, agent)
    via the same regex as data_inventory.py, applied to each record's `text` field
  - plain text word count (same convention as data_inventory.py)
  - raw integer token arrays: any JSON field named `*_token`/`*_tokens` holding a
    list of ints (confirmed present as `snac_token` in valid_data_snac.jsonl.gz /
    train_data_snac.jsonl.gz via tools/inventory/peek_multimodal.py) — these are real
    tokens, just not in the `<tag_N>` string format, so they're counted
    separately under `raw_<fieldname>` and still extrapolated/reported.

Extrapolates sample counts to the full file size when sampled (not exact).
Resumable via an atomic JSON checkpoint (tools/inventory/multimodal_inventory_checkpoint.json).

Usage:
    python count_multimodal_tokens.py                        # sample all 15, 75MB/file
    python count_multimodal_tokens.py --sample-mb 150
    python count_multimodal_tokens.py --only youtube.tar.gz --full
    python count_multimodal_tokens.py --only train_data_snac.jsonl.gz --sample-mb 10
"""
import argparse
import json
import os
import sys
import tarfile
import time
import zlib

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_inventory import (  # reuse existing regex, auth, checkpoint machinery
    PATTERNS, TOKEN_TYPES, count_tokens, add_counts, zero_counts, fmt, elapsed,
    _hf_token, hf_url, load_checkpoint, save_checkpoint,
)
from peek_multimodal import REPO, PREFIX, FILES, TEXT_EXTENSIONS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "multimodal_inventory_checkpoint.json")


class CappedReader:
    """File-like wrapper that stops (returns b"") once `cap` compressed bytes have
    been read. cap=None means unlimited (still bounds bare read() with no size)."""

    def __init__(self, raw, cap):
        self.raw = raw
        self.cap = cap
        self.consumed = 0

    def read(self, n=None):
        if self.cap is not None:
            if self.consumed >= self.cap:
                return b""
            remaining = self.cap - self.consumed
            if n is None or n < 0 or n > remaining:
                n = remaining
        elif n is None or n < 0:
            n = 1 << 20
        chunk = self.raw.read(n)
        self.consumed += len(chunk)
        return chunk


def _open_stream(filename: str):
    url = hf_url(REPO, f"{PREFIX}/{filename}")
    token = _hf_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(url, stream=True, headers=headers, timeout=120)
    r.raise_for_status()
    r.raw.decode_content = True
    total_size = int(r.headers.get("content-length", 0))
    return r, total_size


_JSON_DECODER = json.JSONDecoder()
_SKIP_CHARS = " \t\r\n,[]"


def _extract_records(buf: str):
    """Pulls as many complete top-level JSON values as possible from the start
    of `buf`. Handles both true JSONL (one compact object per line) and a
    pretty-printed JSON array spanning many lines per object (confirmed the
    actual format of valid_data_snac/train_data_snac/emo .jsonl.gz — plain
    line-splitting fails on those since one record spans multiple lines).
    Returns (records, remainder) where remainder is carried over to the next
    chunk (either an incomplete trailing value, or trailing whitespace)."""
    records = []
    i, n = 0, len(buf)
    while True:
        while i < n and buf[i] in _SKIP_CHARS:
            i += 1
        if i >= n:
            break
        try:
            obj, end = _JSON_DECODER.raw_decode(buf, i)
        except json.JSONDecodeError:
            break  # incomplete trailing value — wait for more data
        records.append(obj)
        i = end
    return records, buf[i:]


def _record_tokens(counts: dict, rec) -> None:
    """Count tokens for one parsed JSON record. Handles both tagged-string tokens
    (via count_tokens/regex on `text`) and raw integer token arrays (any
    `*_token`/`*_tokens` field holding a list of ints — e.g. snac_token)."""
    if isinstance(rec, dict):
        text = rec.get("text", "")
        if text:
            add_counts(counts, count_tokens(text))
        for key, val in rec.items():
            if (key.endswith("_token") or key.endswith("_tokens")) and isinstance(val, list) and val \
                    and isinstance(val[0], (int, float)):
                ck = f"raw_{key}"
                counts[ck] = counts.get(ck, 0) + len(val)
        counts["_records"] = counts.get("_records", 0) + 1
    elif isinstance(rec, str) and rec.strip():
        add_counts(counts, count_tokens(rec))
        counts["_records"] = counts.get("_records", 0) + 1


def _scan_jsonl_lines(raw_reader, cap_bytes, counts: dict, label: str, t0: float, report_every=20000) -> int:
    """Streams gzip-compressed JSONL bounded by cap_bytes of *compressed* input,
    decompressing incrementally so a single oversized line can't force an
    unbounded read (bounded memory, bounded network)."""
    d = zlib.decompressobj(zlib.MAX_WBITS | 16)
    leftover = ""
    consumed = 0
    chunk_size = 1 << 16
    while True:
        if cap_bytes is not None and consumed >= cap_bytes:
            break
        want = chunk_size if cap_bytes is None else min(chunk_size, cap_bytes - consumed)
        chunk = raw_reader.read(want)
        if not chunk:
            break
        consumed += len(chunk)
        try:
            decompressed = d.decompress(chunk)
        except zlib.error as e:
            print(f"    {label}decompress error after {consumed} bytes: {e}", flush=True)
            break
        if not decompressed:
            continue
        buf = leftover + decompressed.decode("utf-8", errors="replace")
        records, leftover = _extract_records(buf)
        for rec in records:
            _record_tokens(counts, rec)
        if counts.get("_records", 0) and counts["_records"] % report_every == 0:
            print(f"    {label}records: {counts['_records']:,}  "
                  f"text: {fmt(counts.get('text', 0))}  "
                  f"raw_snac_codes: {fmt(counts.get('raw_snac_token', 0))}  "
                  f"consumed: {consumed / 1e6:.1f}MB  elapsed: {elapsed(t0)}", flush=True)
    return consumed


def scan_jsonl_gz(filename: str, sample_bytes, label: str) -> dict:
    counts = zero_counts()
    counts["_records"] = 0
    t0 = time.time()
    r, total_size = _open_stream(filename)
    try:
        consumed = _scan_jsonl_lines(r.raw, sample_bytes, counts, label, t0)
    finally:
        r.close()
    counts["_bytes_sampled"] = consumed
    counts["_bytes_total"] = total_size
    return counts


def scan_tar_gz(filename: str, sample_bytes, label: str) -> dict:
    counts = zero_counts()
    counts["_records"] = 0
    counts["_members_seen"] = 0
    t0 = time.time()
    r, total_size = _open_stream(filename)
    capped = CappedReader(r.raw, sample_bytes)
    try:
        try:
            tf = tarfile.open(fileobj=capped, mode="r|gz")
        except Exception as e:
            print(f"    {label}could not open as tar.gz: {e}", flush=True)
            counts["_bytes_sampled"] = capped.consumed
            counts["_bytes_total"] = total_size
            return counts
        try:
            for member in tf:
                if not member.isfile():
                    continue
                counts["_members_seen"] += 1
                ext = os.path.splitext(member.name)[1].lower()
                if ext in TEXT_EXTENSIONS and member.size > 0:
                    fobj = tf.extractfile(member)
                    if fobj is None:
                        continue
                    text = fobj.read().decode("utf-8", errors="replace")
                    records, _ = _extract_records(text)
                    for rec in records:
                        _record_tokens(counts, rec)
                # binary/.wds/.uploaded members: skipped; tarfile still consumes
                # their bytes from the stream to advance (unavoidable for a
                # non-seekable HTTP stream), which counts against the sample budget.
                if counts["_records"] and counts["_records"] % 20000 == 0:
                    print(f"    {label}members: {counts['_members_seen']:,}  "
                          f"records: {counts['_records']:,}  "
                          f"consumed: {capped.consumed / 1e6:.1f}MB  elapsed: {elapsed(t0)}", flush=True)
        except (tarfile.ReadError, tarfile.StreamError, EOFError, OSError) as e:
            print(f"    {label}stopped at sample boundary ({capped.consumed / 1e6:.1f}MB): {e}", flush=True)
        finally:
            try:
                tf.close()
            except Exception:
                pass
    finally:
        r.close()
    counts["_bytes_sampled"] = capped.consumed
    counts["_bytes_total"] = total_size
    return counts


def extrapolate(counts: dict) -> dict:
    """Scales every numeric, non-underscore-prefixed counter by
    total_size/bytes_sampled — covers TOKEN_TYPES plus any dynamic raw_* keys."""
    sampled = counts.get("_bytes_sampled", 0)
    total = counts.get("_bytes_total", 0)
    result = dict(counts)
    if sampled and total and sampled < total:
        factor = total / sampled
        result["_extrapolation_factor"] = round(factor, 3)
        result["_exact"] = False
        for k, v in counts.items():
            if not k.startswith("_") and isinstance(v, (int, float)):
                result[k + "_extrapolated"] = int(v * factor)
    else:
        result["_exact"] = True
    return result


def count_file(filename: str, sample_mb, checkpoint: dict, checkpoint_path: str) -> dict:
    if filename in checkpoint["completed"]:
        print(f"[skip, already in checkpoint] {filename}")
        return checkpoint["completed"][filename]

    sample_bytes = None if sample_mb is None else int(sample_mb * 1_000_000)
    print(f"\n{'=' * 72}\n{filename}  (sample_mb={sample_mb if sample_mb else 'FULL'})\n{'=' * 72}", flush=True)
    label = f"{filename}: "
    t0 = time.time()
    try:
        if filename.endswith(".jsonl.gz"):
            counts = scan_jsonl_gz(filename, sample_bytes, label)
        elif filename.endswith(".tar.gz"):
            counts = scan_tar_gz(filename, sample_bytes, label)
        else:
            counts = {"_error": "unrecognized extension"}
    except Exception as e:
        counts = {"_error": str(e)}

    counts["_elapsed_s"] = round(time.time() - t0, 1)
    result = extrapolate(counts) if "_error" not in counts else counts
    print(json.dumps(result, indent=2)[:2500], flush=True)

    checkpoint["completed"][filename] = result
    save_checkpoint(checkpoint_path, checkpoint)
    return result


def print_summary(checkpoint: dict):
    print(f"\n{'=' * 96}\nSUMMARY (extrapolated to full file size where sampled; 'exact' = fully consumed within sample budget)\n{'=' * 96}")
    header = f"{'file':38s} {'exact?':7s} {'text':>10s} {'raw_snac':>10s} {'records':>10s}  note"
    print(header)
    print("-" * len(header))
    for fn in FILES:
        c = checkpoint["completed"].get(fn)
        if not c:
            print(f"{fn:38s} not run yet")
            continue
        if "_error" in c:
            print(f"{fn:38s} ERROR: {c['_error']}")
            continue
        exact = "yes" if c.get("_exact") else "no"
        text_v = c.get("text_extrapolated", c.get("text", 0))
        snac_v = c.get("raw_snac_token_extrapolated", c.get("raw_snac_token", 0))
        rec_v = c.get("_records_extrapolated", c.get("_records", 0))
        vla_hits = [k for k in ("seed2", "seed", "cosmos", "avclm", "snac", "agent")
                    if c.get(k, 0) > 0]
        note = f"tagged VLA tokens: {','.join(vla_hits)}" if vla_hits else ""
        print(f"{fn:38s} {exact:7s} {fmt(text_v):>10s} {fmt(snac_v):>10s} {fmt(rec_v):>10s}  {note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-mb", type=float, default=75, help="compressed MB to sample per file (default 75)")
    ap.add_argument("--full", action="store_true", help="no cap -- scan the entire file")
    ap.add_argument("--only", help="only run this filename")
    ap.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    args = ap.parse_args()

    sample_mb = None if args.full else args.sample_mb
    checkpoint = load_checkpoint(args.checkpoint)
    checkpoint.setdefault("completed", {})

    targets = [args.only] if args.only else FILES
    for fn in targets:
        count_file(fn, sample_mb, checkpoint, args.checkpoint)

    print_summary(checkpoint)


if __name__ == "__main__":
    main()
