#!/usr/bin/env python3
"""Peek at the first few records of valid_with_seed to diagnose token format."""
import gzip, json, os, tarfile
from collections import Counter

TMP_DIR = "/p/data1/mmlaion/nguyen38/.tmp"
# The shard should still be cached
import glob
cached = glob.glob(f"{TMP_DIR}/**/valid_with_seed_shard_00000.tar.gz", recursive=True)
if not cached:
    print("Shard not cached, run data_inventory.py first to download it")
    raise SystemExit(1)
shard_path = cached[0]
print(f"Using cached shard: {shard_path}")

key_counter = Counter()
sample_records = []

def peek_tar(fobj, mode, depth=0):
    indent = "  " * depth
    try:
        with tarfile.open(fileobj=fobj, mode=mode) as t:
            for member in t:
                if not member.isfile() or member.size == 0:
                    continue
                name = member.name
                mfobj = t.extractfile(member)
                if mfobj is None:
                    continue
                if name.endswith((".tar.gz", ".tgz")):
                    print(f"{indent}→ inner tar: {name} ({member.size/1e6:.0f} MB)")
                    peek_tar(mfobj, "r|gz", depth+1)
                    return  # one inner shard is enough
                elif name.endswith(".tar"):
                    peek_tar(mfobj, "r|", depth+1)
                    return
                else:
                    print(f"{indent}→ data file: {name} ({member.size/1e6:.0f} MB)")
                    raw = mfobj.read()
                    if name.endswith(".gz"):
                        raw = gzip.decompress(raw)
                    lines = raw.decode("utf-8", errors="replace").splitlines()
                    print(f"{indent}  {len(lines):,} lines")
                    for line in lines[:5]:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                            if isinstance(d, dict):
                                key_counter.update(d.keys())
                                sample_records.append(d)
                                print(f"{indent}  keys: {list(d.keys())}")
                                # Print first 200 chars of each value
                                for k, v in d.items():
                                    snippet = str(v)[:200]
                                    print(f"{indent}    [{k!r}]: {snippet!r}")
                            else:
                                print(f"{indent}  non-dict JSON: {type(d).__name__} = {str(d)[:100]!r}")
                        except json.JSONDecodeError:
                            print(f"{indent}  non-JSON line: {line[:100]!r}")
                    if len(sample_records) >= 3:
                        return
    except Exception as e:
        print(f"{indent}  error: {e}")

with tarfile.open(shard_path, "r:gz") as outer:
    for member in outer:
        if not member.isfile():
            continue
        fobj = outer.extractfile(member)
        if fobj:
            print(f"outer member: {member.name} ({member.size/1e6:.0f} MB)")
            peek_tar(fobj, "r|gz", depth=1)
        if len(sample_records) >= 3:
            break

print(f"\nKey counts across sampled records: {dict(key_counter.most_common(20))}")
