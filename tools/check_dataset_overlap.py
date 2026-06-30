"""
Check overlap between 3 datasets by YouTube video ID:
  - valid_with_seed  (hf_shards/): 64 outer tar.gz, each has loose files + inner tar
  - omni_valid       (hf_snac/):   6 jsonl.gz, metadata.params.id = YouTube ID
  - ontocord/VALID   (head.txt):   head sample only (full dataset not downloaded)

Run from: /p/data1/mmlaion/nguyen38/3d-human-pose/
"""

import tarfile, gzip, json, os, re, sys
from pathlib import Path

CACHE = Path("/p/data1/mmlaion/nguyen38/inventory_cache")
SHARDS_DIR = CACHE / "hf_shards"
SNAC_DIR   = CACHE / "hf_snac"
HEAD_FILE  = Path("/p/data1/mmlaion/nguyen38/3d-human-pose/multimodal/head.txt")


# ── helpers ─────────────────────────────────────────────────────────────────

def extract_yt_id_from_filename(name: str):
    """Extract the 11-char YouTube video ID from a filename like
    '1D-i20_vpZs_74_seed2.jsonl' → '1D-i20_vpZs'
    Works because YouTube IDs are always 11 chars of [A-Za-z0-9_-].
    """
    # strip known suffixes and extension
    stem = Path(name).stem            # e.g. '1D-i20_vpZs_74_seed2'
    stem = re.sub(r'_seed2$', '', stem)   # '1D-i20_vpZs_74'
    # take first 11 chars; they must match YouTube ID charset
    candidate = stem[:11]
    if re.fullmatch(r'[A-Za-z0-9_\-]{11}', candidate):
        return candidate
    return None


# ── 1. valid_with_seed ──────────────────────────────────────────────────────

def collect_valid_with_seed_ids():
    ids: set[str] = set()
    shards = sorted(SHARDS_DIR.glob("valid_with_seed_shard_*.tar.gz"))
    print(f"[valid_with_seed] scanning {len(shards)} outer shards ...")
    for i, shard_path in enumerate(shards):
        try:
            with tarfile.open(shard_path, "r:gz") as outer_tf:
                for member in outer_tf.getmembers():
                    name = Path(member.name).name
                    if name.endswith(".tar.gz") or name.endswith(".tar"):
                        # open inner tar without extracting to disk
                        try:
                            inner_fobj = outer_tf.extractfile(member)
                            if inner_fobj is None:
                                continue
                            with tarfile.open(fileobj=inner_fobj, mode="r:gz") as inner_tf:
                                for inner_member in inner_tf.getmembers():
                                    iname = Path(inner_member.name).name
                                    yt_id = extract_yt_id_from_filename(iname)
                                    if yt_id:
                                        ids.add(yt_id)
                        except Exception:
                            pass
                    else:
                        yt_id = extract_yt_id_from_filename(name)
                        if yt_id:
                            ids.add(yt_id)
        except Exception as e:
            print(f"  WARNING: could not open {shard_path.name}: {e}")
        if (i + 1) % 8 == 0:
            print(f"  ... {i+1}/{len(shards)} shards done, {len(ids)} unique IDs so far")
    print(f"[valid_with_seed] total unique video IDs: {len(ids)}")
    return ids


# ── 2. omni_valid (MixtureVitae-Omni valid_snac) ───────────────────────────

def collect_omni_valid_ids():
    ids: set[str] = set()
    files = sorted(SNAC_DIR.glob("valid_snac_*.jsonl.gz"))
    print(f"\n[omni_valid] scanning {len(files)} gzip files ...")
    for f in files:
        count = 0
        try:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        meta_list = json.loads(d.get("metadata", "[]"))
                        for meta in meta_list:
                            params_raw = meta.get("params", "")
                            if not params_raw:
                                continue
                            try:
                                params = json.loads(params_raw)
                                yt_id = params.get("id", "")
                                if yt_id and len(yt_id) == 11:
                                    ids.add(yt_id)
                            except json.JSONDecodeError:
                                pass
                    except json.JSONDecodeError:
                        pass
                    count += 1
        except Exception as e:
            print(f"  WARNING: {f.name}: {e}")
        print(f"  {f.name}: {count} lines processed")
    print(f"[omni_valid] total unique video IDs: {len(ids)}")
    return ids


# ── 3. ontocord/VALID (head.txt sample only) ───────────────────────────────

def collect_valid_head_ids():
    """Parse head.txt which is a 3-row-per-record JSONL (audio, image, emotion).
    Audio rows have file_name like '-mbDQC0y0PY_6.ogg'.
    Also try params.id from metadata if present.
    """
    ids: set[str] = set()
    if not HEAD_FILE.exists():
        print("\n[ontocord/VALID] head.txt not found, skipping")
        return ids
    count = 0
    with open(HEAD_FILE) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                # try file_name
                fname = d.get("file_name", "")
                if fname:
                    yt_id = extract_yt_id_from_filename(fname)
                    if yt_id:
                        ids.add(yt_id)
                # try nested metadata params
                meta_raw = d.get("metadata", {})
                if isinstance(meta_raw, str):
                    meta_raw = json.loads(meta_raw)
                if isinstance(meta_raw, dict):
                    params_raw = meta_raw.get("params", "")
                    if params_raw:
                        try:
                            p = json.loads(params_raw)
                            yt_id = p.get("id", "")
                            if yt_id and len(yt_id) == 11:
                                ids.add(yt_id)
                        except Exception:
                            pass
            except json.JSONDecodeError:
                pass
            count += 1
    print(f"\n[ontocord/VALID] head.txt: {count} lines → {len(ids)} unique video IDs (SAMPLE ONLY)")
    return ids


# ── main ────────────────────────────────────────────────────────────────────

def main():
    seed_ids  = collect_valid_with_seed_ids()
    omni_ids  = collect_omni_valid_ids()
    valid_ids = collect_valid_head_ids()

    # --- overlaps ---
    print("\n" + "="*60)
    print("OVERLAP ANALYSIS")
    print("="*60)

    overlap_seed_omni = seed_ids & omni_ids
    print(f"\nvalid_with_seed  : {len(seed_ids):,} unique video IDs")
    print(f"omni_valid       : {len(omni_ids):,} unique video IDs")
    print(f"overlap (both)   : {len(overlap_seed_omni):,} video IDs")
    if seed_ids:
        print(f"  = {100*len(overlap_seed_omni)/len(seed_ids):.1f}% of valid_with_seed is in omni_valid")
    if omni_ids:
        print(f"  = {100*len(overlap_seed_omni)/len(omni_ids):.1f}% of omni_valid comes from valid_with_seed")

    only_in_seed = seed_ids - omni_ids
    only_in_omni = omni_ids - seed_ids
    print(f"\nOnly in valid_with_seed (not in omni): {len(only_in_seed):,}")
    print(f"Only in omni_valid (not in valid_with_seed): {len(only_in_omni):,}")

    if valid_ids:
        print(f"\nontocord/VALID head sample: {len(valid_ids):,} video IDs")
        overlap_valid_seed = valid_ids & seed_ids
        overlap_valid_omni = valid_ids & omni_ids
        print(f"  head ∩ valid_with_seed: {len(overlap_valid_seed):,} ({100*len(overlap_valid_seed)/len(valid_ids):.0f}% of head)")
        print(f"  head ∩ omni_valid:      {len(overlap_valid_omni):,} ({100*len(overlap_valid_omni)/len(valid_ids):.0f}% of head)")

    # --- save results ---
    out = {
        "valid_with_seed_count": len(seed_ids),
        "omni_valid_count": len(omni_ids),
        "overlap_count": len(overlap_seed_omni),
        "only_in_seed": len(only_in_seed),
        "only_in_omni": len(only_in_omni),
        "overlap_pct_of_seed": round(100*len(overlap_seed_omni)/len(seed_ids), 2) if seed_ids else 0,
        "overlap_pct_of_omni": round(100*len(overlap_seed_omni)/len(omni_ids), 2) if omni_ids else 0,
        "overlap_video_ids": sorted(overlap_seed_omni),
        "only_in_seed_ids": sorted(only_in_seed)[:1000],  # cap to avoid huge file
        "only_in_omni_ids": sorted(only_in_omni)[:1000],
    }
    out_path = Path("/p/data1/mmlaion/nguyen38/3d-human-pose/tools/dataset_overlap_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
