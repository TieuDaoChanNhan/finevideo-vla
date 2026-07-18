#!/usr/bin/env python3
"""
Extract MINT-1T-HTML parquet shards (texts/images columns) into per-shard JSONL
manifests, one line per record, preserving the interleaved texts[i]/images[i]
alignment verified against real data on 18/07/2026 (same length, mutually
exclusive: at each position either texts[i] or images[i] is non-null, never
both). image_hashes/images_metadata/metadata are NOT positionally aligned to
texts/images (shorter, image-only order) and are intentionally dropped here --
downstream code should not try to zip them against images[i] by index.

Output: {output_dir}/{shard_stem}.jsonl, one line per record:
  {"record_id": "<shard_stem>_<row_idx>", "source_url": ..., "cc_dump": ...,
   "texts": [...], "images": [...]}   # texts/images same length, interleaved

Usage:
    python3 tools/extract/extract_mint1t_manifest.py --num-shards 20
    python3 tools/extract/extract_mint1t_manifest.py --shard-list shards.txt
"""
import argparse
import glob
import json
import os

import pyarrow.parquet as pq

INPUT_DIR_DEFAULT = "/p/data1/mmlaion/shared/vla/mint1t_html/data_v1_1"
OUTPUT_DIR_DEFAULT = "/p/data1/mmlaion/shared/vla/mint1t_html/manifest"


def pick_shards(input_dir, num_shards, shard_list):
    files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if shard_list:
        wanted = set(open(shard_list).read().split())
        files = [f for f in files if os.path.basename(f) in wanted]
    elif num_shards:
        files = files[:num_shards]
    return files


def extract_one(path, output_dir, skip_existing):
    stem = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(output_dir, f"{stem}.jsonl")
    if skip_existing and os.path.exists(out_path):
        return stem, 0, 0, True

    pf = pq.ParquetFile(path)
    n_records = 0
    n_images = 0
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as out:
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["texts", "images", "url", "cc_dump"])
            for row_idx, rec in enumerate(tbl.to_pylist()):
                texts = rec["texts"] or []
                images = rec["images"] or []
                if len(texts) != len(images):
                    # defensive -- should not happen per the 18/07 verification,
                    # but skip rather than silently misalign if source data varies
                    continue
                n_records += 1
                n_images += sum(1 for u in images if u)
                out.write(json.dumps({
                    "record_id": f"{stem}_{row_idx}",
                    "source_url": rec["url"],
                    "cc_dump": rec["cc_dump"],
                    "texts": texts,
                    "images": images,
                }) + "\n")
    os.replace(tmp_path, out_path)
    return stem, n_records, n_images, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=INPUT_DIR_DEFAULT)
    ap.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    ap.add_argument("--num-shards", type=int, default=None, help="pilot: only first N shards (sorted)")
    ap.add_argument("--shard-list", default=None, help="file with one parquet filename per line")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = pick_shards(args.input_dir, args.num_shards, args.shard_list)
    print(f"Extracting {len(files)} shard(s) -> {args.output_dir}", flush=True)

    total_records = 0
    total_images = 0
    for i, path in enumerate(files):
        stem, n_rec, n_img, skipped = extract_one(path, args.output_dir, args.skip_existing)
        total_records += n_rec
        total_images += n_img
        tag = "SKIP" if skipped else "DONE"
        print(f"[{i+1}/{len(files)}] {tag} {stem}: records={n_rec} image_urls={n_img}", flush=True)

    print(f"\n=== Extract complete: {total_records:,} records, {total_images:,} image URLs across {len(files)} shard(s) ===")


if __name__ == "__main__":
    main()
