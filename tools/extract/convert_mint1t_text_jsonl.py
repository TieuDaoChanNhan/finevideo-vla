#!/usr/bin/env python3
"""
Convert MINT-1T-HTML parquet shards (texts[] column, interleaved with image
slots -- see extract_mint1t_manifest.py's docstring for the verified schema)
into flat {"text": "..."} JSONL, one line per record -- the only format
mv_preprocess_data.py (Megatron tokenizer) accepts.

Joins non-null texts[i] spans (in document order) with a blank line, dropping
the interleaved image positions (this is the text-only path -- image URLs
were dropped project-wide on 18/07/2026, see datasets.md).

Worker-split pattern matches the rest of this repo (SLURM_ARRAY_TASK_ID/COUNT
contiguous slice), safe to re-run (skips shards whose output already exists).

Usage (single machine, pilot):
    python3 tools/extract/convert_mint1t_text_jsonl.py --num-shards 20

Usage (SLURM array, see slurm/convert_mint1t_text.sbatch):
    python3 tools/extract/convert_mint1t_text_jsonl.py
"""
import argparse
import glob
import json
import os

import pyarrow.parquet as pq

INPUT_DIR_DEFAULT = "/p/data1/mmlaion/shared/vla/mint1t_html/data_v1_1"
OUTPUT_DIR_DEFAULT = "/p/data1/mmlaion/shared/vla/mint1t_html/text_jsonl"


def convert_one(path, output_dir):
    stem = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(output_dir, f"{stem}.jsonl")
    if os.path.exists(out_path):
        return stem, 0, True

    pf = pq.ParquetFile(path)
    n = 0
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as out:
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["texts"])
            for rec in tbl.to_pylist():
                spans = [t for t in (rec["texts"] or []) if t]
                if not spans:
                    continue
                text = "\n\n".join(spans)
                out.write(json.dumps({"text": text}) + "\n")
                n += 1
    os.replace(tmp_path, out_path)
    return stem, n, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=INPUT_DIR_DEFAULT)
    ap.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    ap.add_argument("--num-shards", type=int, default=None, help="pilot: only first N shards (sorted)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_files = sorted(glob.glob(os.path.join(args.input_dir, "*.parquet")))
    if args.num_shards:
        all_files = all_files[:args.num_shards]

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    my_files = all_files[task_id - 1::num_tasks]

    print(f"[task {task_id}/{num_tasks}] {len(my_files)}/{len(all_files)} shard(s) assigned", flush=True)

    total_records = 0
    for i, path in enumerate(my_files):
        stem, n, skipped = convert_one(path, args.output_dir)
        total_records += n
        tag = "SKIP" if skipped else "DONE"
        print(f"[task {task_id}] [{i+1}/{len(my_files)}] {tag} {stem}: {n:,} records", flush=True)

    print(f"[task {task_id}] finished: {total_records:,} records written", flush=True)


if __name__ == "__main__":
    main()
