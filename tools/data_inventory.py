#!/usr/bin/env python3
"""
Data inventory: count tokens across all available multimodal datasets
and generate pie charts for Huu's data overview request.

Usage (on JUPITER login node, aarch64):
    module --force purge
    module load Stages/2025 GCC/13.3.0 Python/3.12.3 SciPy-bundle/2024.05 matplotlib/3.9.2
    python tools/data_inventory.py

Or on JUSUF (x86_64), use env_tools:
    module --force purge
    module load Stages/2025 GCC/13.3.0 Python/3.12.3 CUDA/12
    source /p/data1/mmlaion/nguyen38/env_tools/bin/activate
    pip install matplotlib  # if not installed
    python tools/data_inventory.py

Skipping steps:
    python tools/data_inventory.py --skip-finevideo   # use hardcoded counts
    python tools/data_inventory.py --skip-download     # skip HF downloads, use fallback estimates

Output:
    - Prints summary table to stdout
    - Saves two pie charts to tools/data_inventory_charts.png
"""

import argparse
import json
import glob
import gzip
import os
import re
import sys
import tarfile
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = "/p/data1/mmlaion/nguyen38/.tmp"
FINEVIDEO_DIR = "/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/megatron_dataset_adaptive"

SEED2_PAT = re.compile(r"<seed2_\d+>")
SEED_PAT = re.compile(r"<seed_\d+>")
COSMOS_PAT = re.compile(r"<cosmos_\d+>")
AVCLM_PAT = re.compile(r"<avclm_\d+>")
SNAC_PAT = re.compile(r"<snac_\d+>")
AGENT_PAT = re.compile(
    r"<(?:fps_\d+|(?:pelvis|r_hip|r_knee|r_ankle|l_hip|l_knee|l_ankle|"
    r"spine|thorax|nose|head_top|l_shoulder|l_elbow|l_wrist|"
    r"r_shoulder|r_elbow|r_wrist)(?:_[txyz]_\d+)?)>"
    r"|</(?:pelvis|r_hip|r_knee|r_ankle|l_hip|l_knee|l_ankle|"
    r"spine|thorax|nose|head_top|l_shoulder|l_elbow|l_wrist|"
    r"r_shoulder|r_elbow|r_wrist)>"
    r"|</?agent>"
)
ALL_VLA_PAT = re.compile(
    r"</?(?:seed2?_\d+|cosmos_\d+|avclm_\d+|snac_\d+|fps_\d+|agent|"
    r"seed2|cosmos|avc_lm|see|listen|speak|"
    r"(?:pelvis|r_hip|r_knee|r_ankle|l_hip|l_knee|l_ankle|"
    r"spine|thorax|nose|head_top|l_shoulder|l_elbow|l_wrist|"
    r"r_shoulder|r_elbow|r_wrist)(?:_[txyz]_\d+)?)>"
)


def count_finevideo():
    """Count tokens in FineVideo-Phase7-Flattened (local files)."""
    print("=== Counting FineVideo-Phase7-Flattened ===")
    files = sorted(glob.glob(os.path.join(FINEVIDEO_DIR, "flat_*.jsonl")))
    if not files:
        print(f"  No files found in {FINEVIDEO_DIR}")
        print("  Using hardcoded values from previous count")
        return {
            "records": 69844,
            "size_gb": 19.20,
            "seed2": 89_880_864,
            "cosmos": 210_154_800,
            "avclm": 474_362_547,
            "agent": 637_924_374,
            "snac": 0,
            "text_tokens": 362_522_978,
        }

    total_records = 0
    total_bytes = 0
    counts = {"seed2": 0, "cosmos": 0, "avclm": 0, "agent": 0, "snac": 0, "text_chars": 0}

    t0 = time.time()
    for fi, fpath in enumerate(files):
        total_bytes += os.path.getsize(fpath)
        with open(fpath) as f:
            for line in f:
                total_records += 1
                text = json.loads(line).get("text", "")
                counts["seed2"] += len(SEED2_PAT.findall(text))
                counts["cosmos"] += len(COSMOS_PAT.findall(text))
                counts["avclm"] += len(AVCLM_PAT.findall(text))
                counts["agent"] += len(AGENT_PAT.findall(text))
                clean = ALL_VLA_PAT.sub("", text)
                counts["text_chars"] += len(clean.strip())
        if (fi + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (fi + 1) / elapsed
            eta = (len(files) - fi - 1) / rate
            print(f"  [{fi+1}/{len(files)}] {total_records:,} records | "
                  f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining", flush=True)

    print(f"  Done: {total_records:,} records, {total_bytes/1e9:.2f} GB in {time.time()-t0:.0f}s")
    return {
        "records": total_records,
        "size_gb": total_bytes / 1e9,
        "seed2": counts["seed2"],
        "cosmos": counts["cosmos"],
        "avclm": counts["avclm"],
        "agent": counts["agent"],
        "snac": 0,
        "text_tokens": int(counts["text_chars"] / 4),
    }


def count_mv_omni():
    """Count tokens in MixtureVitae-Omni by sampling."""
    print("\n=== Sampling MixtureVitae-Omni ===")
    from huggingface_hub import HfApi

    api = HfApi()
    siblings = list(api.list_repo_tree(
        "mixture-vitae/MixtureVitae-Omni", path_in_repo="data/data", repo_type="dataset"
    ))
    total_compressed = sum(s.size for s in siblings if hasattr(s, "size"))
    print(f"  Total compressed size: {total_compressed/1e9:.2f} GB, {len(siblings)} files")

    import urllib.request
    import io

    url = "https://huggingface.co/datasets/mixture-vitae/MixtureVitae-Omni/resolve/main/data/data/snac_emo_seed_0.jsonl.gz"
    sample_file = os.path.join(TMP_DIR, "mv_omni_sample.jsonl.gz")

    print("  Downloading 200MB sample...")
    req = urllib.request.Request(url, headers={"Range": "bytes=0-200000000"})
    resp = urllib.request.urlopen(req, timeout=120)
    with open(sample_file, "wb") as f:
        f.write(resp.read())

    sample_compressed_bytes = os.path.getsize(sample_file)
    counts = {"seed": 0, "snac": 0, "text_chars": 0}
    n_records = 0

    with gzip.open(sample_file, "rt") as f:
        for line in f:
            d = json.loads(line)
            text = d.get("text", "")
            counts["seed"] += len(SEED_PAT.findall(text))
            counts["snac"] += len(SNAC_PAT.findall(text))
            clean = ALL_VLA_PAT.sub("", text)
            counts["text_chars"] += len(clean.strip())
            n_records += 1

    os.unlink(sample_file)

    scale = total_compressed / sample_compressed_bytes
    print(f"  Sampled {n_records:,} records, scale factor: {scale:.1f}x")

    return {
        "records": int(n_records * scale),
        "size_gb": total_compressed / 1e9,
        "seed2": int(counts["seed"] * scale),
        "cosmos": 0,
        "avclm": 0,
        "agent": 0,
        "snac": int(counts["snac"] * scale),
        "text_tokens": int(counts["text_chars"] / 4 * scale),
        "note": "sampled + extrapolated",
    }


def count_valid_with_seed():
    """Count tokens in valid_with_seed by downloading and sampling 1 shard."""
    print("\n=== Sampling valid_with_seed (1 shard of 64) ===")
    from huggingface_hub import hf_hub_download, HfApi

    api = HfApi()
    siblings = list(api.list_repo_tree(
        "mixture-vitae-backup/MixtureVitae-Backup",
        path_in_repo="data/valid_with_seed",
        repo_type="dataset",
    ))
    total_compressed = sum(s.size for s in siblings if hasattr(s, "size"))
    n_shards = len([s for s in siblings if hasattr(s, "size") and s.size > 0])
    print(f"  {n_shards} shards, total compressed: {total_compressed/1e9:.2f} GB")

    shard_name = "data/valid_with_seed/valid_with_seed_shard_00000.tar.gz"
    shard_size = next(s.size for s in siblings if s.path == shard_name)
    print(f"  Downloading shard 0 ({shard_size/1e9:.2f} GB)... this may take a while")

    shard_path = hf_hub_download(
        "mixture-vitae-backup/MixtureVitae-Backup",
        filename=shard_name,
        repo_type="dataset",
        cache_dir=TMP_DIR,
    )

    counts = {"seed": 0, "seed2": 0, "snac": 0, "cosmos": 0, "text_chars": 0}
    n_records = 0

    print("  Counting tokens in shard...")
    t0 = time.time()
    with tarfile.open(shard_path, "r:gz") as tar:
        n_members = 0
        for member in tar:
            if not member.name.endswith(".jsonl") and not member.name.endswith(".json"):
                continue
            n_members += 1
            f = tar.extractfile(member)
            if f is None:
                continue
            for line in f:
                try:
                    text = json.loads(line).get("text", "")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                counts["seed"] += len(SEED_PAT.findall(text))
                counts["seed2"] += len(SEED2_PAT.findall(text))
                counts["snac"] += len(SNAC_PAT.findall(text))
                counts["cosmos"] += len(COSMOS_PAT.findall(text))
                clean = ALL_VLA_PAT.sub("", text)
                counts["text_chars"] += len(clean.strip())
                n_records += 1
                if n_records % 5000 == 0:
                    elapsed = time.time() - t0
                    print(f"    {n_records:,} records from {n_members} files | {elapsed:.0f}s", flush=True)

    scale = total_compressed / shard_size
    print(f"  Sampled {n_records:,} records from 1 shard, scale factor: {scale:.1f}x")
    print(f"  Token types found: seed={counts['seed']:,}, seed2={counts['seed2']:,}, "
          f"snac={counts['snac']:,}, cosmos={counts['cosmos']:,}")

    # Clean up downloaded shard
    try:
        os.unlink(shard_path)
        cache_parent = os.path.dirname(shard_path)
        if TMP_DIR in cache_parent:
            import shutil
            shutil.rmtree(os.path.join(TMP_DIR, "datasets--mixture-vitae-backup--MixtureVitae-Backup"),
                         ignore_errors=True)
    except Exception:
        pass

    total_seed = counts["seed"] + counts["seed2"]
    return {
        "records": int(n_records * scale),
        "size_gb": total_compressed / 1e9,
        "seed2": int(total_seed * scale),
        "cosmos": int(counts["cosmos"] * scale),
        "avclm": 0,
        "agent": 0,
        "snac": int(counts["snac"] * scale),
        "text_tokens": int(counts["text_chars"] / 4 * scale),
        "note": f"sampled 1/{n_shards} shards + extrapolated",
    }


def make_charts(datasets, output_path):
    """Generate two pie charts: by modality and by dataset."""
    modalities = ["seed2", "cosmos", "avclm", "agent", "snac", "text_tokens"]
    modality_labels = ["Seed2 (image)", "Cosmos (video)", "AVC-LM (audio)",
                       "Agent (3D pose)", "SNAC (audio)", "Text"]
    modality_colors = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63", "#9C27B0", "#607D8B"]

    # Aggregate by modality
    modality_totals = defaultdict(int)
    for ds in datasets.values():
        for m in modalities:
            modality_totals[m] += ds.get(m, 0)

    # Filter out zero modalities
    nonzero = [(m, l, c) for m, l, c in zip(modalities, modality_labels, modality_colors)
               if modality_totals[m] > 0]
    mod_vals = [modality_totals[m] for m, _, _ in nonzero]
    mod_labels = [f"{l}\n{modality_totals[m]/1e9:.2f}B" for m, l, _ in nonzero]
    mod_colors = [c for _, _, c in nonzero]

    # Dataset totals
    ds_names = list(datasets.keys())
    ds_totals = []
    for name in ds_names:
        ds = datasets[name]
        total = sum(ds.get(m, 0) for m in modalities)
        ds_totals.append(total)

    ds_colors = ["#FF5722", "#3F51B5", "#009688", "#FFC107", "#795548"]
    ds_labels = [f"{name}\n{total/1e9:.2f}B" for name, total in zip(ds_names, ds_totals)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Chart 1: By modality
    wedges1, texts1, autotexts1 = ax1.pie(
        mod_vals, labels=mod_labels, colors=mod_colors,
        autopct="%1.1f%%", startangle=90, pctdistance=0.75,
        textprops={"fontsize": 10}
    )
    for t in autotexts1:
        t.set_fontsize(9)
    total_all = sum(mod_vals)
    ax1.set_title(f"Token Distribution by Modality\nTotal: {total_all/1e9:.1f}B tokens",
                  fontsize=14, fontweight="bold")

    # Chart 2: By dataset
    wedges2, texts2, autotexts2 = ax2.pie(
        ds_totals, labels=ds_labels, colors=ds_colors[:len(ds_names)],
        autopct="%1.1f%%", startangle=90, pctdistance=0.75,
        textprops={"fontsize": 10}
    )
    for t in autotexts2:
        t.set_fontsize(9)
    ax2.set_title(f"Token Distribution by Dataset\nTotal: {sum(ds_totals)/1e9:.1f}B tokens",
                  fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nCharts saved to: {output_path}")


def print_summary(datasets):
    """Print a summary table."""
    modalities = ["seed2", "cosmos", "avclm", "agent", "snac", "text_tokens"]

    print("\n" + "=" * 100)
    print("DATA INVENTORY SUMMARY")
    print("=" * 100)

    header = f"{'Dataset':<30} {'Records':>12} {'Size':>8} {'Seed2':>10} {'Cosmos':>10} {'AVC-LM':>10} {'Agent':>10} {'SNAC':>10} {'Text':>10} {'Total':>12}"
    print(header)
    print("-" * len(header))

    grand_total = defaultdict(int)
    for name, ds in datasets.items():
        total = sum(ds.get(m, 0) for m in modalities)
        fmt = lambda v: f"{v/1e6:.1f}M" if v > 0 else "—"
        print(f"{name:<30} {ds['records']:>12,} {ds['size_gb']:>7.1f}G "
              f"{fmt(ds.get('seed2',0)):>10} {fmt(ds.get('cosmos',0)):>10} "
              f"{fmt(ds.get('avclm',0)):>10} {fmt(ds.get('agent',0)):>10} "
              f"{fmt(ds.get('snac',0)):>10} {fmt(ds.get('text_tokens',0)):>10} "
              f"{total/1e9:>11.2f}B")
        for m in modalities:
            grand_total[m] += ds.get(m, 0)

    print("-" * len(header))
    gt = sum(grand_total.values())
    fmt = lambda v: f"{v/1e6:.1f}M" if v < 1e9 else f"{v/1e9:.2f}B"
    print(f"{'TOTAL':<30} {'':>12} {'':>8} "
          f"{fmt(grand_total['seed2']):>10} {fmt(grand_total['cosmos']):>10} "
          f"{fmt(grand_total['avclm']):>10} {fmt(grand_total['agent']):>10} "
          f"{fmt(grand_total['snac']):>10} {fmt(grand_total['text_tokens']):>10} "
          f"{gt/1e9:>11.2f}B")

    print("\n--- Not yet tokenized (excluded from charts) ---")
    print("  MV-Backup stack_images3_gzip:  200 GB compressed, raw images, needs seed2/cosmos tokenization")
    print("  SenseNova-SI-8M:               8M image-text pairs, Apache 2.0, needs tokenization")
    print("  stera-10m:                     10M egocentric video clips, restrictive license")
    print("  OmniAction:                    action-labeled video, CC-BY-NC-4.0")


def main():
    parser = argparse.ArgumentParser(description="Data inventory: count tokens and generate pie charts")
    parser.add_argument("--skip-finevideo", action="store_true",
                        help="Use hardcoded FineVideo counts instead of re-scanning")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip HF downloads, use fallback estimates for MV-Omni and valid_with_seed")
    parser.add_argument("--output", default=os.path.join(SCRIPT_DIR, "data_inventory_charts.png"),
                        help="Output path for pie chart PNG")
    args = parser.parse_args()

    os.makedirs(TMP_DIR, exist_ok=True)
    datasets = {}
    total_start = time.time()

    # 1. FineVideo
    if args.skip_finevideo:
        print("=== FineVideo-Phase7-Flattened (using hardcoded counts) ===")
        datasets["FineVideo-VLA"] = {
            "records": 69844, "size_gb": 19.20,
            "seed2": 89_880_864, "cosmos": 210_154_800,
            "avclm": 474_362_547, "agent": 637_924_374,
            "snac": 0, "text_tokens": 362_522_978,
        }
    else:
        datasets["FineVideo-VLA"] = count_finevideo()

    # 2. MixtureVitae-Omni
    if args.skip_download:
        print("\n=== MV-Omni (using fallback estimate) ===")
        datasets["MV-Omni"] = {
            "records": 180_850, "size_gb": 36.2,
            "seed2": 5_787_200, "cosmos": 0, "avclm": 0, "agent": 0,
            "snac": 106_837_860, "text_tokens": 55_120_457,
            "note": "fallback estimate",
        }
    else:
        try:
            datasets["MV-Omni"] = count_mv_omni()
        except Exception as e:
            print(f"  Error sampling MV-Omni: {e}")
            datasets["MV-Omni"] = {
                "records": 180_850, "size_gb": 36.2,
                "seed2": 5_787_200, "cosmos": 0, "avclm": 0, "agent": 0,
                "snac": 106_837_860, "text_tokens": 55_120_457,
                "note": "fallback estimate",
            }

    # 3. valid_with_seed
    if args.skip_download:
        print("\n=== valid_with_seed (using fallback estimate) ===")
        datasets["MV-Backup valid_with_seed"] = {
            "records": 0, "size_gb": 1233.0,
            "seed2": 30_000_000_000, "cosmos": 0, "avclm": 0, "agent": 0,
            "snac": 0, "text_tokens": 137_700_000_000,
            "note": "chat estimate, unverified",
        }
    else:
        try:
            datasets["MV-Backup valid_with_seed"] = count_valid_with_seed()
        except Exception as e:
            print(f"  Error sampling valid_with_seed: {e}")
            print("  Using chat estimate (~167.7B tokens)")
            datasets["MV-Backup valid_with_seed"] = {
                "records": 0, "size_gb": 1233.0,
                "seed2": 30_000_000_000, "cosmos": 0, "avclm": 0, "agent": 0,
                "snac": 0, "text_tokens": 137_700_000_000,
                "note": "chat estimate, unverified",
            }

    # Print summary
    print_summary(datasets)

    # Generate charts
    make_charts(datasets, args.output)

    print(f"\nTotal runtime: {time.time() - total_start:.0f}s")


if __name__ == "__main__":
    main()
