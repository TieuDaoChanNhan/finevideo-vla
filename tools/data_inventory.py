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
    """Count tokens in MixtureVitae-Omni by streaming (no partial-gzip download)."""
    print("\n=== Sampling MixtureVitae-Omni ===")
    import urllib.request
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN", "")
    api = HfApi(token=token or None)
    siblings = list(api.list_repo_tree(
        "mixture-vitae/MixtureVitae-Omni", path_in_repo="data/data", repo_type="dataset"
    ))
    data_files = [s for s in siblings if hasattr(s, "size") and s.size > 0]
    total_compressed = sum(s.size for s in data_files)
    print(f"  Total compressed size: {total_compressed/1e9:.2f} GB, {len(data_files)} files")

    # Pick the first file and stream-decompress it — avoids downloading a partial gzip
    # which is undecompressable (gzip requires the end-of-stream marker).
    first_file = data_files[0]
    sample_file_size = first_file.size
    url = (
        f"https://huggingface.co/datasets/mixture-vitae/MixtureVitae-Omni"
        f"/resolve/main/{first_file.path}"
    )
    MAX_RECORDS = 5000
    print(f"  Streaming up to {MAX_RECORDS} records from {first_file.path} ...")

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(url, headers=headers)

    counts = {"seed": 0, "snac": 0, "text_chars": 0}
    n_records = 0
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            with gzip.open(resp, "rt", encoding="utf-8", errors="replace") as gz:
                for line in gz:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = d.get("text", "")
                    counts["seed"] += len(SEED_PAT.findall(text))
                    counts["snac"] += len(SNAC_PAT.findall(text))
                    clean = ALL_VLA_PAT.sub("", text)
                    counts["text_chars"] += len(clean.strip())
                    n_records += 1
                    if n_records >= MAX_RECORDS:
                        break
    except Exception as e:
        if n_records == 0:
            raise
        print(f"  Stream ended early after {n_records} records: {e}")

    # Scale from one sampled file to all files by compressed-size ratio
    scale = total_compressed / sample_file_size
    print(f"  Sampled {n_records:,} records from 1 file, scale factor: {scale:.1f}x")

    return {
        "records": int(n_records * scale),
        "size_gb": total_compressed / 1e9,
        "seed2": int(counts["seed"] * scale),
        "cosmos": 0,
        "avclm": 0,
        "agent": 0,
        "snac": int(counts["snac"] * scale),
        "text_tokens": int(counts["text_chars"] / 4 * scale),
        "note": f"streamed {n_records} records from 1 file, scaled {scale:.1f}x",
    }


def count_valid_with_seed():
    """Count tokens in valid_with_seed by downloading and sampling 1 shard."""
    print("\n=== Sampling valid_with_seed (1 shard of 64) ===")
    import shutil
    from huggingface_hub import hf_hub_download, HfApi

    token = os.environ.get("HF_TOKEN", "")
    api = HfApi(token=token or None)
    hf_kwargs = {"token": token} if token else {}

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
        **hf_kwargs,
    )

    # The structure is doubly-nested:
    #   valid_with_seed_shard_00000.tar.gz
    #   └── shard_NNNNN.tar.gz  (×9, ~2 GB each)
    #       └── *.jsonl  (actual data)
    # We handle arbitrary nesting by recursing with streaming (r|gz) inner tars.

    TEXT_KEYS = ("text", "content", "caption", "instruction", "output", "input")
    MAX_RECORDS = 5000

    counts = {"seed": 0, "seed2": 0, "snac": 0, "cosmos": 0, "text_chars": 0}
    n_records = [0]  # list so nested functions can mutate it

    def _count_text(text):
        counts["seed"] += len(SEED_PAT.findall(text))
        counts["seed2"] += len(SEED2_PAT.findall(text))
        counts["snac"] += len(SNAC_PAT.findall(text))
        counts["cosmos"] += len(COSMOS_PAT.findall(text))
        clean = ALL_VLA_PAT.sub("", text)
        counts["text_chars"] += len(clean.strip())

    def _extract_text(d):
        """Pull a text string out of a parsed JSON object."""
        if not isinstance(d, dict):
            return ""
        for key in TEXT_KEYS:
            val = d.get(key, "")
            if isinstance(val, str) and val:
                return val
            if isinstance(val, list):
                return " ".join(
                    (item.get("content", "") if isinstance(item, dict) else str(item))
                    for item in val if item
                )
        return ""

    def _process_fileobj(fobj, name_lower):
        """Read a file-like object as JSONL/text and count tokens."""
        try:
            raw = fobj.read()
            if name_lower.endswith(".gz") and not name_lower.endswith(".tar.gz"):
                raw = gzip.decompress(raw)
            text_data = raw.decode("utf-8", errors="replace")
        except Exception:
            return
        for line in text_data.splitlines():
            if n_records[0] >= MAX_RECORDS:
                return
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                text = _extract_text(d)
            except (json.JSONDecodeError, UnicodeDecodeError):
                text = line  # plain-text fallback
            if text:
                _count_text(text)
            n_records[0] += 1
            if n_records[0] % 2000 == 0:
                print(f"    {n_records[0]:,} records ...", flush=True)

    def _process_tar(fobj, mode):
        """Recursively iterate a tar archive, diving into nested tar.gz members."""
        if n_records[0] >= MAX_RECORDS:
            return
        try:
            with tarfile.open(fileobj=fobj, mode=mode) as inner:
                for member in inner:
                    if n_records[0] >= MAX_RECORDS:
                        return
                    if not member.isfile() or member.size == 0:
                        continue
                    name_lower = member.name.lower()
                    mfobj = inner.extractfile(member)
                    if mfobj is None:
                        continue
                    if name_lower.endswith((".tar.gz", ".tgz")):
                        # Nested tar — use streaming mode so we don't load 2GB into RAM
                        _process_tar(mfobj, "r|gz")
                    elif name_lower.endswith(".tar"):
                        _process_tar(mfobj, "r|")
                    else:
                        _process_fileobj(mfobj, name_lower)
        except Exception as e:
            print(f"    tar error ({mode}): {e}")

    print(f"  Counting tokens (up to {MAX_RECORDS} records, handles nested tars)...")
    t0 = time.time()
    with tarfile.open(shard_path, "r:gz") as outer:
        for member in outer:
            if n_records[0] >= MAX_RECORDS:
                break
            if not member.isfile() or member.size == 0:
                continue
            name_lower = member.name.lower()
            fobj = outer.extractfile(member)
            if fobj is None:
                continue
            if name_lower.endswith((".tar.gz", ".tgz")):
                _process_tar(fobj, "r|gz")
            elif name_lower.endswith(".tar"):
                _process_tar(fobj, "r|")
            else:
                _process_fileobj(fobj, name_lower)

    n_records = n_records[0]

    scale = total_compressed / shard_size
    print(f"  Sampled {n_records:,} records from 1 shard, scale factor: {scale:.1f}x")
    print(f"  Token types found: seed={counts['seed']:,}, seed2={counts['seed2']:,}, "
          f"snac={counts['snac']:,}, cosmos={counts['cosmos']:,}")

    # Clean up downloaded shard
    try:
        os.unlink(shard_path)
        cache_parent = os.path.join(TMP_DIR, "datasets--mixture-vitae-backup--MixtureVitae-Backup")
        shutil.rmtree(cache_parent, ignore_errors=True)
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
    """Generate a 4-panel dashboard: pie + stacked bar + per-dataset breakdown + coverage heatmap."""
    MODALITIES   = ["seed2",        "cosmos",        "avclm",         "agent",          "snac",          "text_tokens"]
    MOD_LABELS   = ["Seed2\n(image keyframe)", "Cosmos\n(video spatial)", "AVC-LM\n(H.264 BPE)",
                    "Agent\n(3D pose)",  "SNAC\n(audio)",  "Text"]
    MOD_COLORS   = ["#43A047", "#1E88E5", "#FB8C00", "#E53935", "#8E24AA", "#546E7A"]
    DS_COLORS    = ["#FF5722", "#3F51B5", "#009688", "#FFC107", "#795548", "#607D8B"]

    ds_names = list(datasets.keys())

    # Aggregate by modality across all datasets
    modality_totals = defaultdict(int)
    for ds in datasets.values():
        for m in MODALITIES:
            modality_totals[m] += ds.get(m, 0)

    # Per-dataset totals
    ds_totals = [sum(datasets[n].get(m, 0) for m in MODALITIES) for n in ds_names]
    grand_total = sum(ds_totals)

    # Short dataset display names
    short_names = []
    for n in ds_names:
        if "FineVideo" in n:
            short_names.append("FineVideo-VLA")
        elif "Omni" in n:
            short_names.append("MV-Omni")
        elif "valid_with_seed" in n:
            short_names.append("MV-Backup\nvalid_with_seed")
        else:
            short_names.append(n[:20])

    def fmt_b(v):
        if v >= 1e12:
            return f"{v/1e12:.1f}T"
        if v >= 1e9:
            return f"{v/1e9:.2f}B"
        if v >= 1e6:
            return f"{v/1e6:.1f}M"
        return str(int(v))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor("#F8F9FA")

    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32,
                          left=0.07, right=0.97, top=0.91, bottom=0.07)
    ax_pie   = fig.add_subplot(gs[0, 0])
    ax_ds    = fig.add_subplot(gs[0, 1])
    ax_stack = fig.add_subplot(gs[1, 0])
    ax_cov   = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        f"VLA Dataset Inventory  —  {fmt_b(grand_total)} tokens total",
        fontsize=18, fontweight="bold", color="#212121", y=0.97
    )

    # ── Panel 1: Donut by modality ──────────────────────────────────────────
    nonzero_idx = [i for i, m in enumerate(MODALITIES) if modality_totals[m] > 0]
    pie_vals    = [modality_totals[MODALITIES[i]] for i in nonzero_idx]
    pie_colors  = [MOD_COLORS[i]  for i in nonzero_idx]
    pie_labels  = [MOD_LABELS[i]  for i in nonzero_idx]

    wedges, _, autotexts = ax_pie.pie(
        pie_vals, colors=pie_colors, autopct="%1.1f%%",
        startangle=140, pctdistance=0.78,
        wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
        textprops={"fontsize": 9},
    )
    for at in autotexts:
        at.set_fontsize(8.5)
        at.set_fontweight("bold")
    # Centre hole → donut
    centre = plt.Circle((0, 0), 0.52, color="#F8F9FA")
    ax_pie.add_patch(centre)
    ax_pie.text(0, 0.08, fmt_b(sum(pie_vals)), ha="center", va="center",
                fontsize=14, fontweight="bold", color="#212121")
    ax_pie.text(0, -0.14, "total tokens", ha="center", va="center",
                fontsize=9, color="#616161")
    ax_pie.legend(wedges, [f"{l.replace(chr(10),' ')}  {fmt_b(v)}"
                            for l, v in zip(pie_labels, pie_vals)],
                  loc="lower center", bbox_to_anchor=(0.5, -0.22),
                  fontsize=8, ncol=2, frameon=False)
    ax_pie.set_title("Token Mix by Modality", fontsize=12, fontweight="bold",
                     pad=10, color="#212121")

    # ── Panel 2: Horizontal bar by dataset ──────────────────────────────────
    sorted_idx = sorted(range(len(ds_names)), key=lambda i: ds_totals[i])
    bar_vals   = [ds_totals[i]  for i in sorted_idx]
    bar_labels = [short_names[i] for i in sorted_idx]
    bar_colors = [DS_COLORS[i % len(DS_COLORS)] for i in sorted_idx]
    y_pos = range(len(bar_vals))

    bars = ax_ds.barh(list(y_pos), bar_vals, color=bar_colors,
                      edgecolor="white", linewidth=1.2, height=0.6)
    for bar, val in zip(bars, bar_vals):
        ax_ds.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                   fmt_b(val), va="center", ha="left", fontsize=9, fontweight="bold",
                   color="#212121")
    ax_ds.set_yticks(list(y_pos))
    ax_ds.set_yticklabels(bar_labels, fontsize=9)
    ax_ds.set_xlabel("Tokens", fontsize=9)
    ax_ds.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: fmt_b(x))
    )
    ax_ds.set_title("Tokens by Dataset", fontsize=12, fontweight="bold",
                    pad=10, color="#212121")
    ax_ds.spines[["top", "right"]].set_visible(False)
    ax_ds.set_facecolor("#F8F9FA")

    # ── Panel 3: Stacked bar (modality breakdown per dataset) ───────────────
    x = np.arange(len(ds_names))
    bar_w = 0.55
    bottoms = np.zeros(len(ds_names))
    for mi, (mod, color) in enumerate(zip(MODALITIES, MOD_COLORS)):
        vals = np.array([datasets[n].get(mod, 0) for n in ds_names], dtype=float)
        if vals.sum() == 0:
            continue
        ax_stack.bar(x, vals, bar_w, bottom=bottoms, color=color,
                     label=MOD_LABELS[mi].replace("\n", " "), edgecolor="white", linewidth=0.8)
        bottoms += vals

    ax_stack.set_xticks(x)
    ax_stack.set_xticklabels(short_names, fontsize=8.5)
    ax_stack.set_ylabel("Tokens", fontsize=9)
    ax_stack.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: fmt_b(y))
    )
    ax_stack.legend(loc="upper left", fontsize=7.5, frameon=True,
                    framealpha=0.9, ncol=2)
    ax_stack.set_title("Modality Breakdown per Dataset", fontsize=12,
                       fontweight="bold", pad=10, color="#212121")
    ax_stack.spines[["top", "right"]].set_visible(False)
    ax_stack.set_facecolor("#F8F9FA")

    # ── Panel 4: Coverage heatmap (% of each modality present per dataset) ──
    present_mods = [m for m in MODALITIES if modality_totals[m] > 0]
    present_labels = [MOD_LABELS[MODALITIES.index(m)].replace("\n", " ")
                      for m in present_mods]
    heat_data = np.zeros((len(ds_names), len(present_mods)))
    for di, name in enumerate(ds_names):
        ds = datasets[name]
        for mi, mod in enumerate(present_mods):
            total_mod = modality_totals[mod]
            if total_mod > 0:
                heat_data[di, mi] = ds.get(mod, 0) / total_mod * 100

    im = ax_cov.imshow(heat_data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax_cov.set_xticks(range(len(present_mods)))
    ax_cov.set_xticklabels(present_labels, rotation=30, ha="right", fontsize=8)
    ax_cov.set_yticks(range(len(ds_names)))
    ax_cov.set_yticklabels(short_names, fontsize=8.5)
    for di in range(len(ds_names)):
        for mi in range(len(present_mods)):
            val = heat_data[di, mi]
            color = "white" if val > 55 else "#212121"
            ax_cov.text(mi, di, f"{val:.0f}%", ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color=color)
    cbar = fig.colorbar(im, ax=ax_cov, fraction=0.035, pad=0.03)
    cbar.set_label("% of modality total", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax_cov.set_title("Dataset Share per Modality (%)", fontsize=12,
                     fontweight="bold", pad=10, color="#212121")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
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
