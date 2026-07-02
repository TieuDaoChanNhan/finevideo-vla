#!/usr/bin/env python3
"""
Adaptive PCHIP compression analysis.

Measures:
  A) Token saving vs hypothetical fixed 8-CP baseline
  B) CP tier distribution (2/4/8) per joint and overall
  C) Pelvis coordinate sanity check (confirms root-centering is working)

Token count formula per joint:
  tokens = 2 (open + close tag) + num_cp * 4 (t + x + y + z per control point)
  e.g. 2-CP joint = 10 tokens, 4-CP = 18, 8-CP = 34

Fixed 8-CP baseline:
  17 joints * 34 = 578 tokens/window  (+1 for <fps_30> = 579 total)

Usage:
    python tools/analyze_pchip_compression.py
    python tools/analyze_pchip_compression.py --files 500   # sample 500 files
    python tools/analyze_pchip_compression.py --data-dir /path/to/agent_tokens_adaptive
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

JOINTS = [
    "pelvis", "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle", "spine", "thorax",
    "nose", "head_top",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_shoulder", "r_elbow", "r_wrist",
]

FIXED_CP = 8
TOKENS_PER_CP = 4          # t + x + y + z
JOINT_WRAPPER_TOKENS = 2   # <joint_name> + </joint_name>
FPS_TOKEN = 1              # <fps_30>

DEFAULT_DATA_DIR = (
    "/p/data1/mmlaion/shared/nguyen38/data/outputs/agent_tokens_adaptive"
)


def tokens_for_joint(num_cp: int) -> int:
    return JOINT_WRAPPER_TOKENS + num_cp * TOKENS_PER_CP


def fixed_tokens_per_window() -> int:
    return FPS_TOKEN + len(JOINTS) * tokens_for_joint(FIXED_CP)


def adaptive_tokens_per_window(cp_counts: dict) -> int:
    return FPS_TOKEN + sum(tokens_for_joint(cp_counts[j]) for j in JOINTS)


def parse_pelvis_xyz(token_str: str):
    """
    Extract pelvis x/y/z quantized values from the first pelvis block.
    Returns list of (x, y, z) tuples — one per control point in that block.
    Quantized uint8 [0..255] maps to [-2.0m, +2.0m] via: pos = val/255*4 - 2
    """
    block_match = re.search(r'<pelvis>(.*?)</pelvis>', token_str)
    if not block_match:
        return []
    block = block_match.group(1)
    xs = [int(m.group(1)) for m in re.finditer(r'<pelvis_x_(\d+)>', block)]
    ys = [int(m.group(1)) for m in re.finditer(r'<pelvis_y_(\d+)>', block)]
    zs = [int(m.group(1)) for m in re.finditer(r'<pelvis_z_(\d+)>', block)]
    return list(zip(xs, ys, zs))


def dequant(v: int) -> float:
    return v / 255.0 * 4.0 - 2.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help="Directory with {video_id}_tokens.jsonl files",
    )
    parser.add_argument(
        "--files", type=int, default=200,
        help="Number of video files to sample (default: 200, use 0 for all)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    all_files = [
        f for f in os.listdir(args.data_dir) if f.endswith("_tokens.jsonl")
    ]
    if not all_files:
        print(f"ERROR: no *_tokens.jsonl files found in {args.data_dir}")
        sys.exit(1)

    random.seed(args.seed)
    if args.files > 0 and args.files < len(all_files):
        sampled = random.sample(all_files, args.files)
    else:
        sampled = all_files

    print(f"Analyzing {len(sampled)} / {len(all_files)} video files...")

    fixed_baseline = fixed_tokens_per_window()

    # --- Accumulators ---
    total_windows = 0
    total_adaptive_tokens = 0
    total_fixed_tokens = 0

    # CP tier counters: per joint → {2: count, 4: count, 8: count}
    joint_tier_counts = {j: defaultdict(int) for j in JOINTS}
    overall_tier_counts = defaultdict(int)

    # Pelvis coordinate samples for sanity check
    pelvis_samples = []   # list of (x_m, y_m, z_m) at t=0

    for fname in sampled:
        fpath = os.path.join(args.data_dir, fname)
        try:
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    cp = rec.get("cp_counts", {})
                    if not cp or set(cp.keys()) != set(JOINTS):
                        continue

                    # A: token counting
                    adaptive = adaptive_tokens_per_window(cp)
                    total_adaptive_tokens += adaptive
                    total_fixed_tokens += fixed_baseline
                    total_windows += 1

                    # B: tier distribution
                    for joint in JOINTS:
                        tier = cp[joint]
                        joint_tier_counts[joint][tier] += 1
                        overall_tier_counts[tier] += 1

                    # C: pelvis sanity (collect first 500)
                    if len(pelvis_samples) < 500:
                        pts = parse_pelvis_xyz(rec.get("token_str", ""))
                        if pts:
                            x, y, z = pts[0]
                            pelvis_samples.append(
                                (dequant(x), dequant(y), dequant(z))
                            )
        except Exception as e:
            print(f"  WARN: {fname}: {e}")
            continue

    if total_windows == 0:
        print("ERROR: 0 valid windows found.")
        sys.exit(1)

    saving_pct = (1 - total_adaptive_tokens / total_fixed_tokens) * 100
    mean_adaptive = total_adaptive_tokens / total_windows
    mean_fixed = fixed_baseline

    # --- Report ---
    sep = "=" * 68
    print()
    print(sep)
    print("  ADAPTIVE PCHIP COMPRESSION ANALYSIS")
    print(sep)
    print(f"  Files sampled   : {len(sampled):,}  ({total_windows:,} windows)")
    print()
    print("  A) TOKEN SAVING vs FIXED 8-CP BASELINE")
    print(f"     Fixed 8-CP tokens/window  : {mean_fixed}")
    print(f"     Adaptive tokens/window    : {mean_adaptive:.1f}")
    print(f"     Token saving              : {saving_pct:.1f}%")
    print(f"     Total adaptive tokens     : {total_adaptive_tokens:,}")
    print(f"     Total if fixed 8-CP       : {total_fixed_tokens:,}")
    print()

    print("  B) CP TIER DISTRIBUTION")
    total_jw = sum(overall_tier_counts.values())
    for tier in [2, 4, 8]:
        n = overall_tier_counts[tier]
        print(f"     {tier}-CP : {n:>10,}  ({n/total_jw*100:5.1f}%)")
    print()

    # Per-joint breakdown: sort by % of 8-CP (most dynamic first)
    print("  B2) PER-JOINT CP TIER — sorted most→least dynamic")
    print(f"     {'Joint':<14} {'2-CP':>8} {'4-CP':>8} {'8-CP':>8}  (% at each tier)")
    joint_8cp_pct = {
        j: joint_tier_counts[j][8] / sum(joint_tier_counts[j].values()) * 100
        for j in JOINTS
    }
    for j in sorted(JOINTS, key=lambda x: -joint_8cp_pct[x]):
        counts = joint_tier_counts[j]
        total_j = sum(counts.values())
        p2 = counts[2] / total_j * 100
        p4 = counts[4] / total_j * 100
        p8 = counts[8] / total_j * 100
        print(f"     {j:<14}  {p2:5.1f}%   {p4:5.1f}%   {p8:5.1f}%")
    print()

    print("  C) PELVIS COORDINATE SANITY CHECK (root-centering)")
    if pelvis_samples:
        xs = [p[0] for p in pelvis_samples]
        ys = [p[1] for p in pelvis_samples]
        zs = [p[2] for p in pelvis_samples]
        print(f"     Samples: {len(pelvis_samples)}")
        print(f"     Pelvis X — mean: {sum(xs)/len(xs):+.3f}m  "
              f"range: [{min(xs):.3f}, {max(xs):.3f}]m")
        print(f"     Pelvis Y — mean: {sum(ys)/len(ys):+.3f}m  "
              f"range: [{min(ys):.3f}, {max(ys):.3f}]m")
        print(f"     Pelvis Z — mean: {sum(zs)/len(zs):+.3f}m  "
              f"range: [{min(zs):.3f}, {max(zs):.3f}]m")
        print()

        near_zero = sum(
            1 for x, y, z in pelvis_samples
            if abs(x) < 0.1 and abs(y) < 0.1 and abs(z) < 0.1
        )
        pct_near = near_zero / len(pelvis_samples) * 100
        print(f"     Within ±0.1m of origin: {near_zero}/{len(pelvis_samples)} "
              f"({pct_near:.1f}%)")
        if pct_near > 70:
            print("     ✓ Pelvis is root-centered (absolute xyz is correct)")
        else:
            print("     ⚠ Pelvis is NOT near origin — check root-centering")
    else:
        print("     No pelvis samples extracted.")

    print()
    print(sep)


if __name__ == "__main__":
    main()
