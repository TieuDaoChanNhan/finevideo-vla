#!/usr/bin/env python3
"""
CP count vs compression/accuracy tradeoff analysis.

For each fixed CP count (1, 2, 3, 4, 5, 6, 7, 8):
  - Token count per window
  - Mean reconstruction error (mm) vs ground truth

Also reports:
  - 1-CP eligibility among current 2-CP joints (quantized start == end)
  - Token overhead breakdown (wrappers + t tokens vs xyz payload)

Usage:
    python tools/analyze_cp_tradeoff.py
    python tools/analyze_cp_tradeoff.py --samples 300 --data-dir /path/to/yolo_cleaned_30fps
"""

import argparse
import json
import os
import random
import sys

import numpy as np

JOINTS = [
    "pelvis", "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle", "spine", "thorax",
    "nose", "head_top",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_shoulder", "r_elbow", "r_wrist",
]
N_JOINTS = len(JOINTS)
WINDOW_FRAMES = 8
COORD_RANGE = 2.0

# Current adaptive thresholds from phase5
TAU_LOW = 0.005
TAU_HIGH = 0.05

DEFAULT_DATA_DIR = "/p/data1/mmlaion/shared/nguyen38/data/outputs/yolo_cleaned_30fps"


def quantize(v: float) -> int:
    return int(np.clip(round((v + COORD_RANGE) / (2.0 * COORD_RANGE) * 255), 0, 255))


def dequantize(n: int) -> float:
    return n / 255.0 * (2.0 * COORD_RANGE) - COORD_RANGE


def tokens_per_window(n_cp: int) -> int:
    """Token count for one window with all joints at same fixed CP count."""
    if n_cp == 1:
        # No t token: <name> <x> <y> <z> </name>
        return 1 + N_JOINTS * (2 + 3)
    else:
        # <name> [<t_i> <x> <y> <z>] × n_cp </name>
        return 1 + N_JOINTS * (2 + n_cp * 4)


def cp_indices_for_n(n_cp: int, n_frames: int = WINDOW_FRAMES) -> np.ndarray:
    """Evenly-spaced CP frame indices for a given CP count."""
    if n_cp == 1:
        return np.array([0])
    return np.round(np.linspace(0, n_frames - 1, n_cp)).astype(int)


def pchip_reconstruct(frames_all: int, cp_t: np.ndarray, cp_vals: np.ndarray) -> np.ndarray:
    """
    Reconstruct trajectory at all frame positions using PCHIP interpolation.
    cp_t: (n_cp,) control point frame indices
    cp_vals: (n_cp, 3) control point xyz values
    Returns: (frames_all, 3)
    """
    from scipy.interpolate import PchipInterpolator
    t_query = np.arange(frames_all)
    if len(cp_t) == 1:
        # Constant: repeat single value
        return np.tile(cp_vals[0], (frames_all, 1))
    elif len(cp_t) == 2:
        # Linear interpolation (PCHIP reduces to linear for 2 points)
        out = np.zeros((frames_all, 3))
        for d in range(3):
            out[:, d] = np.interp(t_query, cp_t, cp_vals[:, d])
        return out
    else:
        out = np.zeros((frames_all, 3))
        for d in range(3):
            interp = PchipInterpolator(cp_t, cp_vals[:, d])
            out[:, d] = interp(t_query)
        return out


def reconstruct_with_quantization(trajectory: np.ndarray, n_cp: int) -> np.ndarray:
    """
    Simulate quantization: select CPs, quantize to uint8, dequantize, then interpolate.
    trajectory: (8, 3) float32
    Returns: (8, 3) reconstructed trajectory
    """
    cp_t = cp_indices_for_n(n_cp)
    cp_vals_raw = trajectory[cp_t]  # (n_cp, 3)

    # Quantize and dequantize to simulate token round-trip
    cp_vals_q = np.array([[dequantize(quantize(v)) for v in row] for row in cp_vals_raw])

    return pchip_reconstruct(WINDOW_FRAMES, cp_t, cp_vals_q)


def joint_curvature(traj: np.ndarray) -> float:
    """Max 2nd-derivative norm for a (8, 3) trajectory."""
    if traj.shape[0] < 3:
        return 0.0
    vel = np.diff(traj, axis=0)
    acc = np.diff(vel, axis=0)
    return float(np.max(np.linalg.norm(acc, axis=1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--samples", type=int, default=200,
                        help="Number of video files to sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_files = [f for f in os.listdir(args.data_dir) if f.endswith("_cleaned.jsonl")]
    if not all_files:
        print(f"ERROR: no *_cleaned.jsonl in {args.data_dir}")
        sys.exit(1)

    random.seed(args.seed)
    sample_files = random.sample(all_files, min(args.samples, len(all_files)))
    print(f"Sampling {len(sample_files)}/{len(all_files)} files...")

    # Accumulators
    # errors[n_cp] = list of per-joint MAE in meters
    errors = {n: [] for n in range(1, 9)}
    # windows per adaptive tier (for 1-CP static test)
    tier2_total = 0
    tier2_static = 0  # quantized start == end for all 3 dims
    total_windows = 0

    for fname in sample_files:
        fpath = os.path.join(args.data_dir, fname)
        try:
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    states_raw = rec.get("states")
                    if states_raw is None:
                        continue

                    # Replace null → nan, then check for completeness
                    states = np.array(
                        [[[x if x is not None else float('nan') for x in xyz]
                          for xyz in frame]
                         for frame in states_raw], dtype=np.float32
                    )
                    if states.shape != (WINDOW_FRAMES, N_JOINTS, 3):
                        continue
                    if np.isnan(states).any():
                        continue

                    total_windows += 1

                    # Per CP count: measure reconstruction error
                    for n_cp in range(1, 9):
                        joint_errors = []
                        for j in range(N_JOINTS):
                            traj = states[:, j, :]  # (8, 3)
                            recon = reconstruct_with_quantization(traj, n_cp)
                            mae = np.mean(np.abs(traj - recon))  # mean over all frames and dims
                            joint_errors.append(mae)
                        errors[n_cp].extend(joint_errors)

                    # 1-CP static test: check tier-2 joints (low curvature)
                    for j in range(N_JOINTS):
                        traj = states[:, j, :]
                        curv = joint_curvature(traj)
                        if curv < TAU_LOW:
                            tier2_total += 1
                            q0 = tuple(quantize(v) for v in traj[0])
                            q7 = tuple(quantize(v) for v in traj[-1])
                            if q0 == q7:
                                tier2_static += 1

        except Exception as e:
            print(f"  WARN {fname}: {e}")

    if total_windows == 0:
        print("ERROR: 0 valid windows found.")
        sys.exit(1)

    sep = "=" * 68
    print()
    print(sep)
    print("  CP COUNT vs COMPRESSION / ACCURACY TRADEOFF")
    print(sep)
    print(f"  Windows analyzed : {total_windows:,}")
    print(f"  Joint samples    : {total_windows * N_JOINTS:,}")
    print()

    # Token counts (theoretical, if all joints at same CP count)
    print("  A) TOKEN COUNT per window (all 17 joints at same N)")
    print(f"     {'N CP':>4}  {'Tokens/win':>10}  {'vs 8-CP baseline':>18}  {'Format'}")
    baseline = tokens_per_window(8)
    for n in range(1, 9):
        t = tokens_per_window(n)
        saving = (1 - t / baseline) * 100
        note = " ← current min" if n == 2 else (" ← baseline" if n == 8 else "")
        if n == 1:
            note = " ← no t token (static)"
        print(f"     {n:>4}  {t:>10}  {saving:>16.1f}%  {note}")

    print()
    print(f"  B) RECONSTRUCTION ERROR per joint (mean absolute error, mm)")
    print(f"     {'N CP':>4}  {'MAE (mm)':>10}  {'Max joint err (mm)':>20}  Note")
    for n in range(1, 9):
        errs = errors[n]
        if not errs:
            continue
        mean_mm = np.mean(errs) * 1000
        p95_mm = np.percentile(errs, 95) * 1000
        note = ""
        if n == 1:
            note = "constant (no interp)"
        elif n == 2:
            note = "linear (current min)"
        elif n == 8:
            note = "all frames (baseline)"
        print(f"     {n:>4}  {mean_mm:>10.1f}  {p95_mm:>20.1f}  {note}")

    print()
    print("  C) 1-CP STATIC TEST (among current tier-2 joints)")
    if tier2_total > 0:
        static_pct = tier2_static / tier2_total * 100
        print(f"     Tier-2 joint-windows (curv < tau_low) : {tier2_total:,}")
        print(f"     Of those, quantized start==end (all 3d): {tier2_static:,} ({static_pct:.1f}%)")
        # eligible_per_window = avg number of joints/window that qualify for 1-CP
        eligible_per_window = tier2_static / total_windows
        tokens_saved = eligible_per_window * (10 - 5)  # 5 tokens saved per 1-CP joint
        print(f"     Avg qualifying joints per window       : {eligible_per_window:.1f}")
        print(f"     Avg extra tokens saved by 1-CP         : {tokens_saved:.1f} tokens/window")
        current_adaptive_avg = 284.1
        new_avg = current_adaptive_avg - tokens_saved
        extra_saving_pct = tokens_saved / current_adaptive_avg * 100
        print(f"     Current adaptive avg                   : {current_adaptive_avg:.0f} tokens")
        print(f"     Estimated new avg (with 1-CP)          : {new_avg:.0f} tokens")
        print(f"     Additional compression                 : {extra_saving_pct:.1f}%")
    else:
        print("     No tier-2 joints found in sample.")

    print()

    # D: Token overhead breakdown
    print("  D) TOKEN OVERHEAD BREAKDOWN (for current adaptive avg = 284 tokens)")
    fps_tok = 1
    # Reproduce from known stats: 55.2% 2-CP, 25.6% 4-CP, 19.2% 8-CP
    avg_per_joint = 0.552 * 10 + 0.256 * 18 + 0.192 * 34
    total_joint_tokens = N_JOINTS * avg_per_joint
    wrapper_tokens = N_JOINTS * 2  # open + close
    t_tokens_per_joint = 0.552 * 2 + 0.256 * 4 + 0.192 * 8
    t_tokens_total = N_JOINTS * t_tokens_per_joint
    xyz_tokens_total = total_joint_tokens - wrapper_tokens - t_tokens_total
    overhead = wrapper_tokens + t_tokens_total + fps_tok
    print(f"     fps token      : {fps_tok:>6.1f}  ({fps_tok/(284)*100:.1f}%)")
    print(f"     wrapper tokens : {wrapper_tokens:>6.1f}  ({wrapper_tokens/284*100:.1f}%)")
    print(f"     t tokens       : {t_tokens_total:>6.1f}  ({t_tokens_total/284*100:.1f}%)")
    print(f"     xyz tokens     : {xyz_tokens_total:>6.1f}  ({xyz_tokens_total/284*100:.1f}%)")
    print(f"     OVERHEAD total : {overhead:>6.1f}  ({overhead/284*100:.1f}%)  (non-payload)")

    print()
    print(sep)


if __name__ == "__main__":
    main()
