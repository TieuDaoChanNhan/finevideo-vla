#!/usr/bin/env python3
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def inspect_file(path: Path, strides=(1, 8, 16), expect_frames=8, expect_joints=17, expect_dims=3, preview=10):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    total_lines = 0
    empty_lines = 0
    json_errors = 0
    missing_window_id = 0
    missing_states = 0

    valid_json_records = 0
    window_ids = []
    stride_pass = {s: 0 for s in strides}
    stride_pass_and_no_nan = {s: 0 for s in strides}

    shape_counter = Counter()
    nan_windows = 0
    inf_windows = 0
    non_numeric_windows = 0
    good_windows = 0

    first_bad = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            total_lines += 1
            line = raw.strip()
            if not line:
                empty_lines += 1
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                json_errors += 1
                if len(first_bad) < preview:
                    first_bad.append({"line": line_no, "reason": f"json_error: {e}"})
                continue

            if "window_id" not in rec:
                missing_window_id += 1
                if len(first_bad) < preview:
                    first_bad.append({"line": line_no, "reason": "missing_window_id"})
                continue
            if "states" not in rec:
                missing_states += 1
                if len(first_bad) < preview:
                    first_bad.append({"line": line_no, "reason": "missing_states", "window_id": rec.get("window_id")})
                continue

            valid_json_records += 1
            wid = rec["window_id"]
            window_ids.append(wid)

            states_raw = rec["states"]
            try:
                arr = np.asarray(states_raw, dtype=float)
            except Exception as e:
                non_numeric_windows += 1
                shape_counter["non_numeric"] += 1
                if len(first_bad) < preview:
                    first_bad.append({"line": line_no, "window_id": wid, "reason": f"non_numeric_states: {e}"})
                continue

            shape_counter[str(tuple(arr.shape))] += 1

            has_nan = np.isnan(arr).any()
            has_inf = np.isinf(arr).any()
            correct_shape = arr.shape == (expect_frames, expect_joints, expect_dims)

            for s in strides:
                if isinstance(wid, int) and wid % s == 0:
                    stride_pass[s] += 1
                    if correct_shape and not has_nan and not has_inf:
                        stride_pass_and_no_nan[s] += 1

            reasons = []
            if not correct_shape:
                reasons.append(f"bad_shape={arr.shape}")
            if has_nan:
                nan_windows += 1
                reasons.append("has_nan")
            if has_inf:
                inf_windows += 1
                reasons.append("has_inf")

            if reasons:
                if len(first_bad) < preview:
                    first_bad.append({"line": line_no, "window_id": wid, "reason": ", ".join(reasons)})
            else:
                good_windows += 1

    print("=" * 80)
    print(f"FILE: {path}")
    print("=" * 80)
    print(f"total_lines                : {total_lines}")
    print(f"empty_lines                : {empty_lines}")
    print(f"json_errors                : {json_errors}")
    print(f"missing_window_id          : {missing_window_id}")
    print(f"missing_states             : {missing_states}")
    print(f"valid_json_records         : {valid_json_records}")
    print(f"good_windows               : {good_windows}")
    print(f"nan_windows                : {nan_windows}")
    print(f"inf_windows                : {inf_windows}")
    print(f"non_numeric_windows        : {non_numeric_windows}")

    print("\nwindow_id stats")
    if window_ids:
        ints = [w for w in window_ids if isinstance(w, int)]
        print(f"  count                    : {len(window_ids)}")
        print(f"  int_count                : {len(ints)}")
        if ints:
            print(f"  min                      : {min(ints)}")
            print(f"  max                      : {max(ints)}")
            uniq = sorted(set(ints))
            gaps = np.diff(uniq) if len(uniq) > 1 else np.array([])
            if len(gaps) > 0:
                gap_counter = Counter(gaps.tolist())
                print(f"  most_common_gaps         : {gap_counter.most_common(10)}")
            print(f"  first_20_window_ids      : {uniq[:20]}")
    else:
        print("  no window_id found")

    print("\nshape distribution")
    for shape, cnt in shape_counter.most_common(10):
        print(f"  {shape:>20} : {cnt}")

    print("\nstride viability (matches tokenizer filter)")
    for s in strides:
        print(f"  stride={s:<3} pass_window_id={stride_pass[s]:<6} pass_and_encodable={stride_pass_and_no_nan[s]}")

    print("\nfirst problematic records")
    if first_bad:
        for item in first_bad:
            print(" ", item)
    else:
        print("  none")


def main():
    ap = argparse.ArgumentParser(description="Inspect a *_cleaned.jsonl file for tokenizer readiness.")
    ap.add_argument("file", type=Path, help="Path to *_cleaned.jsonl")
    ap.add_argument("--strides", type=int, nargs="*", default=[1, 8, 16], help="Stride values to test")
    ap.add_argument("--preview", type=int, default=10, help="Number of problematic records to print")
    args = ap.parse_args()
    inspect_file(args.file, strides=tuple(args.strides), preview=args.preview)


if __name__ == "__main__":
    main()
