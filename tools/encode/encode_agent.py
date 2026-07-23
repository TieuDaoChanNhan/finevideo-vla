#!/usr/bin/env python3
"""
Agent (3D pose) encoder -- turns a real 17-joint pose window into `<agent>`
tokens, the reverse of tools/eval/decode_agent_tokens.py. Reuses
pipeline_pose/phase5_adaptive_pchip.py's build_token_str() verbatim (pure
numpy, no cluster dependency at all) rather than re-deriving the adaptive
PCHIP control-point selection.

Use case: you have a REAL 3D pose sequence (motion capture, or your own
video run through an HRNet+MotionBERT-style pipeline) and want the model to
continue/predict from it -- this is exactly the "agent completion" behavior
already verified for this model (give a partial <agent> block, it completes
all 17 joints). Unlike seed2/cosmos/snac, this is the only encoder where
"raw input" isn't a stock media file -- it's already-estimated 3D joint
positions, which is a fair thing to require (you can't derive metric 3D pose
from nothing; some upstream pose-estimation step is unavoidable no matter
who's doing the encoding).

Input contract (IMPORTANT, easy to get wrong):
  - shape (8, 17, 3) float -- exactly 8 frames (this model's WINDOW_FRAMES),
    NOT 24 (that's the newer 2026-07-23 pipeline convention this model never
    saw), 17 joints in the exact order below, xyz in METRES.
  - ROOT-CENTERED: pelvis (joint 0) must be at [0,0,0] in every frame --
    subtract the pelvis position from all 17 joints per-frame yourself first
    if your source data isn't already root-relative (see
    pipeline_pose/phase3_kinematics_processor.py's split_root_motion() for
    the exact convention this project uses).
  - Values should stay within [-2.0, +2.0]m per axis (COORD_RANGE) --
    quantize() clips silently outside that range, so a badly-scaled pose
    (e.g. millimetres instead of metres) will silently flatten to the
    boundary rather than erroring. No automatic unit detection is attempted.

Joint order:
    pelvis, r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle, spine, thorax,
    nose, head_top, l_shoulder, l_elbow, l_wrist, r_shoulder, r_elbow, r_wrist

Usage:
    python tools/encode/encode_agent.py --input pose.json
    # pose.json: {"states": [[[x,y,z], ...17 joints...], ...8 frames...]}
    python tools/encode/encode_agent.py --input pose.npy
    # pose.npy: numpy array, shape (8, 17, 3)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pipeline_pose"))
from phase5_adaptive_pchip import build_token_str, JOINT_NAMES, N_JOINTS, TARGET_FPS, COORD_RANGE  # noqa: E402

WINDOW_FRAMES = 8  # this model's convention -- NOT the newer 24-frame pipeline


def load_states(path: str):
    import numpy as np

    if path.endswith(".npy"):
        states = np.load(path)
    else:
        with open(path) as f:
            data = json.load(f)
        states = np.array(data["states"], dtype=np.float32)

    if states.shape != (WINDOW_FRAMES, N_JOINTS, 3):
        raise ValueError(
            f"Expected shape ({WINDOW_FRAMES}, {N_JOINTS}, 3), got {states.shape}. "
            f"This model was trained on 8-frame windows -- 24-frame input (the newer "
            f"pipeline convention) will NOT tokenize correctly here."
        )

    pelvis = states[:, 0, :]
    if not (abs(pelvis).max() < 1e-4):
        print(f"WARNING: pelvis (joint 0) is not at origin (max |pelvis|={abs(pelvis).max():.4f}m) -- "
              f"auto-centering now. If this wasn't intended, check your source data's convention.",
              file=sys.stderr)
        states = states - pelvis[:, None, :]

    bad = states[abs(states) > COORD_RANGE]
    if bad.size > 0:
        print(f"WARNING: {bad.size} coordinate value(s) outside [-{COORD_RANGE}, {COORD_RANGE}]m -- "
              f"will be silently clipped by quantize(). Check units (expected metres).",
              file=sys.stderr)

    return states


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help=".json (with a 'states' key) or .npy file, shape (8,17,3)")
    ap.add_argument("--fps", type=int, default=TARGET_FPS)
    args = ap.parse_args()

    states = load_states(args.input)
    token_str, cp_counts = build_token_str(states, fps=args.fps)

    print(f"Encoded {N_JOINTS} joints, {sum(cp_counts.values())} total control points:")
    for name in JOINT_NAMES:
        print(f"  {name}: {cp_counts[name]} CPs")
    print()
    print("<agent> " + token_str + " </agent>")


if __name__ == "__main__":
    main()
