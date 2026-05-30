#!/usr/bin/env python3
"""
Decode agent tokens from a random sample in the final_vla dataset.
Saves all decoded data to a JSON file alongside printing to stdout.

Usage:
    python decode_agent_tokens.py
    python decode_agent_tokens.py --output my_sample.json
    python decode_agent_tokens.py --output my_sample.json --seed 42

256-token layout per <agent> block:
  token[0]       — scale  (uint8, range [0, 2.0 m])
                   decode: scale = token / 255.0 * 2.0
  tokens[1–51]   — anchor (uint8, 17 joints × xyz, root-centred, range [-2, 2 m])
                   decode: coord = token / 255.0 * 4.0 - 2.0
  tokens[52–255] — motion control points (uint8, 4 CPs × 17 joints × xyz, [-1, 1])
                   decode: val = token / 127.5 - 1.0

Full skeleton reconstruction:
  1. Decode scale, anchor (shape 17×3), motion CPs (shape 4×17×3, normalised)
  2. cp_absolute = motion_CPs * scale + anchor   (anchor[pelvis] ≈ 0 since root-centred)
  3. PCHIP interpolation over cp_absolute at t in [0,1] → trajectory (n_frames, 17, 3)

H36M joint index map:
  0  pelvis      (root, always ~0)
  1  r_hip       7  spine        14 r_shoulder
  2  r_knee      8  thorax       15 r_elbow
  3  r_ankle     9  nose         16 r_wrist  <- right end-effector
  4  l_hip      10  head_top
  5  l_knee     11  l_shoulder
  6  l_ankle    12  l_elbow
               13  l_wrist  <- left end-effector
"""

import argparse
import glob
import json
import random
import re
import sys
from datetime import datetime

import numpy as np
from scipy.interpolate import PchipInterpolator

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_GLOB = "/e/scratch/reformo/nguyen38/FineVideo-VLA/final_dataset/final_vla_rank_*.jsonl"
N_FRAMES_RECONSTRUCT = 8

# Token layout constants (must match merge_agent_tokens.py)
N_JOINTS = 17
N_DIMS = 3
N_MOTION_TOKENS = 204    # 4 CPs × 17 joints × 3 dims
N_ANCHOR_TOKENS = N_JOINTS * N_DIMS   # 51
N_AGENT_TOKENS = 1 + N_ANCHOR_TOKENS + N_MOTION_TOKENS  # 256
ANCHOR_RANGE = 2.0   # metres
SCALE_MAX = 2.0      # metres

H36M_JOINT_NAMES = [
    "pelvis", "r_hip", "r_knee", "r_ankle",
    "l_hip",  "l_knee", "l_ankle", "spine",
    "thorax", "nose",   "head_top",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_shoulder", "r_elbow", "r_wrist",
]

ARM_JOINTS = {
    "l_shoulder": 11, "l_elbow": 12, "l_wrist": 13,
    "r_shoulder": 14, "r_elbow": 15, "r_wrist": 16,
}

# ── Core decode ───────────────────────────────────────────────────────────────

def decode_agent_block(inner: str) -> dict:
    """
    Parse a 256-token <agent> block into scale, anchor, and motion control points.

    Returns a dict with:
        scale     : float (metres)
        anchor    : ndarray (17, 3) root-centred absolute positions (metres)
        motion_cp : ndarray (4, 17, 3) normalised relative control points [-1, 1]
        cp_abs    : ndarray (4, 17, 3) absolute control points (metres)
    """
    values = list(map(int, inner.strip().split()))
    if len(values) != N_AGENT_TOKENS:
        raise ValueError(f"Expected {N_AGENT_TOKENS} tokens, got {len(values)}")

    scale_tok = values[0]
    anchor_toks = values[1 : 1 + N_ANCHOR_TOKENS]
    motion_toks = values[1 + N_ANCHOR_TOKENS :]

    scale = scale_tok / 255.0 * SCALE_MAX
    anchor = (
        np.array(anchor_toks, dtype=np.float32) / 255.0 * 2.0 * ANCHOR_RANGE - ANCHOR_RANGE
    ).reshape(N_JOINTS, N_DIMS)
    motion_cp = (
        np.array(motion_toks, dtype=np.float32) / 127.5 - 1.0
    ).reshape(4, N_JOINTS, N_DIMS)

    # Reconstruct absolute control points: scale normalised motion + first-frame anchor
    # anchor[0] = pelvis ≈ 0 (root-centred); other joints are their rest positions
    cp_abs = motion_cp * scale + anchor[0]   # broadcast pelvis position

    return {
        "scale": scale,
        "anchor": anchor,
        "motion_cp": motion_cp,
        "cp_abs": cp_abs,
    }


def reconstruct_trajectory(cp_abs: np.ndarray, n_frames: int = N_FRAMES_RECONSTRUCT) -> np.ndarray:
    """PCHIP reconstruction over absolute control points, evenly-spaced t in [0, 1]."""
    t_cp = np.array([0.0, 1/3, 2/3, 1.0])
    t = np.linspace(0.0, 1.0, n_frames)
    traj = np.zeros((n_frames, N_JOINTS, N_DIMS), dtype=np.float32)
    for j in range(N_JOINTS):
        for d in range(N_DIMS):
            spline = PchipInterpolator(t_cp, cp_abs[:, j, d])
            traj[:, j, d] = spline(t)
    return traj


def extract_all_agent_blocks(video_tokens: str) -> list:
    pattern = r"<agent>\s*(.*?)\s*</agent>"
    return [decode_agent_block(m) for m in re.findall(pattern, video_tokens, re.DOTALL)]


# ── Sampling ──────────────────────────────────────────────────────────────────

def load_random_record_with_agents(glob_pattern: str, max_tries: int = 200):
    files = sorted(glob.glob(glob_pattern))
    if not files:
        sys.exit(f"No files found matching: {glob_pattern}")

    for _ in range(max_tries):
        chosen_file = random.choice(files)
        with open(chosen_file, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        if not lines:
            continue
        record = json.loads(random.choice(lines))
        for scene in record.get("scenes", []):
            for act in scene.get("activities", []):
                vt = act.get("video_tokens", "")
                if not vt:
                    continue
                blocks = extract_all_agent_blocks(vt)
                if blocks:
                    return chosen_file, record, act, blocks

    sys.exit(f"Could not find a record with agent blocks after {max_tries} tries.")


# ── Build output dict ─────────────────────────────────────────────────────────

def build_chunk_dict(chunk_idx: int, block: dict, traj: np.ndarray) -> dict:
    cp_abs = block["cp_abs"]
    anchor = block["anchor"]
    scale = block["scale"]
    motion = np.linalg.norm(traj[-1] - traj[0], axis=-1)
    top_joints = sorted(
        [{"joint": H36M_JOINT_NAMES[j], "index": j, "delta_magnitude": round(float(motion[j]), 6)}
         for j in range(N_JOINTS)],
        key=lambda x: x["delta_magnitude"], reverse=True
    )

    return {
        "chunk_index": chunk_idx,
        "time_offset_sec": round(chunk_idx * N_FRAMES_RECONSTRUCT / 30.0, 4),
        "scale_m": round(scale, 6),
        "anchor_m": {
            "shape": [N_JOINTS, N_DIMS],
            "joints": H36M_JOINT_NAMES,
            "data": anchor.tolist(),
        },
        "control_points_absolute": {
            "shape": [4, N_JOINTS, N_DIMS],
            "joints": H36M_JOINT_NAMES,
            "units": "metres, root-centred",
            "data": cp_abs.tolist(),
        },
        "trajectory": {
            "shape": [N_FRAMES_RECONSTRUCT, N_JOINTS, N_DIMS],
            "joints": H36M_JOINT_NAMES,
            "frames_per_second": 30,
            "units": "metres, root-centred",
            "data": traj.tolist(),
        },
        "arm_joints_trajectory": {
            name: {
                "joint_index": idx,
                "frames_m": traj[:, idx, :].tolist(),
            }
            for name, idx in ARM_JOINTS.items()
        },
        "end_effectors": {
            "l_wrist": {
                "joint_index": 13,
                "control_points_m": cp_abs[:, 13, :].tolist(),
                "trajectory_m": traj[:, 13, :].tolist(),
            },
            "r_wrist": {
                "joint_index": 16,
                "control_points_m": cp_abs[:, 16, :].tolist(),
                "trajectory_m": traj[:, 16, :].tolist(),
            },
        },
        "motion_stats": {
            "value_range_m": [round(float(traj.min()), 6), round(float(traj.max()), 6)],
            "joints_by_displacement": top_joints,
        },
    }


def build_output(src_file: str, record: dict, activity: dict, agent_blocks: list) -> dict:
    all_traj = [reconstruct_trajectory(b["cp_abs"]) for b in agent_blocks]
    all_traj_np = np.stack(all_traj)

    chunks = [build_chunk_dict(i, b, traj) for i, (b, traj) in enumerate(zip(agent_blocks, all_traj))]

    return {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "source_file": src_file,
            "decoder_note": (
                "256 tokens per chunk: [scale(1)] + [anchor(51)] + [motion_CPs(204)]. "
                "Absolute metric coordinates (root-centred, metres). "
                "t_cp approximated as evenly-spaced [0, 1/3, 2/3, 1]; "
                "original used arc-length parameterisation."
            ),
            "quantization": {
                "scale":  "token / 255.0 * 2.0  (range [0, 2.0] m)",
                "anchor": "token / 255.0 * 4.0 - 2.0  (range [-2, 2] m per dim)",
                "motion": "token / 127.5 - 1.0  (range [-1, 1])",
            },
            "canonical_bone_lengths_m": {
                "pelvis_to_r_hip": 0.2, "r_hip_to_r_knee": 0.45, "r_knee_to_r_ankle": 0.45,
                "pelvis_to_l_hip": 0.2, "l_hip_to_l_knee": 0.45, "l_knee_to_l_ankle": 0.45,
                "pelvis_to_spine": 0.3, "spine_to_thorax": 0.3, "thorax_to_nose": 0.2,
                "thorax_to_l_shoulder": 0.2, "l_shoulder_to_l_elbow": 0.35, "l_elbow_to_l_wrist": 0.3,
                "thorax_to_r_shoulder": 0.2, "r_shoulder_to_r_elbow": 0.35, "r_elbow_to_r_wrist": 0.3,
            },
        },
        "video": {
            "video_id": record.get("video_id", ""),
            "youtube_title": record.get("metadata", {}).get("youtube_title", ""),
            "category": record.get("metadata", {}).get("category", ""),
            "resolution": record.get("metadata", {}).get("resolution", ""),
            "fps": record.get("metadata", {}).get("fps", 30),
            "duration_sec": record.get("metadata", {}).get("duration_sec", 0),
            "global_context": record.get("global_context", ""),
        },
        "activity": {
            "activity_id": activity.get("activity_id", ""),
            "text_prompt": activity.get("text_prompt", ""),
            "time_range_sec": activity.get("time_range_sec", []),
            "speech_transcript": activity.get("speech_transcript", ""),
            "props_present": activity.get("props_present", []),
            "video_editing": activity.get("video_editing", []),
        },
        "agent_tokens": {
            "n_chunks": len(agent_blocks),
            "total_frames": len(agent_blocks) * N_FRAMES_RECONSTRUCT,
            "total_duration_sec": round(len(agent_blocks) * N_FRAMES_RECONSTRUCT / 30.0, 4),
            "all_trajectories_shape": list(all_traj_np.shape),
            "all_trajectories_value_range": [
                round(float(all_traj_np.min()), 6),
                round(float(all_traj_np.max()), 6),
            ],
            "chunks": chunks,
        },
    }


# ── Pretty print ──────────────────────────────────────────────────────────────

def print_summary(out: dict) -> None:
    sep = "=" * 70
    print(sep)
    print("Agent Token Decoder — absolute metric coordinates (root-centred, metres)")
    print("256 tokens/chunk: [scale(1)] + [anchor(51)] + [motion_CPs(204)]")
    print(sep)

    v = out["video"]
    print(f"\nSource file : {out['meta']['source_file']}")
    print(f"Video ID    : {v['video_id']}")
    print(f"Title       : {v['youtube_title']}")
    print(f"Category    : {v['category']}")
    print(f"Duration    : {v['duration_sec']} sec")

    a = out["activity"]
    print(f"\nActivity    : {a['text_prompt'][:80]}")
    print(f"Time range  : {a['time_range_sec']}")
    if a["speech_transcript"]:
        print(f"Speech      : {a['speech_transcript'][:80]}")
    print(f"Props       : {a['props_present']}")

    ag = out["agent_tokens"]
    print(f"\nAgent chunks: {ag['n_chunks']}  ({ag['total_frames']} frames ≈ {ag['total_duration_sec']}s)")

    for chunk in ag["chunks"][:3]:
        print(f"\n  ── Chunk {chunk['chunk_index']} (t={chunk['time_offset_sec']}s, scale={chunk['scale_m']:.4f}m) ──")
        print(f"  {'joint':<14}" + "".join(f"  cp{i}(m)         " for i in range(4)))
        for name, idx in ARM_JOINTS.items():
            row = f"  {name:<14}"
            for cp_i in range(4):
                xyz = chunk["control_points_absolute"]["data"][cp_i][idx]
                row += f"  [{xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f}]"
            print(row)

        print(f"\n  {'frame':<7}  {'l_wrist (m)':<28}  {'r_wrist (m)':<28}")
        for fi, (lw, rw) in enumerate(zip(
            chunk["end_effectors"]["l_wrist"]["trajectory_m"],
            chunk["end_effectors"]["r_wrist"]["trajectory_m"],
        )):
            print(f"  {fi:<7}  [{lw[0]:+.3f},{lw[1]:+.3f},{lw[2]:+.3f}]          [{rw[0]:+.3f},{rw[1]:+.3f},{rw[2]:+.3f}]")

        print(f"\n  Top-5 joints by displacement (m):")
        for j in chunk["motion_stats"]["joints_by_displacement"][:5]:
            print(f"    {j['joint']:<14}  Δ = {j['delta_magnitude']:.4f} m")

    if ag["n_chunks"] > 3:
        print(f"\n  ... ({ag['n_chunks'] - 3} more chunks in JSON output)")

    print(f"\n{sep}")
    print(f"all_trajectories shape : {ag['all_trajectories_shape']}  (chunks, frames, joints, xyz)")
    print(f"value range (m)        : {ag['all_trajectories_value_range']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="decoded_agent_sample.json",
                        help="Output JSON file path (default: decoded_agent_sample.json)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    src_file, record, activity, agent_blocks = load_random_record_with_agents(DATASET_GLOB)
    out = build_output(src_file, record, activity, agent_blocks)

    print_summary(out)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
