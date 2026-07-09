#!/usr/bin/env python3
"""
Side-by-side skeleton visualization: original video | skeleton overlay.

Reads:
  - Original video from videos_staging/{video_id}.mp4
  - 3D pose states from yolo_cleaned/{video_id}_cleaned.jsonl
    (Phase 4 output — float xyz before tokenization, shape per window: (8, 17, 3))

Output: MP4 with left = raw video, right = video + skeleton overlay.

Uses imageio + PIL (available in env_tools). No cv2 needed.

Usage:
    python tools/visualize/visualize_skeleton_sidebyside.py --list-available
    python tools/visualize/visualize_skeleton_sidebyside.py --video-id 001bwvuSYyA
    python tools/visualize/visualize_skeleton_sidebyside.py --video-id 001bwvuSYyA --flip-y
    python tools/visualize/visualize_skeleton_sidebyside.py --video-id 001bwvuSYyA --max-frames 300
"""

import argparse
import json
import os
import random
import sys

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Paths ──────────────────────────────────────────────────────────────────────
VIDEO_DIR   = "/p/data1/mmlaion/shared/nguyen38/data/videos_staging"
STATES_DIR  = "/p/data1/mmlaion/shared/nguyen38/data/outputs/yolo_cleaned"
OUTPUT_DIR  = "/p/data1/mmlaion/nguyen38/3d-human-pose/logs"

# ── H36M 17-joint skeleton ─────────────────────────────────────────────────────
SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3),       # pelvis → r_hip → r_knee → r_ankle
    (0, 4), (4, 5), (5, 6),       # pelvis → l_hip → l_knee → l_ankle
    (0, 7), (7, 8),                # pelvis → spine → thorax
    (8, 9), (9, 10),               # thorax → nose → head_top
    (8, 11), (11, 12), (12, 13),   # thorax → l_shoulder → l_elbow → l_wrist
    (8, 14), (14, 15), (15, 16),   # thorax → r_shoulder → r_elbow → r_wrist
]

JOINT_NAMES = [
    "pelvis", "r_hip", "r_knee", "r_ankle",
    "l_hip",  "l_knee", "l_ankle",
    "spine",  "thorax", "nose", "head_top",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_shoulder", "r_elbow", "r_wrist",
]

# RGB colours per edge group
EDGE_COLOURS = [
    (220, 50,  50),   # (0,1)  right leg
    (220, 50,  50),   # (1,2)
    (220, 50,  50),   # (2,3)
    (50,  80, 220),   # (0,4)  left leg
    (50,  80, 220),   # (4,5)
    (50,  80, 220),   # (5,6)
    (180, 180,  0),   # (0,7)  spine
    (180, 180,  0),   # (7,8)
    (180, 180,  0),   # (8,9)
    (180, 180,  0),   # (9,10)
    (50,  200, 200),  # (8,11) left arm
    (50,  200, 200),  # (11,12)
    (50,  200, 200),  # (12,13)
    (50,  220, 100),  # (8,14) right arm
    (50,  220, 100),  # (14,15)
    (50,  220, 100),  # (15,16)
]


def load_states(states_path: str, max_frame: int) -> np.ndarray:
    """Load yolo_cleaned JSONL → (max_frame, 17, 3) float array, NaN where missing."""
    pose = np.full((max_frame, 17, 3), np.nan, dtype=np.float32)
    with open(states_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            win_id = int(rec["window_id"])
            states = np.array(rec["states"], dtype=np.float32)  # (8, 17, 3)
            for fi in range(states.shape[0]):
                frame_idx = win_id + fi
                if frame_idx >= max_frame:
                    break
                if np.isnan(pose[frame_idx, 0, 0]):
                    pose[frame_idx] = states[fi]
    return pose


def project_to_2d(pose_3d: np.ndarray, width: int, height: int,
                  flip_y: bool = False) -> np.ndarray:
    """
    Project (N, 17, 3) → (N, 17, 2) pixel coords using XY front-view.

    Scale computed globally so skeleton size is stable across frames.
    flip_y=True negates Y before projecting (fixes upside-down when
    MotionBERT outputs in camera space where Y points downward).
    """
    pose = pose_3d[:, :, :2].copy()  # take X, Y only
    if flip_y:
        pose[:, :, 1] *= -1

    finite = np.all(np.isfinite(pose_3d), axis=-1)  # (N, 17)

    valid_y = pose[:, :, 1][finite]
    if valid_y.size == 0:
        return np.zeros((pose.shape[0], 17, 2), dtype=np.int32)

    y_range = max(valid_y.max() - valid_y.min(), 1e-6)
    scale = (height * 0.80) / y_range

    valid_xy = pose[finite]
    center = (valid_xy.min(axis=0) + valid_xy.max(axis=0)) / 2

    xy = (pose - center) * scale
    px = np.clip(np.rint(xy[:, :, 0] + width  / 2), 0, width  - 1).astype(np.int32)
    py = np.clip(np.rint(xy[:, :, 1] + height / 2), 0, height - 1).astype(np.int32)
    return np.stack([px, py], axis=2)


def draw_skeleton(img: Image.Image, joints_2d: np.ndarray,
                  finite_mask: np.ndarray) -> Image.Image:
    """Draw skeleton edges + joint dots onto a PIL Image. Returns a new image."""
    out = img.copy()
    draw = ImageDraw.Draw(out)

    for ei, (a, b) in enumerate(SKELETON_EDGES):
        if not (finite_mask[a] and finite_mask[b]):
            continue
        xa, ya = int(joints_2d[a, 0]), int(joints_2d[a, 1])
        xb, yb = int(joints_2d[b, 0]), int(joints_2d[b, 1])
        draw.line([(xa, ya), (xb, yb)], fill=EDGE_COLOURS[ei], width=2)

    for ji in range(17):
        if not finite_mask[ji]:
            continue
        x, y = int(joints_2d[ji, 0]), int(joints_2d[ji, 1])
        r = 3
        draw.ellipse([(x - r, y - r), (x + r, y + r)],
                     fill=(255, 255, 255), outline=(80, 80, 80))
    return out


def add_label(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.text((10, 8), text, fill=(0, 0, 0))
    draw.text((9, 7), text, fill=(255, 255, 255))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", help="YouTube video ID, e.g. 001bwvuSYyA")
    parser.add_argument(
        "--flip-y", action="store_true",
        help="Negate Y axis before projecting — try if skeleton appears upside-down",
    )
    parser.add_argument(
        "--max-frames", type=int, default=600,
        help="Max frames to render (default: 600 = 20s at 30fps). 0 = full video.",
    )
    parser.add_argument("--list-available", action="store_true",
                        help="Print 10 random IDs that have both video + states")
    parser.add_argument("--video-dir",  default=VIDEO_DIR)
    parser.add_argument("--states-dir", default=STATES_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.list_available:
        state_ids = {
            f.replace("_cleaned.jsonl", "")
            for f in os.listdir(args.states_dir) if f.endswith("_cleaned.jsonl")
        }
        video_ids = {
            f.replace(".mp4", "")
            for f in os.listdir(args.video_dir) if f.endswith(".mp4")
        }
        common = sorted(state_ids & video_ids)
        sample = random.sample(common, min(10, len(common)))
        print(f"{len(common)} videos have both .mp4 + _cleaned.jsonl. Sample:")
        for v in sample:
            print(f"  {v}")
        return

    if not args.video_id:
        parser.error("Provide --video-id or --list-available")

    vid = args.video_id
    video_path  = os.path.join(args.video_dir,  f"{vid}.mp4")
    states_path = os.path.join(args.states_dir, f"{vid}_cleaned.jsonl")
    suffix = "_flipy" if args.flip_y else ""
    out_path = os.path.join(args.output_dir, f"skeleton_{vid}{suffix}.mp4")

    for p, label in [(video_path, "video"), (states_path, "states")]:
        if not os.path.exists(p):
            print(f"ERROR: {label} not found: {p}")
            sys.exit(1)

    # ── Read video metadata ──────────────────────────────────────────────────
    reader = imageio.get_reader(video_path, format="ffmpeg")
    meta   = reader.get_meta_data()
    fps    = meta.get("fps", 30.0)
    size   = meta.get("size", (1280, 720))
    width, height = size
    # Count frames cheaply
    nframes_meta = meta.get("nframes")
    if nframes_meta and np.isfinite(nframes_meta):
        total_frames = int(nframes_meta)
    else:
        # ffmpeg didn't report frame count — count by iterating (slow for long videos)
        total_frames = sum(1 for _ in reader)
    reader.close()

    render_n = total_frames if args.max_frames == 0 else min(args.max_frames, total_frames)
    print(f"Video : {vid}  {width}×{height}  {fps:.1f}fps  {total_frames} frames")
    print(f"Render: {render_n} frames  |  flip_y={args.flip_y}")

    # ── Load pose ────────────────────────────────────────────────────────────
    print("Loading pose states...")
    pose_3d = load_states(states_path, total_frames)
    valid_n = int(np.sum(~np.isnan(pose_3d[:, 0, 0])))
    print(f"Pose  : {valid_n}/{total_frames} frames ({valid_n/total_frames*100:.1f}%)")

    # Print sample pelvis values to check coordinate system
    sample_idx = [i for i in range(min(5, total_frames)) if not np.isnan(pose_3d[i, 0, 0])]
    if sample_idx:
        print("Sample pelvis xyz (frame 0 of first windows, raw float before tokenization):")
        for i in sample_idx[:3]:
            p = pose_3d[i, 0]
            print(f"  frame {i:4d}: pelvis = ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m")

    # ── Project ──────────────────────────────────────────────────────────────
    pose_2d = project_to_2d(pose_3d, width, height, flip_y=args.flip_y)
    finite_masks = np.all(np.isfinite(pose_3d), axis=-1)  # (N, 17)

    # ── Render ───────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    writer = imageio.get_writer(out_path, format="ffmpeg", fps=fps,
                                codec="libx264", quality=7)

    reader = imageio.get_reader(video_path, format="ffmpeg")
    label_flip = " [flip_y]" if args.flip_y else ""

    print("Rendering...")
    for fi, frame_np in enumerate(reader):
        if fi >= render_n:
            break

        img = Image.fromarray(frame_np)
        left  = add_label(img, "Original")

        if finite_masks[fi].any():
            right = draw_skeleton(img, pose_2d[fi], finite_masks[fi])
        else:
            right = img.copy()
        right = add_label(right, f"Skeleton{label_flip}")

        combined = Image.new("RGB", (width * 2, height))
        combined.paste(left,  (0,     0))
        combined.paste(right, (width, 0))

        writer.append_data(np.array(combined))

        if fi % 150 == 0:
            print(f"  frame {fi}/{render_n}")

    reader.close()
    writer.close()

    print(f"\nSaved: {out_path}")
    print("If skeleton is upside-down → re-run with --flip-y")
    print("If skeleton is mirrored    → left/right labels may be swapped (camera mirror)")


if __name__ == "__main__":
    main()
