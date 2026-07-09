import argparse
import json
import os

import cv2
import numpy as np


SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]



def get_video_info(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if frame_count <= 0 or width <= 0 or height <= 0 or fps <= 0:
        raise RuntimeError(
            f"Invalid video metadata: frame_count={frame_count}, width={width}, height={height}, fps={fps}"
        )

    return frame_count, width, height, fps


def load_jsonl_states(jsonl_path: str, frame_count: int):
    pose_3d = np.full((frame_count, 17, 3), np.nan, dtype=np.float32)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_num}: {exc}") from exc

            if "window_id" not in record or "states" not in record:
                continue

            start_idx = int(record["window_id"])
            states = np.asarray(record["states"], dtype=np.float32)

            if states.shape != (8, 17, 3):
                continue

            end_idx = min(start_idx + states.shape[0], frame_count)
            valid_len = end_idx - start_idx
            if valid_len <= 0:
                continue

            chunk = states[:valid_len]
            existing = pose_3d[start_idx:end_idx]

            existing_nan = np.isnan(existing).all(axis=(1, 2))
            existing[existing_nan] = chunk[existing_nan]
            pose_3d[start_idx:end_idx] = existing

    return pose_3d


def compute_global_xy_projection(pose_3d, width, height):
    """Project 3D metric coordinates to 2D pixel space, fitting to frame height and centring."""

    pose = np.asarray(pose_3d, dtype=np.float64)
    finite_joint_mask = np.all(np.isfinite(pose), axis=-1)

    # 1. Get globally valid Y (vertical axis) coordinates to compute scale
    valid_y = pose[:, :, 1][finite_joint_mask]
    if valid_y.size == 0:
        return np.zeros((pose.shape[0], pose.shape[1], 2), dtype=np.float64)

    y_min_global = np.nanmin(valid_y)
    y_max_global = np.nanmax(valid_y)
    y_range_global = y_max_global - y_min_global + 1e-6

    # 2. SCALE: fit skeleton height to 90% of frame height
    scale = (height * 0.9) / y_range_global

    # 3. GLOBAL CENTRE (bring skeleton to origin)
    xy = pose[:, :, :2].copy()
    valid_xy = xy[finite_joint_mask]
    xy_min_global = np.nanmin(valid_xy, axis=0)
    xy_max_global = np.nanmax(valid_xy, axis=0)
    origin_center_global = (xy_min_global + xy_max_global) / 2

    # 4. TRANSFORM: centre → scale → shift to frame centre
    xy_centered = xy - origin_center_global[None, None, :]
    xy_pix = xy_centered * scale

    x_pix = xy_pix[:, :, 0] + width / 2
    y_pix = xy_pix[:, :, 1] + height / 2

    # Round and clip
    x_pix = np.clip(np.rint(x_pix), 0, width - 1).astype(np.int32)
    y_pix = np.clip(np.rint(y_pix), 0, height - 1).astype(np.int32)

    pose_2d = np.stack([x_pix, y_pix], axis=2)  # (N, 17, 2)
    return pose_2d


def draw_skeleton(frame: np.ndarray, joints_2d: np.ndarray, color=(0, 255, 0), thickness: int = 2):
    for a, b in SKELETON_EDGES:
        xa, ya = joints_2d[a]
        xb, yb = joints_2d[b]
        if xa < 0 or ya < 0 or xb < 0 or yb < 0:
            continue
        cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), color, thickness, lineType=cv2.LINE_AA)

    for x, y in joints_2d:
        if x < 0 or y < 0:
            continue
        cv2.circle(frame, (int(x), int(y)), 3, color, -1, lineType=cv2.LINE_AA)


def render_video(pose_2d: np.ndarray, output_mp4: str, fps: float, width: int, height: int):
    os.makedirs(os.path.dirname(output_mp4) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_mp4, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for file: {output_mp4}")

    try:
        for frame_idx in range(pose_2d.shape[0]):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            draw_skeleton(frame, pose_2d[frame_idx], color=(0, 255, 0), thickness=2)
            writer.write(frame)
    finally:
        writer.release()


def main():
    parser = argparse.ArgumentParser(description="Render filtered skeleton video from JSONL windows.")
    parser.add_argument("--video-real", required=True, help="Path to the real/original video for FPS and resolution.")
    parser.add_argument("--jsonl", required=True, help="Path to the filtered JSONL file.")
    parser.add_argument("--output", required=True, help="Output MP4 video path.")
    args = parser.parse_args()

    frame_count, width, height, fps = get_video_info(args.video_real)
    pose_3d = load_jsonl_states(args.jsonl, frame_count)
    pose_2d = compute_global_xy_projection(pose_3d, width, height)
    render_video(pose_2d, args.output, fps, width, height)

    print(f"Saved rendered skeleton video to: {args.output}")


if __name__ == "__main__":
    main()
