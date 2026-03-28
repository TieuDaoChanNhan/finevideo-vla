#!/usr/bin/env python3
"""
Visualize Top-K highest-similarity clips from a V-JEPA result JSON.

This script:
1) reads a result JSON produced by vjepa_filter.py
2) sorts clip_scores by similarity in descending order
3) selects the top-K clips
4) extracts the corresponding frames from the real and skeleton videos
5) renders a single side-by-side MP4 highlight reel

Output:
- a single MP4 file such as top_10_highlights.mp4

Install
-------
pip install -U opencv-python numpy

Example
-------
python visualize_top_clips.py \
  --video-real videos/example.mp4 \
  --video-skeleton skeletons/example_skeleton.mp4 \
  --result-json outputs/result.json \
  --top-k 10 \
  --output outputs/top_10_highlights.mp4
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class ClipScore:
    clip_index: int
    start_frame: int
    end_frame_inclusive: int
    similarity: float
    is_anomaly: bool


def load_result_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Result JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "clip_scores" not in data or not isinstance(data["clip_scores"], list):
        raise ValueError("Invalid result JSON: missing 'clip_scores' list.")
    return data


def parse_clip_scores(data: Dict[str, Any]) -> List[ClipScore]:
    clips: List[ClipScore] = []
    for item in data.get("clip_scores", []):
        try:
            clip = ClipScore(
                clip_index=int(item["clip_index"]),
                start_frame=int(item["start_frame"]),
                end_frame_inclusive=int(item["end_frame_inclusive"]),
                similarity=float(item["similarity"]),
                is_anomaly=bool(item["is_anomaly"]),
            )
        except Exception as e:
            raise ValueError(f"Malformed clip score entry: {item}") from e
        clips.append(clip)

    clips.sort(key=lambda x: (-x.similarity, x.start_frame, x.clip_index))
    return clips


def select_top_k(clips: Sequence[ClipScore], top_k: int) -> List[ClipScore]:
    if top_k <= 0:
        raise ValueError("top_k must be >= 1")
    return list(clips[:top_k])


class VideoReaderCV2:
    def __init__(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video not found: {path}")
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0:
            self.fps = 30.0
        self._next_expected: Optional[int] = None

    def read_frame(self, frame_index: int) -> np.ndarray:
        if frame_index < 0 or frame_index >= self.frame_count:
            raise IndexError(
                f"Frame index {frame_index} out of range for {self.path}; "
                f"valid range: [0, {self.frame_count - 1}]"
            )

        if self._next_expected is None or frame_index != self._next_expected:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_index} from {self.path}")

        self._next_expected = frame_index + 1
        return frame

    def close(self) -> None:
        self.cap.release()


def resize_to_height(frame: np.ndarray, target_height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h == target_height:
        return frame
    scale = target_height / max(1, h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_LINEAR)


def make_side_by_side(real_frame: np.ndarray, skeleton_frame: np.ndarray) -> np.ndarray:
    target_height = max(real_frame.shape[0], skeleton_frame.shape[0])
    real_resized = resize_to_height(real_frame, target_height)
    skel_resized = resize_to_height(skeleton_frame, target_height)

    separator = np.zeros((target_height, 8, 3), dtype=np.uint8)
    canvas = np.concatenate([real_resized, separator, skel_resized], axis=1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "REAL", (12, 30), font, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    offset_x = real_resized.shape[1] + separator.shape[1] + 12
    cv2.putText(canvas, "SKELETON", (offset_x, 30), font, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def draw_text_block(
    frame: np.ndarray,
    lines: Sequence[str],
    x: int = 14,
    y: int = 30,
    line_gap: int = 28,
) -> np.ndarray:
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    widths = []
    heights = []
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
        widths.append(tw)
        heights.append(th)

    box_w = max(widths) + 24 if widths else 120
    box_h = sum(heights) + line_gap * max(0, len(lines) - 1) + 20
    x1, y1 = x - 8, y - 22
    x2, y2 = x1 + box_w, y1 + box_h

    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)

    cur_y = y
    for line in lines:
        cv2.putText(out, line, (x, cur_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cur_y += line_gap

    return out


def open_writer(output_path: str, frame_size: Tuple[int, int], fps: float) -> cv2.VideoWriter:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    width, height = frame_size
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create mp4v writer for: {output_path}")
    return writer


def render_top_k_highlight(
    video_real: str,
    video_skeleton: str,
    top_clips: Sequence[ClipScore],
    output_path: str,
    separator_frames: int = 12,
    output_fps: Optional[float] = None,
) -> None:
    if not top_clips:
        raise RuntimeError("No clips available to render.")

    real_reader = VideoReaderCV2(video_real)
    skel_reader = VideoReaderCV2(video_skeleton)

    try:
        max_valid_frame = min(real_reader.frame_count, skel_reader.frame_count) - 1
        if max_valid_frame < 0:
            raise RuntimeError("No readable frames available.")

        probe_idx = min(max(0, top_clips[0].start_frame), max_valid_frame)
        probe_real = real_reader.read_frame(probe_idx)
        probe_skel = skel_reader.read_frame(probe_idx)
        probe_canvas = make_side_by_side(probe_real, probe_skel)
        out_h, out_w = probe_canvas.shape[:2]

        fps = output_fps if output_fps is not None else min(real_reader.fps, skel_reader.fps)
        if fps <= 0:
            fps = 30.0

        writer = open_writer(output_path, (out_w, out_h), fps)
        try:
            num_clips = len(top_clips)

            for rank_idx, clip in enumerate(top_clips, start=1):
                start = max(0, clip.start_frame)
                end = min(max_valid_frame, clip.end_frame_inclusive)

                for frame_idx in range(start, end + 1):
                    real_frame = real_reader.read_frame(frame_idx)
                    skel_frame = skel_reader.read_frame(frame_idx)
                    canvas = make_side_by_side(real_frame, skel_frame)

                    overlay_lines = [
                        f"Rank #{rank_idx}/{num_clips}",
                        f"Score: {clip.similarity:.4f}",
                        f"Frames: {start}-{end}",
                    ]
                    canvas = draw_text_block(canvas, overlay_lines)
                    writer.write(canvas)

                if separator_frames > 0 and rank_idx < num_clips:
                    blank = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                    blank = draw_text_block(
                        blank,
                        [
                            "NEXT RANK",
                            f"Upcoming: Rank #{rank_idx + 1}/{num_clips}",
                        ],
                        x=out_w // 2 - 120,
                        y=out_h // 2,
                        line_gap=36,
                    )
                    for _ in range(separator_frames):
                        writer.write(blank)
        finally:
            writer.release()
    finally:
        real_reader.close()
        skel_reader.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize Top-K highest-similarity clips from a V-JEPA result JSON.")
    parser.add_argument("--video-real", type=str, required=True, help="Path to the real-person video.")
    parser.add_argument("--video-skeleton", type=str, required=True, help="Path to the skeleton video.")
    parser.add_argument("--result-json", type=str, required=True, help="Path to the JSON result from vjepa_filter.py.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top-scoring clips to visualize.")
    parser.add_argument("--output", type=str, default="top_10_highlights.mp4", help="Output MP4 filename.")
    parser.add_argument("--separator-frames", type=int, default=12, help="Blank separator frames between ranked clips.")
    parser.add_argument("--output-fps", type=float, default=None, help="Optional output FPS override.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    result_data = load_result_json(args.result_json)
    clips = parse_clip_scores(result_data)
    top_clips = select_top_k(clips, args.top_k)

    print("=" * 88)
    print("Top-K Clip Visualization")
    print("=" * 88)
    print(f"real video       : {args.video_real}")
    print(f"skeleton video   : {args.video_skeleton}")
    print(f"result json      : {args.result_json}")
    print(f"top_k            : {args.top_k}")
    print(f"output video     : {args.output}")
    print(f"clips available  : {len(clips)}")
    print(f"clips selected   : {len(top_clips)}")
    for rank_idx, clip in enumerate(top_clips, start=1):
        print(
            f"[#{rank_idx:02d}] clip_index={clip.clip_index} | "
            f"frames {clip.start_frame}-{clip.end_frame_inclusive} | "
            f"similarity={clip.similarity:.4f} | anomaly={clip.is_anomaly}"
        )

    render_top_k_highlight(
        video_real=args.video_real,
        video_skeleton=args.video_skeleton,
        top_clips=top_clips,
        output_path=args.output,
        separator_frames=args.separator_frames,
        output_fps=args.output_fps,
    )

    print(f"\nSaved Top-K highlight reel to: {args.output}")


if __name__ == "__main__":
    main()
