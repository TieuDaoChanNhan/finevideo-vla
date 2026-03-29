#!/usr/bin/env python3
"""
Visualize KEPT spans from V-JEPA filtering results.

This script is the complement of visualize_anomalies.py:
- it reads the original real video
- it reads the original skeleton video
- it reads the JSON result produced by vjepa_filter.py
- it computes the anomaly spans
- then it inverts them to obtain the KEPT spans
- finally it writes a side-by-side highlight reel containing only the kept regions

Output:
- kept_highlight.mp4

Install
-------
pip install -U opencv-python numpy

Example
-------
python visualize_kept_frames.py \
  --video-real videos/example.mp4 \
  --video-skeleton skeletons/example_skeleton.mp4 \
  --result-json outputs/result.json \
  --output outputs/kept_highlight.mp4
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


@dataclass
class Span:
    start_frame: int
    end_frame_inclusive: int
    mean_similarity: float
    clip_indices: List[int]
    num_clips: int


def load_result_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Result JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "clip_scores" not in data or not isinstance(data["clip_scores"], list):
        raise ValueError("Invalid result JSON: missing 'clip_scores' list.")
    return data


def parse_all_clips(data: Dict[str, Any]) -> List[ClipScore]:
    clips: List[ClipScore] = []
    for item in data.get("clip_scores", []):
        try:
            clip = ClipScore(
                clip_index=int(item["clip_index"]),
                start_frame=int(item["start_frame"]),
                end_frame_inclusive=int(item["end_frame_inclusive"]),
                similarity=float(item.get("similarity", 1.0)),
                is_anomaly=bool(item["is_anomaly"]),
            )
        except Exception as e:
            raise ValueError(f"Malformed clip score entry: {item}") from e
        clips.append(clip)

    clips.sort(key=lambda x: (x.start_frame, x.end_frame_inclusive, x.clip_index))
    return clips


def merge_spans(clips: Sequence[ClipScore], anomaly_only: bool) -> List[Span]:
    chosen = [c for c in clips if c.is_anomaly == anomaly_only]
    if not chosen:
        return []

    chosen.sort(key=lambda x: (x.start_frame, x.end_frame_inclusive, x.clip_index))

    merged: List[Span] = []
    cur_start = chosen[0].start_frame
    cur_end = chosen[0].end_frame_inclusive
    cur_sims = [chosen[0].similarity]
    cur_clip_indices = [chosen[0].clip_index]

    for clip in chosen[1:]:
        if clip.start_frame <= cur_end + 1:
            cur_end = max(cur_end, clip.end_frame_inclusive)
            cur_sims.append(clip.similarity)
            cur_clip_indices.append(clip.clip_index)
        else:
            merged.append(
                Span(
                    start_frame=cur_start,
                    end_frame_inclusive=cur_end,
                    mean_similarity=float(sum(cur_sims) / len(cur_sims)),
                    clip_indices=list(cur_clip_indices),
                    num_clips=len(cur_clip_indices),
                )
            )
            cur_start = clip.start_frame
            cur_end = clip.end_frame_inclusive
            cur_sims = [clip.similarity]
            cur_clip_indices = [clip.clip_index]

    merged.append(
        Span(
            start_frame=cur_start,
            end_frame_inclusive=cur_end,
            mean_similarity=float(sum(cur_sims) / len(cur_sims)),
            clip_indices=list(cur_clip_indices),
            num_clips=len(cur_clip_indices),
        )
    )
    return merged


def invert_anomaly_spans(
    anomaly_spans: Sequence[Span],
    valid_start: int,
    valid_end: int,
) -> List[Tuple[int, int]]:
    if valid_end < valid_start:
        return []

    if not anomaly_spans:
        return [(valid_start, valid_end)]

    kept: List[Tuple[int, int]] = []
    cursor = valid_start

    for span in anomaly_spans:
        a = max(valid_start, span.start_frame)
        b = min(valid_end, span.end_frame_inclusive)
        if b < valid_start or a > valid_end:
            continue

        if cursor < a:
            kept.append((cursor, a - 1))
        cursor = max(cursor, b + 1)

    if cursor <= valid_end:
        kept.append((cursor, valid_end))

    return [(s, e) for s, e in kept if s <= e]


def compute_kept_spans_from_clips(clips: Sequence[ClipScore]) -> List[Span]:
    if not clips:
        return []

    valid_start = min(c.start_frame for c in clips)
    valid_end = max(c.end_frame_inclusive for c in clips)

    anomaly_spans = merge_spans(clips, anomaly_only=True)
    kept_ranges = invert_anomaly_spans(anomaly_spans, valid_start, valid_end)

    kept_spans: List[Span] = []
    for start, end in kept_ranges:
        overlapped = [
            c for c in clips
            if (not c.is_anomaly) and not (c.end_frame_inclusive < start or c.start_frame > end)
        ]
        if overlapped:
            mean_sim = float(sum(c.similarity for c in overlapped) / len(overlapped))
            clip_indices = [c.clip_index for c in overlapped]
        else:
            mean_sim = 0.0
            clip_indices = []
        kept_spans.append(
            Span(
                start_frame=start,
                end_frame_inclusive=end,
                mean_similarity=mean_sim,
                clip_indices=clip_indices,
                num_clips=len(clip_indices),
            )
        )
    return kept_spans


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


def open_writer(output_path: str, frame_size: Tuple[int, int], fps: float) -> cv2.VideoWriter:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    width, height = frame_size

    for codec in ("mp4v", "avc1"):
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            return writer

    raise RuntimeError(f"Could not create output writer for: {output_path}")


def render_kept_highlight(
    video_real: str,
    video_skeleton: str,
    spans: Sequence[Span],
    output_path: str,
    output_fps: Optional[float] = None,
    gap_frames: int = 8,
) -> None:
    if not spans:
        raise RuntimeError("No kept spans found. Nothing to render.")

    real_reader = VideoReaderCV2(video_real)
    skel_reader = VideoReaderCV2(video_skeleton)

    try:
        max_valid_frame = min(real_reader.frame_count, skel_reader.frame_count) - 1
        if max_valid_frame < 0:
            raise RuntimeError("No readable frames available.")

        probe_idx = min(spans[0].start_frame, max_valid_frame)
        probe_real = real_reader.read_frame(probe_idx)
        probe_skel = skel_reader.read_frame(probe_idx)
        probe_canvas = make_side_by_side(probe_real, probe_skel)
        out_h, out_w = probe_canvas.shape[:2]

        fps = output_fps if output_fps is not None else min(real_reader.fps, skel_reader.fps)
        if fps <= 0:
            fps = 30.0

        writer = open_writer(output_path, (out_w, out_h), fps)
        try:
            for span_idx, span in enumerate(spans):
                start = max(0, span.start_frame)
                end = min(max_valid_frame, span.end_frame_inclusive)

                for frame_idx in range(start, end + 1):
                    real_frame = real_reader.read_frame(frame_idx)
                    skel_frame = skel_reader.read_frame(frame_idx)
                    canvas = make_side_by_side(real_frame, skel_frame)
                    canvas = draw_text_block(
                        canvas,
                        [
                            f"Frame index: {frame_idx}",
                            f"Kept span: {start}-{end}",
                            f"Mean cosine similarity: {span.mean_similarity:.4f}",
                            f"Covered normal clips: {span.num_clips}",
                        ],
                    )
                    writer.write(canvas)

                if gap_frames > 0 and span_idx < len(spans) - 1:
                    gap = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                    gap = draw_text_block(
                        gap,
                        [f"Next kept span: {spans[span_idx + 1].start_frame}-{spans[span_idx + 1].end_frame_inclusive}"],
                    )
                    for _ in range(gap_frames):
                        writer.write(gap)
        finally:
            writer.release()
    finally:
        real_reader.close()
        skel_reader.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize KEPT spans from a V-JEPA result JSON.")
    parser.add_argument("--video-real", type=str, required=True, help="Path to the original real-person video.")
    parser.add_argument("--video-skeleton", type=str, required=True, help="Path to the original skeleton video.")
    parser.add_argument("--result-json", type=str, required=True, help="Path to the JSON output from vjepa_filter.py.")
    parser.add_argument("--output", type=str, default="kept_highlight.mp4", help="Output MP4 filename.")
    parser.add_argument("--output-fps", type=float, default=None, help="Optional output FPS override.")
    parser.add_argument("--gap-frames", type=int, default=8, help="Black separator frames between kept spans.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    result_data = load_result_json(args.result_json)
    clips = parse_all_clips(result_data)
    anomaly_spans = merge_spans(clips, anomaly_only=True)
    kept_spans = compute_kept_spans_from_clips(clips)

    print("=" * 88)
    print("Kept-Span Visualization")
    print("=" * 88)
    print(f"real video       : {args.video_real}")
    print(f"skeleton video   : {args.video_skeleton}")
    print(f"result json      : {args.result_json}")
    print(f"output video     : {args.output}")
    print(f"total clips      : {len(clips)}")
    print(f"anomaly spans    : {len(anomaly_spans)}")
    print(f"kept spans       : {len(kept_spans)}")
    for i, span in enumerate(kept_spans):
        print(
            f"[{i:03d}] frames {span.start_frame}-{span.end_frame_inclusive} | "
            f"mean_sim={span.mean_similarity:.4f} | normal_clips={span.num_clips}"
        )

    render_kept_highlight(
        video_real=args.video_real,
        video_skeleton=args.video_skeleton,
        spans=kept_spans,
        output_path=args.output,
        output_fps=args.output_fps,
        gap_frames=args.gap_frames,
    )

    print(f"\nSaved kept highlight reel to: {args.output}")


if __name__ == "__main__":
    main()
