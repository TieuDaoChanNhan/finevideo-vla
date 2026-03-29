#!/usr/bin/env python3
"""
yolo_filter.py

Filter video clips/windows that contain mostly empty-background frames
(no person detected) using Ultralytics YOLO26 Nano.

Rule:
- Sample `sampled_frames` frames per window with `temporal_stride`
- If >= 8 sampled frames in a window have no person detection
  with confidence >= threshold, mark the clip as anomaly

Example:
    python yolo_filter.py \
        --video-real input.mp4 \
        --threshold 0.75 \
        --sampled-frames 16 \
        --temporal-stride 4 \
        --window-step 16 \
        --output-json result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO


PERSON_CLASS_ID = 0
DEFAULT_MODEL = "yolo26n.pt"


@dataclass(frozen=True)
class WindowPlan:
    clip_index: int
    start_frame: int
    end_frame_inclusive: int
    sample_indices: Tuple[int, ...]


@dataclass(frozen=True)
class ClipScore:
    clip_index: int
    start_frame: int
    end_frame_inclusive: int
    is_anomaly: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter empty-background video windows using Ultralytics YOLO26 Nano."
    )
    parser.add_argument(
        "--video-real",
        type=Path,
        required=True,
        help="Path to input MP4 video.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Confidence threshold for person detection. Default: 0.75",
    )
    parser.add_argument(
        "--sampled-frames",
        type=int,
        default=16,
        help="Number of sampled frames per window. Default: 16",
    )
    parser.add_argument(
        "--temporal-stride",
        type=int,
        default=4,
        help="Stride between sampled frames inside a window. Default: 4",
    )
    parser.add_argument(
        "--window-step",
        type=int,
        default=16,
        help="Sliding window step in frames. Default: 16",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path to save output JSON.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Ultralytics model path/name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size. Default: 640",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.video_real.exists():
        raise FileNotFoundError(f"Video not found: {args.video_real}")
    if args.threshold < 0.0 or args.threshold > 1.0:
        raise ValueError("--threshold must be in [0, 1].")
    if args.sampled_frames <= 0:
        raise ValueError("--sampled-frames must be > 0.")
    if args.temporal_stride <= 0:
        raise ValueError("--temporal-stride must be > 0.")
    if args.window_step <= 0:
        raise ValueError("--window-step must be > 0.")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be > 0.")


def get_video_metadata(video_path: Path) -> Tuple[int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()

    if total_frames <= 0:
        raise RuntimeError(f"Could not determine frame count for video: {video_path}")

    return total_frames, fps


def build_window_plans(
    total_frames: int,
    sampled_frames: int,
    temporal_stride: int,
    window_step: int,
) -> List[WindowPlan]:
    """
    Build valid sliding windows.

    A window contains frames:
        start, start + stride, ..., start + (sampled_frames - 1) * stride
    """
    temporal_span = (sampled_frames - 1) * temporal_stride + 1
    last_valid_start = total_frames - temporal_span
    if last_valid_start < 0:
        return []

    plans: List[WindowPlan] = []
    clip_index = 0

    for start_frame in range(0, last_valid_start + 1, window_step):
        sample_indices = tuple(
            start_frame + i * temporal_stride for i in range(sampled_frames)
        )
        plans.append(
            WindowPlan(
                clip_index=clip_index,
                start_frame=start_frame,
                end_frame_inclusive=sample_indices[-1],
                sample_indices=sample_indices,
            )
        )
        clip_index += 1

    return plans


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def load_model(model_name: str, device: str) -> YOLO:
    model = YOLO(model_name)
    model.to(device)
    return model


def has_person_detection(result, threshold: float) -> bool:
    """
    Return True if the frame has at least one detection:
      - class == PERSON_CLASS_ID
      - confidence >= threshold
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return False

    cls_tensor = boxes.cls
    conf_tensor = boxes.conf

    if cls_tensor is None or conf_tensor is None:
        return False

    person_mask = (cls_tensor == PERSON_CLASS_ID) & (conf_tensor >= threshold)
    return bool(person_mask.any().item())


def infer_window_is_anomaly(
    model: YOLO,
    frames_bgr: Sequence,
    threshold: float,
    device: str,
    imgsz: int,
) -> bool:
    """
    Run batched inference on all sampled frames in the window.
    A window is anomaly if >= 8 frames are empty.
    """
    use_half = device.startswith("cuda")

    results = model.predict(
        source=list(frames_bgr),
        conf=threshold,
        device=device,
        half=use_half,
        imgsz=imgsz,
        verbose=False,
        stream=False,
    )

    empty_count = 0
    for result in results:
        if not has_person_detection(result, threshold=threshold):
            empty_count += 1

    return empty_count >= 8


def process_video(
    video_path: Path,
    model: YOLO,
    threshold: float,
    sampled_frames: int,
    temporal_stride: int,
    window_step: int,
    imgsz: int,
) -> Tuple[List[ClipScore], int]:
    """
    Read the video sequentially once, collect frames for active windows,
    and run batched YOLO inference per completed window.
    """
    total_frames, _ = get_video_metadata(video_path)
    plans = build_window_plans(
        total_frames=total_frames,
        sampled_frames=sampled_frames,
        temporal_stride=temporal_stride,
        window_step=window_step,
    )

    if not plans:
        return [], total_frames

    plans_by_start: Dict[int, List[WindowPlan]] = {}
    plans_by_end: Dict[int, List[WindowPlan]] = {}
    frame_requests: Dict[int, List[Tuple[int, int]]] = {}

    for plan in plans:
        plans_by_start.setdefault(plan.start_frame, []).append(plan)
        plans_by_end.setdefault(plan.end_frame_inclusive, []).append(plan)

        for slot_idx, frame_idx in enumerate(plan.sample_indices):
            frame_requests.setdefault(frame_idx, []).append((plan.clip_index, slot_idx))

    active_buffers: Dict[int, List[Optional]] = {}
    plan_lookup: Dict[int, WindowPlan] = {plan.clip_index: plan for plan in plans}
    results: List[ClipScore] = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    device = choose_device()

    pbar = tqdm(total=total_frames, desc="Scanning video", unit="frame")
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Activate windows starting here
            for plan in plans_by_start.get(frame_idx, []):
                active_buffers[plan.clip_index] = [None] * len(plan.sample_indices)

            # Store sampled frames needed by active windows
            for clip_index, slot_idx in frame_requests.get(frame_idx, []):
                if clip_index in active_buffers:
                    active_buffers[clip_index][slot_idx] = frame.copy()

            # Finalize windows ending here
            ending_plans = plans_by_end.get(frame_idx, [])
            for plan in ending_plans:
                buffer = active_buffers.pop(plan.clip_index, None)
                if buffer is None:
                    raise RuntimeError(
                        f"Internal error: missing active buffer for clip {plan.clip_index}"
                    )
                if any(item is None for item in buffer):
                    raise RuntimeError(
                        f"Internal error: incomplete sampled frames for clip {plan.clip_index}"
                    )

                frames_bgr = buffer  # type: ignore[assignment]
                is_anomaly = infer_window_is_anomaly(
                    model=model,
                    frames_bgr=frames_bgr,
                    threshold=threshold,
                    device=device,
                    imgsz=imgsz,
                )
                results.append(
                    ClipScore(
                        clip_index=plan.clip_index,
                        start_frame=plan.start_frame,
                        end_frame_inclusive=plan.end_frame_inclusive,
                        is_anomaly=is_anomaly,
                    )
                )

            frame_idx += 1
            pbar.update(1)

    finally:
        cap.release()
        pbar.close()

    results.sort(key=lambda x: x.clip_index)
    return results, total_frames


def save_output(output_path: Path, clip_scores: Sequence[ClipScore]) -> None:
    total_clips = len(clip_scores)
    anomaly_count = sum(1 for item in clip_scores if item.is_anomaly)
    anomaly_ratio = (anomaly_count / total_clips) if total_clips > 0 else 0.0

    payload = {
        "total_clips": total_clips,
        "anomaly_ratio": anomaly_ratio,
        "clip_scores": [asdict(item) for item in clip_scores],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def print_summary(
    video_path: Path,
    total_frames: int,
    clip_scores: Sequence[ClipScore],
    output_json: Path,
) -> None:
    total_clips = len(clip_scores)
    anomaly_count = sum(1 for item in clip_scores if item.is_anomaly)
    ok_count = total_clips - anomaly_count
    anomaly_ratio = (anomaly_count / total_clips) if total_clips > 0 else 0.0

    print("\n=== YOLO FILTER SUMMARY ===")
    print(f"Video            : {video_path}")
    print(f"Total frames     : {total_frames}")
    print(f"Total clips      : {total_clips}")
    print(f"Clips OK         : {ok_count}")
    print(f"Clips Anomaly    : {anomaly_count}")
    print(f"Anomaly ratio    : {anomaly_ratio:.4f}")
    print(f"Output JSON      : {output_json}")


def main() -> int:
    try:
        args = parse_args()
        validate_args(args)

        device = choose_device()
        print(f"Loading model: {args.model}")
        print(f"Using device : {device}")

        model = load_model(args.model, device=device)

        clip_scores, total_frames = process_video(
            video_path=args.video_real,
            model=model,
            threshold=args.threshold,
            sampled_frames=args.sampled_frames,
            temporal_stride=args.temporal_stride,
            window_step=args.window_step,
            imgsz=args.imgsz,
        )

        save_output(args.output_json, clip_scores)
        print_summary(
            video_path=args.video_real,
            total_frames=total_frames,
            clip_scores=clip_scores,
            output_json=args.output_json,
        )
        return 0

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())