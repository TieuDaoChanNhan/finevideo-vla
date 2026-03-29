#!/usr/bin/env python3
"""
phase4_yolo_cleaner_v3.py

Final high-performance Phase 4 cleaner:
- Read JSONL first and collect valid windows.
- Build needed_frames so only required frames are sent to YOLO.
- Decode video sequentially with OpenCV (no random seek).
- Cache per-frame boolean person detection result.
- Resolve windows as soon as their 8 frame booleans are available.
- Drop old cache entries continuously to save RAM.
- Use torch.inference_mode() for inference.
- Warm up CUDA/model before main loop.
- Report timing and throughput metrics at the end.

Rule:
- A window is 8 consecutive frames starting at window_id.
- If >= 4 of those 8 frames have no person detection
  (class 0, confidence >= threshold), the window is anomaly and dropped.
- Otherwise, keep the original JSONL line unchanged.

Example:
    python phase4_yolo_cleaner_v3.py \
        --video-real input.mp4 \
        --jsonl-in phase3_states.jsonl \
        --jsonl-out phase4_cleaned.jsonl \
        --threshold 0.75 \
        --batch-size 128
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Deque, Dict, List, Sequence, Set, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO


PERSON_CLASS_ID = 0
WINDOW_SIZE = 8
EMPTY_FRAME_CUTOFF = 4
DEFAULT_THRESHOLD = 0.75
DEFAULT_MODEL = "yolo26n.pt"
DEFAULT_BATCH_SIZE = 128
DEFAULT_IMGSZ = 640


@dataclass(frozen=True)
class WindowRecord:
    start_frame: int
    raw_line: str


@dataclass(frozen=True)
class RunMetrics:
    total_time_sec: float
    video_read_frames: int
    frames_sent_to_yolo: int
    yolo_infer_time_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4 YOLO cleaner v3 with streaming, frame skipping, frame cache, and warmup."
    )
    parser.add_argument(
        "--video-real",
        type=Path,
        required=True,
        help="Path to input MP4 video.",
    )
    parser.add_argument(
        "--jsonl-in",
        type=Path,
        required=True,
        help="Input JSONL file from Phase 3.",
    )
    parser.add_argument(
        "--jsonl-out",
        type=Path,
        required=True,
        help="Output cleaned JSONL file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Confidence threshold for person detection. Default: {DEFAULT_THRESHOLD}",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Ultralytics model path/name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of frames per YOLO batch. Default: {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help=f"Inference image size. Default: {DEFAULT_IMGSZ}",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.video_real.exists():
        raise FileNotFoundError(f"Video not found: {args.video_real}")
    if not args.jsonl_in.exists():
        raise FileNotFoundError(f"Input JSONL not found: {args.jsonl_in}")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1].")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be > 0.")


def choose_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_model(model_name: str, device: str) -> YOLO:
    model = YOLO(model_name)
    model.to(device)
    return model


def get_video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames <= 0:
        raise RuntimeError(f"Could not determine frame count for video: {video_path}")

    return total_frames


def load_windows_and_needed_frames(
    jsonl_path: Path,
    total_video_frames: int,
) -> Tuple[List[WindowRecord], Set[int]]:
    """
    Read JSONL, validate window_id, return:
    - sorted list of WindowRecord
    - needed_frames = union of all frames used by valid windows
    """
    grouped: DefaultDict[int, List[WindowRecord]] = defaultdict(list)
    skipped = 0
    total = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            if not line.strip():
                continue

            total += 1

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc

            if "window_id" not in payload:
                raise KeyError(f"Missing 'window_id' at line {line_no}")

            start_frame = payload["window_id"]
            if not isinstance(start_frame, int):
                raise TypeError(
                    f"'window_id' must be int at line {line_no}, got {type(start_frame).__name__}"
                )

            end_frame = start_frame + WINDOW_SIZE - 1
            if start_frame < 0 or end_frame >= total_video_frames:
                skipped += 1
                continue

            grouped[start_frame].append(WindowRecord(start_frame=start_frame, raw_line=line))

    if total == 0:
        raise RuntimeError(f"Input JSONL is empty: {jsonl_path}")

    if skipped > 0:
        print(f"[WARN] Skipped {skipped} invalid/out-of-range JSONL lines.")

    all_records: List[WindowRecord] = []
    needed_frames: Set[int] = set()

    for start_frame in sorted(grouped.keys()):
        records = grouped[start_frame]
        all_records.extend(records)
        for frame_idx in range(start_frame, start_frame + WINDOW_SIZE):
            needed_frames.add(frame_idx)

    return all_records, needed_frames


def frame_has_person(result: Any, threshold: float) -> bool:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return False

    cls_tensor = boxes.cls
    conf_tensor = boxes.conf
    if cls_tensor is None or conf_tensor is None:
        return False

    mask = (cls_tensor == PERSON_CLASS_ID) & (conf_tensor >= threshold)
    return bool(mask.any().item())


def warmup_model(
    model: YOLO,
    device: str,
    imgsz: int,
) -> None:
    """
    Warm up model / CUDA context with a dummy image.
    """
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    use_half = device.startswith("cuda")

    with torch.inference_mode():
        _ = model.predict(
            source=[dummy],
            conf=0.01,
            device=device,
            half=use_half,
            imgsz=imgsz,
            verbose=False,
            stream=False,
        )

    if device.startswith("cuda"):
        torch.cuda.synchronize()


def infer_frame_batch(
    model: YOLO,
    frames_bgr: Sequence[np.ndarray],
    threshold: float,
    device: str,
    imgsz: int,
) -> List[bool]:
    """
    Run YOLO once for a batch of frames and return per-frame boolean:
        True  -> frame contains person
        False -> empty frame
    """
    use_half = device.startswith("cuda")

    with torch.inference_mode():
        results = model.predict(
            source=list(frames_bgr),
            conf=threshold,
            device=device,
            half=use_half,
            imgsz=imgsz,
            verbose=False,
            stream=False,
        )

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    return [frame_has_person(result, threshold) for result in results]


def try_resolve_ready_windows(
    pending_windows: Deque[WindowRecord],
    frame_cache: Dict[int, bool],
    fout,
) -> Tuple[int, int]:
    """
    Resolve windows from queue front while all 8 booleans are present.
    """
    cut = 0
    keep = 0

    while pending_windows:
        record = pending_windows[0]
        start = record.start_frame
        end = start + WINDOW_SIZE - 1

        values: List[bool] = []
        ready = True
        for frame_idx in range(start, end + 1):
            value = frame_cache.get(frame_idx)
            if value is None:
                ready = False
                break
            values.append(value)

        if not ready:
            break

        pending_windows.popleft()

        empty_count = sum(1 for v in values if not v)
        if empty_count >= EMPTY_FRAME_CUTOFF:
            cut += 1
        else:
            fout.write(record.raw_line + "\n")
            keep += 1

    return cut, keep


def cleanup_frame_cache(
    frame_cache: Dict[int, bool],
    pending_windows: Deque[WindowRecord],
) -> None:
    """
    Remove obsolete cached frame results.
    """
    if not pending_windows:
        frame_cache.clear()
        return

    earliest_needed_frame = pending_windows[0].start_frame
    obsolete_keys = [k for k in frame_cache.keys() if k < earliest_needed_frame]
    for k in obsolete_keys:
        del frame_cache[k]


def process_video(
    video_path: Path,
    jsonl_in: Path,
    jsonl_out: Path,
    model: YOLO,
    threshold: float,
    batch_size: int,
    imgsz: int,
) -> Tuple[int, int, int, RunMetrics]:
    total_video_frames = get_video_frame_count(video_path)
    all_records, needed_frames = load_windows_and_needed_frames(
        jsonl_in,
        total_video_frames=total_video_frames,
    )

    total_input = len(all_records)
    if total_input == 0:
        jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_out.open("w", encoding="utf-8"):
            pass
        empty_metrics = RunMetrics(
            total_time_sec=0.0,
            video_read_frames=0,
            frames_sent_to_yolo=0,
            yolo_infer_time_sec=0.0,
        )
        return 0, 0, 0, empty_metrics

    pending_windows: Deque[WindowRecord] = deque(all_records)
    frame_cache: Dict[int, bool] = {}

    last_needed_frame = max(needed_frames)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    device = choose_device()

    batch_frames: List[np.ndarray] = []
    batch_indices: List[int] = []

    total_cut = 0
    total_keep = 0

    video_read_frames = 0
    frames_sent_to_yolo = 0
    yolo_infer_time_sec = 0.0

    jsonl_out.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()

    def flush_batch(fout) -> Tuple[int, int]:
        nonlocal batch_frames
        nonlocal batch_indices
        nonlocal frames_sent_to_yolo
        nonlocal yolo_infer_time_sec

        if not batch_frames:
            return 0, 0

        infer_start = time.perf_counter()
        person_flags = infer_frame_batch(
            model=model,
            frames_bgr=batch_frames,
            threshold=threshold,
            device=device,
            imgsz=imgsz,
        )
        yolo_infer_time_sec += time.perf_counter() - infer_start
        frames_sent_to_yolo += len(batch_frames)

        for frame_idx, has_person in zip(batch_indices, person_flags):
            frame_cache[frame_idx] = has_person

        cut, keep = try_resolve_ready_windows(
            pending_windows=pending_windows,
            frame_cache=frame_cache,
            fout=fout,
        )
        cleanup_frame_cache(frame_cache=frame_cache, pending_windows=pending_windows)

        batch_frames = []
        batch_indices = []
        return cut, keep

    try:
        with jsonl_out.open("w", encoding="utf-8") as fout:
            with tqdm(total=total_input, desc="Cleaning windows", unit="window") as pbar:
                frame_idx = 0

                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    video_read_frames += 1

                    if frame_idx > last_needed_frame:
                        break

                    if frame_idx not in needed_frames:
                        frame_idx += 1
                        continue

                    batch_frames.append(frame)
                    batch_indices.append(frame_idx)

                    if len(batch_frames) >= batch_size:
                        cut, keep = flush_batch(fout)
                        total_cut += cut
                        total_keep += keep
                        pbar.update(cut + keep)

                    frame_idx += 1

                if batch_frames:
                    cut, keep = flush_batch(fout)
                    total_cut += cut
                    total_keep += keep
                    pbar.update(cut + keep)

    finally:
        cap.release()

    total_time_sec = time.perf_counter() - start_time

    if pending_windows:
        unresolved = len(pending_windows)
        raise RuntimeError(
            f"{unresolved} windows could not be resolved. "
            "This usually means required frames were not decoded/inferred correctly."
        )

    metrics = RunMetrics(
        total_time_sec=total_time_sec,
        video_read_frames=video_read_frames,
        frames_sent_to_yolo=frames_sent_to_yolo,
        yolo_infer_time_sec=yolo_infer_time_sec,
    )
    return total_input, total_cut, total_keep, metrics


def print_summary(
    video_path: Path,
    jsonl_in: Path,
    jsonl_out: Path,
    total_input: int,
    total_cut: int,
    total_keep: int,
    metrics: RunMetrics,
) -> None:
    cut_ratio = (total_cut / total_input) if total_input > 0 else 0.0
    keep_ratio = (total_keep / total_input) if total_input > 0 else 0.0

    video_read_fps = (
        metrics.video_read_frames / metrics.total_time_sec
        if metrics.total_time_sec > 0.0
        else 0.0
    )
    yolo_inference_fps = (
        metrics.frames_sent_to_yolo / metrics.yolo_infer_time_sec
        if metrics.yolo_infer_time_sec > 0.0
        else 0.0
    )

    print("\n=== PHASE 4 YOLO CLEANER V3 SUMMARY ===")
    print(f"Video                 : {video_path}")
    print(f"Input JSONL           : {jsonl_in}")
    print(f"Output JSONL          : {jsonl_out}")
    print(f"Total windows         : {total_input}")
    print(f"Cut anomalies         : {total_cut}")
    print(f"Kept windows          : {total_keep}")
    print(f"Cut ratio             : {cut_ratio:.4f}")
    print(f"Keep ratio            : {keep_ratio:.4f}")
    print(f"Total time (s)        : {metrics.total_time_sec:.4f}")
    print(f"Video read frames     : {metrics.video_read_frames}")
    print(f"Frames sent to YOLO   : {metrics.frames_sent_to_yolo}")
    print(f"YOLO infer time (s)   : {metrics.yolo_infer_time_sec:.4f}")
    print(f"Video read FPS        : {video_read_fps:.2f}")
    print(f"YOLO Inference FPS    : {yolo_inference_fps:.2f}")


def main() -> int:
    try:
        args = parse_args()
        validate_args(args)

        device = choose_device()
        print(f"Loading model : {args.model}")
        print(f"Using device  : {device}")
        print(f"Batch size    : {args.batch_size}")
        print(f"Image size    : {args.imgsz}")

        model = load_model(args.model, device)

        print("Warming up model...")
        warmup_model(model=model, device=device, imgsz=args.imgsz)

        total_input, total_cut, total_keep, metrics = process_video(
            video_path=args.video_real,
            jsonl_in=args.jsonl_in,
            jsonl_out=args.jsonl_out,
            model=model,
            threshold=args.threshold,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
        )

        print_summary(
            video_path=args.video_real,
            jsonl_in=args.jsonl_in,
            jsonl_out=args.jsonl_out,
            total_input=total_input,
            total_cut=total_cut,
            total_keep=total_keep,
            metrics=metrics,
        )
        return 0

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())