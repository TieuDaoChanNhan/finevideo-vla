#!/usr/bin/env python3
"""
phase4_yolo_cleaner_slurm.py

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

SLURM mode:
- Scan all *_states.jsonl in --input-dir
- Split files evenly across SLURM array workers using:
    - SLURM_ARRAY_TASK_ID (default: 1)
    - SLURM_ARRAY_TASK_COUNT (default: 1)
- For each assigned JSONL file:
    - Infer video_id from "<video_id>_states.jsonl"
    - Video path  = --videos-dir / f"{video_id}.mp4"
    - Output path = --output-dir / f"{video_id}_cleaned.jsonl"
- Safe resume:
    - If output exists => skip
    - Otherwise write to temp file "<output>.tmp"
    - On success: os.replace(temp, output)
    - On failure: delete temp

Rule:
- A window is 8 consecutive frames starting at window_id.
- If >= 4 of those 8 frames have no person detection
  (class 0, confidence >= threshold), the window is anomaly and dropped.
- Otherwise, keep the original JSONL line unchanged.

Example:
    python phase4_yolo_cleaner_slurm.py \
        --videos-dir /data/videos \
        --input-dir /data/phase3_states \
        --resampled-npy-dir /data/outputs/3d_npy_30fps \
        --output-dir /data/phase4_cleaned \
        --threshold 0.75 \
        --batch-size 128

--- Fix (2026-07-20): frame-index / fps mismatch ---
window_id in --input-dir's *_states.jsonl is indexed on the 30fps grid
Phase 2.5 (phase2_5_resample_30fps.py) resampled every pose to -- NOT on
the native fps of the file this script decodes from --videos-dir. The
previous version of this script used the raw sequential decode-order
index straight from the native-fps video as the frame_cache key, silently
assuming the two were the same timeline. They are not for any video whose
native fps != 30 (measured: 35% of FineVideo, 15,321/43,751 videos in
outputs/fps_lookup.json deviate >=5% from 30fps). Concretely, verified on
real production output for a 25fps video (-2MKTg-LNio): native frame count
is 12,758 but its states_jsonl_30fps runs up to window_id+8=15,304 -- so
every window past 12,758 was silently dropped (losing the last ~1/6 of
the video), and every window that WAS kept read YOLO's person-presence
result from the wrong point in time, drifting by up to ~20% of the
video's duration by the end.

Fix: build an explicit native_idx <-> resampled_idx mapping via
np.round(np.linspace(0, N-1, M)) (N = native frame count from the video
file, M = resampled frame count read directly from the corresponding
--resampled-npy-dir/{video_id}.npy shape -- the exact M Phase 2.5
produced). This is the same endpoint-aligned linspace mapping
resample_pose() used going the other direction. Frames are still decoded
sequentially with no random seeks (same performance design as before);
each decoded native frame's YOLO result is now written into frame_cache
under every resampled_idx that maps back to it, so windows are always
keyed/read in the timeline they were actually defined in (resampled-space,
matching window_id). For already-30fps videos the mapping is the
identity and behaviour is unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
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
JSONL_SUFFIX = "_states.jsonl"


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
        description="Phase 4 YOLO cleaner for SLURM array workers with streaming, frame skipping, frame cache, and warmup."
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        required=True,
        help="Directory containing input MP4 videos named <video_id>.mp4",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing input JSONL files named <video_id>_states.jsonl",
    )
    parser.add_argument(
        "--resampled-npy-dir",
        type=Path,
        required=True,
        help="Directory containing Phase 2.5 output <video_id>.npy (30fps-resampled), "
             "used to read the exact resampled frame count M for the native<->resampled "
             "index mapping (see module docstring).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write cleaned JSONL files named <video_id>_cleaned.jsonl",
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
    parser.add_argument(
        "--window-size",
        type=int,
        default=WINDOW_SIZE,
        help=f"Must match the --window-size used in Phase 3 (window_id spacing in the input "
             f"JSONL). Default: {WINDOW_SIZE}. 2026-07-22: use 24 to match the wider cosmos "
             f"chunk window -- see REPORT.md #38.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {args.videos_dir}")
    if not args.videos_dir.is_dir():
        raise NotADirectoryError(f"--videos-dir is not a directory: {args.videos_dir}")

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"--input-dir is not a directory: {args.input_dir}")

    if not args.resampled_npy_dir.exists():
        raise FileNotFoundError(f"Resampled npy directory not found: {args.resampled_npy_dir}")
    if not args.resampled_npy_dir.is_dir():
        raise NotADirectoryError(f"--resampled-npy-dir is not a directory: {args.resampled_npy_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.output_dir.is_dir():
        raise NotADirectoryError(f"--output-dir is not a directory: {args.output_dir}")

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


def build_native_resampled_maps(native_frames: int, resampled_frames: int):
    """Endpoint-aligned linspace mapping, inverse of resample_pose()'s own
    linspace(0,1,N) <-> linspace(0,1,M) (phase2_5_resample_30fps.py). Returns
    (native_for_resampled, resampled_for_native): the first is an array of
    length M giving the matching native frame index for each resampled
    index; the second is the reverse lookup (native_idx -> list of
    resampled indices)."""
    if resampled_frames < 1:
        return np.zeros(0, dtype=np.int64), defaultdict(list)
    native_for_resampled = np.round(
        np.linspace(0, max(native_frames - 1, 0), resampled_frames)
    ).astype(np.int64)
    resampled_for_native: DefaultDict[int, List[int]] = defaultdict(list)
    for j, native_idx in enumerate(native_for_resampled):
        resampled_for_native[int(native_idx)].append(j)
    return native_for_resampled, resampled_for_native


def load_windows_and_needed_frames(
    jsonl_path: Path,
    resampled_frames: int,
) -> Tuple[List[WindowRecord], Set[int]]:
    """
    Read JSONL, validate window_id, return:
    - sorted list of WindowRecord
    - needed_frames = union of all frames used by valid windows

    window_id is indexed on the resampled (30fps) grid -- the out-of-range
    bound must be checked against resampled_frames (M, from Phase 2.5's
    output), not the native video's frame count (see module docstring).
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
            if start_frame < 0 or end_frame >= resampled_frames:
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
    resampled_npy_path: Path,
    jsonl_out: Path,
    model: YOLO,
    threshold: float,
    batch_size: int,
    imgsz: int,
) -> Tuple[int, int, int, RunMetrics]:
    native_frames = get_video_frame_count(video_path)
    resampled_frames = int(np.load(resampled_npy_path, mmap_mode="r").shape[0])

    all_records, needed_resampled_frames = load_windows_and_needed_frames(
        jsonl_in,
        resampled_frames=resampled_frames,
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

    native_for_resampled, resampled_for_native = build_native_resampled_maps(
        native_frames, resampled_frames,
    )
    # Only native frames that map back to at least one *needed* resampled index are worth decoding into a batch.
    needed_native_frames = {
        native_idx for native_idx, resampled_idxs in resampled_for_native.items()
        if any(j in needed_resampled_frames for j in resampled_idxs)
    }

    pending_windows: Deque[WindowRecord] = deque(all_records)
    frame_cache: Dict[int, bool] = {}

    last_needed_native_frame = max(needed_native_frames) if needed_native_frames else -1

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

        # batch_indices are native-fps decode indices; fan each result out to
        # every resampled-space index that maps back to it (see build_native_resampled_maps).
        for native_idx, has_person in zip(batch_indices, person_flags):
            for resampled_idx in resampled_for_native.get(native_idx, []):
                frame_cache[resampled_idx] = has_person

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
                native_idx = 0

                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    video_read_frames += 1

                    if native_idx > last_needed_native_frame:
                        break

                    if native_idx not in needed_native_frames:
                        native_idx += 1
                        continue

                    batch_frames.append(frame)
                    batch_indices.append(native_idx)

                    if len(batch_frames) >= batch_size:
                        cut, keep = flush_batch(fout)
                        total_cut += cut
                        total_keep += keep
                        pbar.update(cut + keep)

                    native_idx += 1

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


def get_slurm_array_info() -> Tuple[int, int]:
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
    task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))

    if task_id <= 0:
        raise ValueError(f"SLURM_ARRAY_TASK_ID must be >= 1, got {task_id}")
    if task_count <= 0:
        raise ValueError(f"SLURM_ARRAY_TASK_COUNT must be >= 1, got {task_count}")
    if task_id > task_count:
        raise ValueError(
            f"SLURM_ARRAY_TASK_ID ({task_id}) cannot be greater than SLURM_ARRAY_TASK_COUNT ({task_count})"
        )

    return task_id, task_count


def split_chunk_indices(total_items: int, task_id: int, task_count: int) -> Tuple[int, int]:
    """
    Contiguous even chunk split, 1-based SLURM task_id.

    Example:
      total=10, workers=3
      worker1 -> [0:4]
      worker2 -> [4:7]
      worker3 -> [7:10]
    """
    start = ((task_id - 1) * total_items) // task_count
    end = (task_id * total_items) // task_count
    return start, end


def infer_video_id_from_jsonl(jsonl_path: Path) -> str:
    name = jsonl_path.name
    if not name.endswith(JSONL_SUFFIX):
        raise ValueError(
            f"Input JSONL filename must end with '{JSONL_SUFFIX}', got: {jsonl_path}"
        )
    return name[: -len(JSONL_SUFFIX)]


def list_input_jsonl_files(input_dir: Path) -> List[Path]:
    files = sorted(
        p for p in input_dir.glob(f"*{JSONL_SUFFIX}")
        if p.is_file()
    )
    return files


def remove_file_if_exists(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except FileNotFoundError:
        pass


def main() -> int:
    global WINDOW_SIZE
    try:
        args = parse_args()
        validate_args(args)
        WINDOW_SIZE = args.window_size

        task_id, task_count = get_slurm_array_info()

        all_jsonl_files = list_input_jsonl_files(args.input_dir)
        total_files = len(all_jsonl_files)
        if total_files == 0:
            print(f"[WARN] No '*{JSONL_SUFFIX}' files found in: {args.input_dir}")
            return 0

        start_idx, end_idx = split_chunk_indices(total_files, task_id, task_count)
        assigned_files = all_jsonl_files[start_idx:end_idx]

        device = choose_device()
        print(f"Loading model      : {args.model}")
        print(f"Using device       : {device}")
        print(f"Batch size         : {args.batch_size}")
        print(f"Image size         : {args.imgsz}")
        print(f"Threshold          : {args.threshold}")
        print(f"SLURM task         : {task_id}/{task_count}")
        print(f"Total input files  : {total_files}")
        print(f"Assigned range     : [{start_idx}:{end_idx})")
        print(f"Assigned files     : {len(assigned_files)}")

        if not assigned_files:
            print("[INFO] No files assigned to this worker. Exiting.")
            return 0

        model = load_model(args.model, device)

        print("Warming up model...")
        warmup_model(model=model, device=device, imgsz=args.imgsz)

        processed = 0
        skipped = 0
        failed = 0

        for idx, jsonl_in in enumerate(assigned_files, start=1):
            video_id = infer_video_id_from_jsonl(jsonl_in)
            video_path = args.videos_dir / f"{video_id}.mp4"
            resampled_npy_path = args.resampled_npy_dir / f"{video_id}.npy"
            output_path = args.output_dir / f"{video_id}_cleaned.jsonl"
            temp_path = Path(f"{output_path}.tmp")

            print("\n" + "=" * 100)
            print(f"[{idx}/{len(assigned_files)}] video_id   : {video_id}")
            print(f"JSONL in            : {jsonl_in}")
            print(f"Video path          : {video_path}")
            print(f"Resampled npy       : {resampled_npy_path}")
            print(f"Output path         : {output_path}")
            print(f"Temp path           : {temp_path}")

            if output_path.exists():
                print(f"[SKIP] Output already exists: {output_path}")
                skipped += 1
                continue

            if not video_path.exists():
                print(f"[ERROR] Video not found: {video_path}", file=sys.stderr)
                failed += 1
                continue

            if not resampled_npy_path.exists():
                print(f"[ERROR] Resampled npy not found: {resampled_npy_path}", file=sys.stderr)
                failed += 1
                continue

            remove_file_if_exists(temp_path)

            try:
                total_input, total_cut, total_keep, metrics = process_video(
                    video_path=video_path,
                    jsonl_in=jsonl_in,
                    resampled_npy_path=resampled_npy_path,
                    jsonl_out=temp_path,
                    model=model,
                    threshold=args.threshold,
                    batch_size=args.batch_size,
                    imgsz=args.imgsz,
                )

                os.replace(temp_path, output_path)

                print_summary(
                    video_path=video_path,
                    jsonl_in=jsonl_in,
                    jsonl_out=output_path,
                    total_input=total_input,
                    total_cut=total_cut,
                    total_keep=total_keep,
                    metrics=metrics,
                )
                print(f"[DONE] Wrote: {output_path}")
                processed += 1

            except Exception as exc:
                remove_file_if_exists(temp_path)
                print(f"[ERROR] Failed processing '{video_id}': {exc}", file=sys.stderr)
                failed += 1

        print("\n" + "#" * 100)
        print("WORKER FINAL SUMMARY")
        print(f"SLURM task          : {task_id}/{task_count}")
        print(f"Assigned files      : {len(assigned_files)}")
        print(f"Processed           : {processed}")
        print(f"Skipped             : {skipped}")
        print(f"Failed              : {failed}")

        return 0 if failed == 0 else 1

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())