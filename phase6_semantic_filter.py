#!/usr/bin/env python3
"""
Phase 6: Semantic Filtering with V-JEPA on SLURM

This script keeps `vjepa_filter.py` intact as a reusable module and adds a
cluster-oriented batch runner with:
- Decord-based random-access video reading (no full-video RAM preload)
- SLURM array sharding
- JSONL batch processing
- token/window pruning using V-JEPA anomaly spans
- yield-rate gating for final dataset emission

Expected layout
---------------
outputs/clean_pose_dataset/*.jsonl
videos/{video_id}.mp4
skeletons/{video_id}_skeleton.mp4

Output
------
outputs/phase6_final_dataset/phase6_worker_{task_id:05d}.jsonl
outputs/phase6_final_dataset/phase6_worker_{task_id:05d}.log.jsonl

Install
-------
pip install -U "torch>=2.2" torchvision timm huggingface_hub einops numpy
pip install -U git+https://github.com/huggingface/transformers
pip install -U decord
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from glob import glob
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu

from vjepa_filter import ClipScore, FilterResult, VJEPAFilter, seed_everything


class VideoReaderDecord:
    """
    Memory-safe video reader with random frame access.

    Unlike the original PyAV helper that decodes the whole video into RAM, this
    class keeps a Decord VideoReader handle open and fetches only the frames
    needed for each clip.

    Returned frame format:
        uint8 numpy array with shape [T, H, W, C] in RGB order
    """

    def __init__(self, path: str, num_threads: int = 1, ctx: Optional[Any] = None) -> None:
        self.path = path
        self.ctx = ctx if ctx is not None else cpu(0)
        self.num_threads = int(num_threads)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video not found: {path}")
        self.reader = VideoReader(path, ctx=self.ctx, num_threads=self.num_threads)
        self.num_frames = len(self.reader)
        if self.num_frames <= 0:
            raise RuntimeError(f"No frames available in video: {path}")

    def get_frames(self, indices: Sequence[int]) -> np.ndarray:
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size == 0:
            raise ValueError("indices must be non-empty")
        if idx.min() < 0 or idx.max() >= self.num_frames:
            raise IndexError(
                f"Requested frame indices out of range for {self.path}. "
                f"min={idx.min()}, max={idx.max()}, num_frames={self.num_frames}"
            )
        frames = self.reader.get_batch(idx.tolist()).asnumpy()
        return frames


class VJEPAFilterDecord(VJEPAFilter):
    """
    Keep the original model/preprocessing logic from `vjepa_filter.py`,
    but replace RAM-heavy full decode with Decord random access.
    """

    @torch.inference_mode()
    def filter_pair(
        self,
        video_real: str,
        video_skeleton: str,
        threshold: float = 0.70,
        sampled_frames: int = 16,
        temporal_stride: int = 4,
        window_step: int = 16,
        batch_size: int = 8,
        decord_num_threads: int = 1,
    ) -> FilterResult:
        reader_real = VideoReaderDecord(video_real, num_threads=decord_num_threads)
        reader_skel = VideoReaderDecord(video_skeleton, num_threads=decord_num_threads)

        num_frames = min(reader_real.num_frames, reader_skel.num_frames)

        windows = self.build_windows(
            num_frames=num_frames,
            sampled_frames=sampled_frames,
            temporal_stride=temporal_stride,
            window_step=window_step,
        )

        if not windows:
            raise RuntimeError(
                "Video too short for the requested clip sampling. "
                f"Need at least {(sampled_frames - 1) * temporal_stride + 1} decoded frames."
            )

        clip_scores: List[ClipScore] = []

        for batch_start in range(0, len(windows), batch_size):
            batch_windows = windows[batch_start: batch_start + batch_size]
            real_batch = [reader_real.get_frames(idx) for _, _, idx in batch_windows]
            skel_batch = [reader_skel.get_frames(idx) for _, _, idx in batch_windows]

            sims = self.similarity_scores(real_batch, skel_batch).cpu().tolist()

            for local_i, ((start_frame, end_frame, idx), sim) in enumerate(zip(batch_windows, sims)):
                is_anomaly = bool(sim < threshold)
                clip_scores.append(
                    ClipScore(
                        clip_index=batch_start + local_i,
                        start_frame=int(start_frame),
                        end_frame_inclusive=int(end_frame),
                        sampled_indices=[int(x) for x in idx],
                        similarity=float(sim),
                        is_anomaly=is_anomaly,
                    )
                )

        anomaly_ratio = (
            sum(1 for x in clip_scores if x.is_anomaly) / max(1, len(clip_scores))
        )

        sample_span = sampled_frames * temporal_stride

        return FilterResult(
            video_real=video_real,
            video_skeleton=video_skeleton,
            model_name=self.model_name,
            threshold=float(threshold),
            sample_span=int(sample_span),
            sampled_frames=int(sampled_frames),
            temporal_stride=int(temporal_stride),
            window_step=int(window_step),
            pooling=self.pooling,
            total_clips=len(clip_scores),
            anomaly_ratio=float(anomaly_ratio),
            clip_scores=clip_scores,
        )


def load_jsonl_records(input_dir: str) -> List[Tuple[str, int, Dict[str, Any]]]:
    jsonl_paths = sorted(glob(os.path.join(input_dir, "*.jsonl")))
    tasks: List[Tuple[str, int, Dict[str, Any]]] = []
    for path in jsonl_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                tasks.append((path, line_idx, record))
    return tasks


def shard_tasks(
    tasks: List[Tuple[str, int, Dict[str, Any]]],
    task_id: int,
    task_count: int,
) -> List[Tuple[str, int, Dict[str, Any]]]:
    if task_count <= 0:
        raise ValueError("task_count must be >= 1")
    if task_id < 0 or task_id >= task_count:
        raise ValueError(f"task_id must be in [0, {task_count - 1}]")
    return tasks[task_id::task_count]


def get_slurm_shard() -> Tuple[int, int]:
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    return task_id, task_count


def find_token_list(record: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    candidates = ["token_list", "tokens", "agent_tokens", "pose_tokens", "windows"]
    for key in candidates:
        value = record.get(key)
        if isinstance(value, list):
            return key, value
    raise KeyError(
        "Could not find token list in record. Expected one of: "
        + ", ".join(candidates)
    )


def normalize_token_list(token_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for token in token_list:
        if not isinstance(token, dict):
            continue
        if "window_id" not in token:
            continue
        try:
            window_id = int(token["window_id"])
        except Exception:
            continue
        new_token = dict(token)
        new_token["window_id"] = window_id
        out.append(new_token)
    return out


def merge_anomaly_ranges(clip_scores: List[ClipScore]) -> List[Tuple[int, int]]:
    ranges = sorted(
        (int(c.start_frame), int(c.end_frame_inclusive))
        for c in clip_scores
        if c.is_anomaly
    )
    if not ranges:
        return []

    merged: List[Tuple[int, int]] = [ranges[0]]
    for start, end in ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def interval_contains(value: int, intervals: Sequence[Tuple[int, int]]) -> bool:
    for start, end in intervals:
        if start <= value <= end:
            return True
    return False


def apply_semantic_filter_to_jsonl(
    filter_result: FilterResult,
    token_list: List[Dict[str, Any]],
    min_yield_rate: float = 0.4,
) -> Dict[str, Any]:
    token_list = normalize_token_list(token_list)
    if not token_list:
        return {
            "clean_tokens": [],
            "removed_tokens": 0,
            "removed_ranges": [],
            "expected_token_count": 0,
            "yield_rate": 0.0,
            "passes": False,
        }

    removed_ranges = merge_anomaly_ranges(filter_result.clip_scores)

    clean_tokens: List[Dict[str, Any]] = []
    removed_tokens = 0
    for token in token_list:
        window_id = int(token["window_id"])
        if interval_contains(window_id, removed_ranges):
            removed_tokens += 1
            continue
        clean_tokens.append(token)

    window_ids = [int(t["window_id"]) for t in token_list]
    min_window_id = min(window_ids)
    max_window_id = max(window_ids)

    expected_token_count = ((max_window_id - min_window_id) // 16) + 1
    expected_token_count = max(1, int(expected_token_count))

    yield_rate = len(clean_tokens) / expected_token_count
    passes = (yield_rate >= float(min_yield_rate)) and (len(clean_tokens) >= 3)

    return {
        "clean_tokens": clean_tokens,
        "removed_tokens": int(removed_tokens),
        "removed_ranges": [[int(a), int(b)] for a, b in removed_ranges],
        "expected_token_count": int(expected_token_count),
        "yield_rate": float(yield_rate),
        "passes": bool(passes),
    }


def build_video_paths(video_id: str, video_dir: str, skeleton_dir: str) -> Tuple[str, str]:
    video_path = os.path.join(video_dir, f"{video_id}.mp4")
    skeleton_path = os.path.join(skeleton_dir, f"{video_id}_skeleton.mp4")
    return video_path, skeleton_path


def process_record(
    record: Dict[str, Any],
    model: VJEPAFilterDecord,
    video_dir: str,
    skeleton_dir: str,
    threshold: float,
    sampled_frames: int,
    temporal_stride: int,
    window_step: int,
    batch_size: int,
    min_yield_rate: float,
    decord_num_threads: int,
) -> Dict[str, Any]:
    video_id = str(record["video_id"])
    token_key, token_list = find_token_list(record)
    token_list = normalize_token_list(token_list)

    video_path, skeleton_path = build_video_paths(video_id, video_dir, skeleton_dir)

    if not os.path.exists(video_path):
        return {
            "status": "missing_video",
            "video_id": video_id,
            "video_path": video_path,
        }

    if not os.path.exists(skeleton_path):
        return {
            "status": "missing_skeleton",
            "video_id": video_id,
            "skeleton_path": skeleton_path,
        }

    filter_result = model.filter_pair(
        video_real=video_path,
        video_skeleton=skeleton_path,
        threshold=threshold,
        sampled_frames=sampled_frames,
        temporal_stride=temporal_stride,
        window_step=window_step,
        batch_size=batch_size,
        decord_num_threads=decord_num_threads,
    )

    semantic = apply_semantic_filter_to_jsonl(
        filter_result=filter_result,
        token_list=token_list,
        min_yield_rate=min_yield_rate,
    )

    if not semantic["passes"]:
        return {
            "status": "rejected_low_yield",
            "video_id": video_id,
            "yield_rate": semantic["yield_rate"],
            "clean_token_count": len(semantic["clean_tokens"]),
            "expected_token_count": semantic["expected_token_count"],
            "removed_tokens": semantic["removed_tokens"],
            "anomaly_ratio": float(filter_result.anomaly_ratio),
            "removed_ranges": semantic["removed_ranges"],
        }

    out_record = dict(record)
    out_record[token_key] = semantic["clean_tokens"]
    out_record["phase6_semantic_filter"] = {
        "model_name": filter_result.model_name,
        "threshold": float(filter_result.threshold),
        "window_step": int(filter_result.window_step),
        "sampled_frames": int(filter_result.sampled_frames),
        "temporal_stride": int(filter_result.temporal_stride),
        "anomaly_ratio": float(filter_result.anomaly_ratio),
        "yield_rate": float(semantic["yield_rate"]),
        "expected_token_count": int(semantic["expected_token_count"]),
        "clean_token_count": int(len(semantic["clean_tokens"])),
        "removed_tokens": int(semantic["removed_tokens"]),
        "removed_ranges": semantic["removed_ranges"],
    }

    return {
        "status": "accepted",
        "video_id": video_id,
        "output_record": out_record,
        "yield_rate": semantic["yield_rate"],
        "clean_token_count": len(semantic["clean_tokens"]),
        "expected_token_count": semantic["expected_token_count"],
        "removed_tokens": semantic["removed_tokens"],
        "anomaly_ratio": float(filter_result.anomaly_ratio),
    }


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 6 Semantic Filtering on SLURM")
    parser.add_argument("--input-dir", type=str, default="outputs/clean_pose_dataset")
    parser.add_argument("--video-dir", type=str, default="videos")
    parser.add_argument("--skeleton-dir", type=str, default="skeletons")
    parser.add_argument("--output-dir", type=str, default="outputs/phase6_final_dataset")

    parser.add_argument("--model-name", type=str, default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--min-yield-rate", type=float, default=0.40)

    parser.add_argument("--sampled-frames", type=int, default=16)
    parser.add_argument("--temporal-stride", type=int, default=4)
    parser.add_argument("--window-step", type=int, default=16)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pooling", type=str, choices=["gap", "attn"], default="gap")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--force-size", type=int, default=None)
    parser.add_argument("--attn-implementation", type=str, choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--torch-compile", action="store_true")

    parser.add_argument("--decord-num-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)

    task_id, task_count = get_slurm_shard()

    all_tasks = load_jsonl_records(args.input_dir)
    shard = shard_tasks(all_tasks, task_id=task_id, task_count=task_count)

    os.makedirs(args.output_dir, exist_ok=True)
    accepted_path = os.path.join(args.output_dir, f"phase6_worker_{task_id:05d}.jsonl")
    log_path = os.path.join(args.output_dir, f"phase6_worker_{task_id:05d}.log.jsonl")

    model = VJEPAFilterDecord(
        model_name=args.model_name,
        pooling=args.pooling,
        device=args.device,
        amp=args.amp,
        force_size=args.force_size,
        attn_implementation=args.attn_implementation,
        use_torch_compile=args.torch_compile,
    )

    summary = {
        "task_id": task_id,
        "task_count": task_count,
        "total_records_all": len(all_tasks),
        "total_records_this_worker": len(shard),
        "accepted": 0,
        "rejected_low_yield": 0,
        "missing_video": 0,
        "missing_skeleton": 0,
        "errors": 0,
    }

    print("=" * 96)
    print("Phase 6 Semantic Filter")
    print("=" * 96)
    print(f"task_id={task_id} task_count={task_count}")
    print(f"records_total={len(all_tasks)} records_this_worker={len(shard)}")
    print(f"output_jsonl={accepted_path}")
    print(f"log_jsonl={log_path}")
    print("-" * 96)

    for src_path, line_idx, record in shard:
        video_id = str(record.get("video_id", "UNKNOWN"))
        try:
            result = process_record(
                record=record,
                model=model,
                video_dir=args.video_dir,
                skeleton_dir=args.skeleton_dir,
                threshold=args.threshold,
                sampled_frames=args.sampled_frames,
                temporal_stride=args.temporal_stride,
                window_step=args.window_step,
                batch_size=args.batch_size,
                min_yield_rate=args.min_yield_rate,
                decord_num_threads=args.decord_num_threads,
            )

            status = result["status"]
            summary[status] = summary.get(status, 0) + 1

            if status == "accepted":
                append_jsonl(accepted_path, result["output_record"])

            log_row = {
                "status": status,
                "source_jsonl": src_path,
                "line_index": line_idx,
                **{k: v for k, v in result.items() if k != "output_record"},
            }
            append_jsonl(log_path, log_row)

            print(
                f"[{status}] video_id={video_id} "
                f"src={os.path.basename(src_path)}:{line_idx}"
            )

        except Exception as e:
            summary["errors"] += 1
            append_jsonl(
                log_path,
                {
                    "status": "error",
                    "source_jsonl": src_path,
                    "line_index": line_idx,
                    "video_id": video_id,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"[error] video_id={video_id} src={os.path.basename(src_path)}:{line_idx} err={e}")

    append_jsonl(
        log_path,
        {
            "status": "worker_summary",
            **summary,
        },
    )

    print("-" * 96)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
