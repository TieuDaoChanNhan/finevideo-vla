#!/usr/bin/env python3
"""
Merge Agent Tokens into training_ready_rank_*.jsonl and produce final_vla_rank_*.jsonl.

Each <agent> block contains exactly 256 uint8 tokens:

    Layout (256 tokens total):
    ┌──────────┬────────────────────────────┬──────────────────────────────────────┐
    │ index    │ content                    │ encoding                             │
    ├──────────┼────────────────────────────┼──────────────────────────────────────┤
    │ 0        │ scale  (1 token)           │ clip(scale / SCALE_MAX * 255, 0,255) │
    │ 1 – 51   │ anchor (51 tokens)         │ clip((x+2)/4 * 255, 0, 255) per dim  │
    │ 52 – 255 │ motion control pts (204 t) │ existing uint8 from Phase 5          │
    └──────────┴────────────────────────────┴──────────────────────────────────────┘

    Decoding back to absolute 3D coordinates:
        scale  = token[0] / 255.0 * SCALE_MAX
        anchor = token[1:52].reshape(17,3) / 255.0 * 4.0 - 2.0  (root-centered, metres)
        norm   = dequantize(token[52:256]).reshape(4, 17, 3)       # [-1, 1]
        cp_abs = norm * scale + anchor[0]   (anchor[0] = pelvis = origin, always ~0)
        then PCHIP on cp_abs over t in [0,1] → absolute trajectory

    Constants:
        ANCHOR_RANGE = 2.0 m  →  max joint distance from pelvis (body width/height)
        SCALE_MAX    = 2.0 m  →  max motion range in one 8-frame chunk (~0.27 s)
        Anchor precision  : 4.0 / 255 ≈ 15.7 mm
        Scale  precision  : 2.0 / 255 ≈  7.8 mm

All 256 values are in [0, 255] and map 1-to-1 to the <agent_N> vocab tokens
already registered in vocab_expanded.json.
"""

import argparse
import glob
import json
import math
import os
import re
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

TARGET_FPS = 30
CHUNK_SIZE = 8
AVC_PATTERN = re.compile(r"<avc_lm>\s*.*?\s*</avc_lm>", re.DOTALL)

ANCHOR_RANGE: float = 2.0   # metres; joints clamped to [-2, +2]
SCALE_MAX: float = 2.0      # metres; per-chunk scale clamped to [0, 2]
N_JOINTS: int = 17
N_DIMS: int = 3
N_ANCHOR_TOKENS: int = N_JOINTS * N_DIMS   # 51
N_MOTION_TOKENS: int = 204                  # 4 CPs × 17 joints × 3 dims
N_AGENT_TOKENS: int = 1 + N_ANCHOR_TOKENS + N_MOTION_TOKENS  # 256


# ── Quantization helpers ───────────────────────────────────────────────────────

def _quantize_scale(scale: float) -> int:
    clamped = float(np.clip(scale, 0.0, SCALE_MAX))
    return int(round(clamped / SCALE_MAX * 255))


def _quantize_anchor(anchor) -> List[int]:
    """anchor: list/array of shape (17, 3), root-centred metric coords."""
    arr = np.array(anchor, dtype=np.float32).flatten()   # (51,)
    arr = np.clip(arr, -ANCHOR_RANGE, ANCHOR_RANGE)
    quantized = np.round((arr + ANCHOR_RANGE) / (2.0 * ANCHOR_RANGE) * 255).astype(np.uint8)
    return quantized.tolist()


# ── Inverse helpers (for documentation / offline decoding) ────────────────────

def dequantize_scale(token: int) -> float:
    return token / 255.0 * SCALE_MAX


def dequantize_anchor(tokens: List[int]):
    """Returns ndarray of shape (17, 3) in metres, root-centred."""
    arr = np.array(tokens, dtype=np.float32)
    return (arr / 255.0 * 2.0 * ANCHOR_RANGE - ANCHOR_RANGE).reshape(N_JOINTS, N_DIMS)


def dequantize_motion(tokens: List[int]):
    """Returns ndarray of shape (4, 17, 3) in normalised [-1, 1] space."""
    arr = np.array(tokens, dtype=np.float32)
    return (arr / 127.5 - 1.0).reshape(4, N_JOINTS, N_DIMS)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def count_lines(path: str) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for count, _ in enumerate(f, start=1):
            pass
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge offline Agent Tokens into existing training_ready_rank_{X}.jsonl "
            "files and write final_vla_dataset_rank_{X}.jsonl outputs.\n\n"
            "Each <agent> block will contain exactly 256 uint8 values:\n"
            "  [scale(1)] [anchor(51)] [motion_CPs(204)]"
        )
    )
    parser.add_argument(
        "--input-glob",
        default="training_ready_rank_*.jsonl",
        help="Glob pattern for existing training_ready files.",
    )
    parser.add_argument(
        "--agent-tokens-dir",
        default=os.path.join("outputs", "agent_tokens"),
        help="Directory containing <video_id>_tokens.jsonl files.",
    )
    parser.add_argument(
        "--output-prefix",
        default="final_vla_dataset",
        help="Prefix for output files. Example: final_vla_dataset_rank_0.jsonl",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=2,
        help="Nearest-neighbour frame tolerance for agent token lookup.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip output files that already exist.",
    )
    parser.add_argument(
        "--strict-avc-count",
        action="store_true",
        help=(
            "Raise an error when the number of <avc_lm> blocks does not match the "
            "reconstructed number of chunk starts."
        ),
    )
    return parser.parse_args()


# ── Agent token loading ────────────────────────────────────────────────────────

def load_agent_dict(agent_tokens_path: str) -> Dict[int, str]:
    """
    Returns {window_id: "<agent> t0 t1 ... t255 </agent>"} where
    the 256 values are [scale, *anchor_flat, *motion_tokens].
    """
    agent_dict: Dict[int, str] = {}

    if not os.path.exists(agent_tokens_path):
        return agent_dict

    with open(agent_tokens_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                tqdm.write(
                    f"[WARN] Invalid JSON in {agent_tokens_path} at line {line_num}."
                )
                continue

            window_id = obj.get("window_id")
            package = obj.get("package", {})

            if window_id is None:
                continue
            try:
                window_id = int(window_id)
            except (TypeError, ValueError):
                continue

            motion_tokens = package.get("tokens")
            anchor = package.get("anchor")
            scale = package.get("scale")

            # Validate all required fields are present
            if motion_tokens is None or anchor is None or scale is None:
                tqdm.write(
                    f"[WARN] Missing tokens/anchor/scale in {agent_tokens_path} "
                    f"at line {line_num} — skipping window {window_id}."
                )
                continue

            if not isinstance(motion_tokens, list) or len(motion_tokens) != N_MOTION_TOKENS:
                tqdm.write(
                    f"[WARN] Expected {N_MOTION_TOKENS} motion tokens, "
                    f"got {len(motion_tokens) if isinstance(motion_tokens, list) else type(motion_tokens)} "
                    f"in {agent_tokens_path} at line {line_num}."
                )
                continue

            # Build the 256-token sequence: [scale(1)] + [anchor(51)] + [motion(204)]
            scale_tok = _quantize_scale(scale)
            anchor_toks = _quantize_anchor(anchor)
            all_tokens = [scale_tok] + anchor_toks + motion_tokens

            assert len(all_tokens) == N_AGENT_TOKENS, (
                f"Token count mismatch: expected {N_AGENT_TOKENS}, got {len(all_tokens)}"
            )

            token_text = " ".join(str(t) for t in all_tokens)
            agent_dict[window_id] = f"<agent> {token_text} </agent>"

    return agent_dict


# ── Merge logic (unchanged from original) ────────────────────────────────────

def infer_output_path(input_path: str, output_prefix: str) -> str:
    base_name = os.path.basename(input_path)
    parent_dir = os.path.dirname(input_path) or "."
    match = re.search(r"_rank_(\d+)\.jsonl$", base_name)
    if match:
        rank = match.group(1)
        return os.path.join(parent_dir, f"{output_prefix}_rank_{rank}.jsonl")
    stem, ext = os.path.splitext(base_name)
    return os.path.join(parent_dir, f"{output_prefix}_{stem}{ext or '.jsonl'}")


def compute_chunk_start_indices(start_sec: float, end_sec: float) -> Tuple[int, List[int]]:
    duration_sec = max(0.0, end_sec - start_sec)
    total_frames = int(duration_sec * TARGET_FPS)
    if total_frames <= 0:
        return total_frames, []
    chunk_start_indices = list(range(0, total_frames, CHUNK_SIZE))
    return total_frames, chunk_start_indices


def find_agent_string(
    agent_dict: Dict[int, str], absolute_frame_idx: int, tolerance: int = 2
) -> str:
    search_offsets = [0]
    for delta in range(1, tolerance + 1):
        search_offsets.extend([-delta, delta])
    for offset in search_offsets:
        candidate = absolute_frame_idx + offset
        if candidate in agent_dict:
            return agent_dict[candidate]
    return ""


def inject_agent_tokens(
    video_tokens: str,
    agent_insertions: List[str],
) -> Tuple[str, int]:
    if not video_tokens:
        return video_tokens, 0
    matches = list(AVC_PATTERN.finditer(video_tokens))
    if not matches:
        return video_tokens, 0
    parts: List[str] = []
    cursor = 0
    injected_count = 0
    for idx, match in enumerate(matches):
        parts.append(video_tokens[cursor : match.end()])
        agent_text = agent_insertions[idx] if idx < len(agent_insertions) else ""
        if agent_text:
            parts.append(" " + agent_text)
            injected_count += 1
        cursor = match.end()
    parts.append(video_tokens[cursor:])
    return "".join(parts), injected_count


def process_activity(
    activity: dict,
    agent_dict: Dict[int, str],
    tolerance: int,
    strict_avc_count: bool = False,
) -> Tuple[int, int, int]:
    start_sec, end_sec = activity.get("time_range_sec", [0.0, 0.0])[:2]
    start_sec = safe_float(start_sec, 0.0)
    end_sec = safe_float(end_sec, 0.0)
    video_tokens = activity.get("video_tokens", "")

    avc_matches = list(AVC_PATTERN.finditer(video_tokens or ""))
    avc_count = len(avc_matches)
    if avc_count == 0:
        return 0, 0, 0

    _, chunk_start_indices = compute_chunk_start_indices(start_sec, end_sec)
    absolute_start_frame = int(round(start_sec * TARGET_FPS))

    if avc_count != len(chunk_start_indices):
        message = (
            "[WARN] AVC/chunk count mismatch for activity "
            f"{activity.get('activity_id', '<unknown>')}: "
            f"found {avc_count} <avc_lm> blocks but reconstructed "
            f"{len(chunk_start_indices)} chunk starts."
        )
        if strict_avc_count:
            raise ValueError(message)
        tqdm.write(message)

    agent_insertions: List[str] = []
    misses = 0
    for idx in range(avc_count):
        chunk_start_idx = (
            chunk_start_indices[idx] if idx < len(chunk_start_indices) else idx * CHUNK_SIZE
        )
        absolute_frame_idx = absolute_start_frame + chunk_start_idx
        agent_text = find_agent_string(agent_dict, absolute_frame_idx, tolerance=tolerance)
        if not agent_text:
            misses += 1
        agent_insertions.append(agent_text)

    merged_video_tokens, injected_count = inject_agent_tokens(video_tokens, agent_insertions)
    activity["video_tokens"] = merged_video_tokens
    return avc_count, injected_count, misses


def process_video_record(
    record: dict,
    agent_tokens_dir: str,
    tolerance: int,
    strict_avc_count: bool = False,
) -> dict:
    video_id = record.get("video_id", "")
    agent_tokens_path = os.path.join(agent_tokens_dir, f"{video_id}_tokens.jsonl")
    agent_dict = load_agent_dict(agent_tokens_path)

    record_stats = {
        "video_id": video_id,
        "agent_file_found": os.path.exists(agent_tokens_path),
        "agent_windows": len(agent_dict),
        "activities_seen": 0,
        "avc_blocks_seen": 0,
        "agents_injected": 0,
        "agent_misses": 0,
    }

    for scene in record.get("scenes", []):
        if not isinstance(scene, dict):
            continue
        for activity in scene.get("activities", []):
            if not isinstance(activity, dict):
                continue
            record_stats["activities_seen"] += 1
            avc_count, injected_count, misses = process_activity(
                activity,
                agent_dict,
                tolerance=tolerance,
                strict_avc_count=strict_avc_count,
            )
            record_stats["avc_blocks_seen"] += avc_count
            record_stats["agents_injected"] += injected_count
            record_stats["agent_misses"] += misses

    return record_stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    input_paths = sorted(glob.glob(args.input_glob))

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))

    total_files = len(input_paths)
    chunk_size = math.ceil(total_files / num_tasks)
    start_idx = (task_id - 1) * chunk_size
    end_idx = min(start_idx + chunk_size, total_files)
    my_input_paths = input_paths[start_idx:end_idx]

    print(f"🚀 [Worker {task_id}] Processing {len(my_input_paths)}/{total_files} files.")
    print(f"   Agent token layout: 1 scale + 51 anchor + 204 motion = {N_AGENT_TOKENS} tokens/chunk")

    if not input_paths:
        raise FileNotFoundError(f"No files matched --input-glob={args.input_glob!r}")

    os.makedirs(args.agent_tokens_dir, exist_ok=True)

    grand_totals = {
        "files": 0, "videos": 0, "missing_agent_files": 0,
        "activities": 0, "avc_blocks": 0, "agents_injected": 0, "agent_misses": 0,
    }

    for input_path in my_input_paths:
        output_path = infer_output_path(input_path, args.output_prefix)

        if args.skip_existing and os.path.exists(output_path):
            tqdm.write(f"[SKIP] {output_path}")
            continue

        total_lines = count_lines(input_path)
        file_totals = {
            "videos": 0, "missing_agent_files": 0, "activities": 0,
            "avc_blocks": 0, "agents_injected": 0, "agent_misses": 0,
        }

        tqdm.write(f"[START] {input_path} -> {output_path}")

        with open(input_path, "r", encoding="utf-8") as in_f, open(
            output_path, "w", encoding="utf-8"
        ) as out_f:
            pbar = tqdm(total=total_lines, desc=os.path.basename(input_path), unit="line")
            for line_num, line in enumerate(in_f, start=1):
                pbar.update(1)
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    tqdm.write(
                        f"[WARN] Skipping invalid JSON in {input_path} at line {line_num}: {exc}"
                    )
                    continue

                stats = process_video_record(
                    record,
                    agent_tokens_dir=args.agent_tokens_dir,
                    tolerance=args.tolerance,
                    strict_avc_count=args.strict_avc_count,
                )

                file_totals["videos"] += 1
                file_totals["activities"] += stats["activities_seen"]
                file_totals["avc_blocks"] += stats["avc_blocks_seen"]
                file_totals["agents_injected"] += stats["agents_injected"]
                file_totals["agent_misses"] += stats["agent_misses"]
                if not stats["agent_file_found"]:
                    file_totals["missing_agent_files"] += 1

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            pbar.close()

        grand_totals["files"] += 1
        for key in (
            "videos", "missing_agent_files", "activities",
            "avc_blocks", "agents_injected", "agent_misses",
        ):
            grand_totals[key] += file_totals[key]

        tqdm.write(
            "[DONE] "
            f"{output_path} | videos={file_totals['videos']} | "
            f"missing_agent_files={file_totals['missing_agent_files']} | "
            f"activities={file_totals['activities']} | "
            f"avc_blocks={file_totals['avc_blocks']} | "
            f"agents_injected={file_totals['agents_injected']} | "
            f"agent_misses={file_totals['agent_misses']}"
        )

    print("=" * 80)
    print("Merge complete")
    print(f"Files processed     : {grand_totals['files']}")
    print(f"Videos processed    : {grand_totals['videos']}")
    print(f"Missing agent files : {grand_totals['missing_agent_files']}")
    print(f"Activities seen     : {grand_totals['activities']}")
    print(f"AVC blocks seen     : {grand_totals['avc_blocks']}")
    print(f"Agents injected     : {grand_totals['agents_injected']}")
    print(f"Agent misses        : {grand_totals['agent_misses']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
