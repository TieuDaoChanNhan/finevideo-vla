#!/usr/bin/env python3
"""
Merge Phase-5b per-joint XYZT agent tokens into training_ready JSONL files.

Reads self-describing token strings from phase5b output and injects them
as <agent>…</agent> blocks into the video_tokens field, immediately after
each corresponding <avc_lm>…</avc_lm> block.

Resulting token order per 8-frame chunk:
    <cosmos> … </cosmos> <avc_lm> … </avc_lm> <agent> <fps_30> <joint_0_x_N> … </agent>

Each <agent> block contains 409 tokens:
    1  fps token    : <fps_30>
    408 joint tokens: 8 frames × 17 joints × 3 (x, y, z)
                      <joint_{J}_x_{V}> <joint_{J}_y_{V}> <joint_{J}_z_{V}>

Frame alignment
───────────────
Phase-5b window_ids are absolute frame indices, always multiples of 8.
Training-ready avc_lm chunks start at activity.time_range_sec, which may
not be a multiple of 8.  We round the lookup to the nearest multiple of 8
and check ±8 neighbours — this covers the worst-case 7-frame offset when
an activity starts mid-chunk.
"""

import argparse
import glob
import json
import math
import os
import re
from typing import Dict, List, Tuple

from tqdm import tqdm

TARGET_FPS = 30
CHUNK_SIZE = 8
AVC_PATTERN = re.compile(r"<avc_lm>\s*.*?\s*</avc_lm>", re.DOTALL)


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def count_lines(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for n, _ in enumerate(f, start=1):
            pass
    return n


# ── Agent token loading ──────────────────────────────────────────────────────

def load_agent_dict(path: str) -> Dict[int, str]:
    """Load phase5b output → {window_id: '<agent> <fps_30> … </agent>'}."""
    agent_dict: Dict[int, str] = {}
    if not os.path.exists(path):
        return agent_dict

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            window_id = obj.get("window_id")
            token_str = obj.get("token_str")
            if window_id is None or not token_str:
                continue
            try:
                window_id = int(window_id)
            except (TypeError, ValueError):
                continue

            agent_dict[window_id] = f"<agent> {token_str} </agent>"

    return agent_dict


def find_agent_string(agent_dict: Dict[int, str], abs_frame: int) -> str:
    """
    Look up agent tokens for an avc_lm chunk at absolute frame index.

    Phase5b windows sit at multiples of 8.  An activity starting at e.g.
    frame 1501 produces avc_lm chunks at 1501, 1509, 1517 … which don't
    align with the phase5b grid (1496, 1504, 1512 …).  We snap to the
    nearest multiple of 8 and also check the two adjacent windows.
    """
    nearest = round(abs_frame / CHUNK_SIZE) * CHUNK_SIZE
    for candidate in (nearest, nearest - CHUNK_SIZE, nearest + CHUNK_SIZE):
        if candidate in agent_dict:
            return agent_dict[candidate]
    return ""


# ── Injection ─────────────────────────────────────────────────────────────────

def inject_agent_tokens(
    video_tokens: str,
    agent_insertions: List[str],
) -> Tuple[str, int]:
    """Insert agent strings after each <avc_lm>…</avc_lm> block."""
    if not video_tokens:
        return video_tokens, 0
    matches = list(AVC_PATTERN.finditer(video_tokens))
    if not matches:
        return video_tokens, 0

    parts: List[str] = []
    cursor = 0
    injected = 0
    for idx, m in enumerate(matches):
        parts.append(video_tokens[cursor:m.end()])
        agent_text = agent_insertions[idx] if idx < len(agent_insertions) else ""
        if agent_text:
            parts.append(" " + agent_text)
            injected += 1
        cursor = m.end()
    parts.append(video_tokens[cursor:])
    return "".join(parts), injected


def process_activity(
    activity: dict,
    agent_dict: Dict[int, str],
) -> Tuple[int, int, int]:
    """Merge agent tokens into one activity. Returns (avc_count, injected, misses)."""
    start_sec, end_sec = activity.get("time_range_sec", [0.0, 0.0])[:2]
    start_sec = safe_float(start_sec)
    end_sec = safe_float(end_sec)
    video_tokens = activity.get("video_tokens", "")

    avc_count = len(AVC_PATTERN.findall(video_tokens or ""))
    if avc_count == 0:
        return 0, 0, 0

    duration_frames = int(max(0.0, end_sec - start_sec) * TARGET_FPS)
    chunk_starts = list(range(0, duration_frames, CHUNK_SIZE)) if duration_frames > 0 else []
    abs_start = int(round(start_sec * TARGET_FPS))

    insertions: List[str] = []
    misses = 0
    for idx in range(avc_count):
        rel_start = chunk_starts[idx] if idx < len(chunk_starts) else idx * CHUNK_SIZE
        agent_text = find_agent_string(agent_dict, abs_start + rel_start)
        if not agent_text:
            misses += 1
        insertions.append(agent_text)

    merged, injected = inject_agent_tokens(video_tokens, insertions)
    activity["video_tokens"] = merged

    if injected > 0:
        activity["agent_token_order"] = "image_first"
        activity["agent_fps"] = TARGET_FPS

    return avc_count, injected, misses


def process_video(record: dict, agent_tokens_dir: str) -> dict:
    video_id = record.get("video_id", "")
    path = os.path.join(agent_tokens_dir, f"{video_id}_tokens.jsonl")
    agent_dict = load_agent_dict(path)

    stats = {
        "video_id": video_id,
        "agent_file_found": os.path.exists(path),
        "agent_windows": len(agent_dict),
        "activities": 0, "avc_blocks": 0, "injected": 0, "misses": 0,
    }

    for scene in record.get("scenes", []):
        if not isinstance(scene, dict):
            continue
        for activity in scene.get("activities", []):
            if not isinstance(activity, dict):
                continue
            stats["activities"] += 1
            avc, inj, miss = process_activity(activity, agent_dict)
            stats["avc_blocks"] += avc
            stats["injected"] += inj
            stats["misses"] += miss

    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge phase5b XYZT agent tokens into training_ready JSONL."
    )
    p.add_argument("--input-glob",
                    default="training_ready_rank_*.jsonl",
                    help="Glob for training_ready files.")
    p.add_argument("--agent-tokens-dir",
                    default=os.path.join("outputs", "agent_tokens_xyzt"),
                    help="Dir with <video_id>_tokens.jsonl from phase5b.")
    p.add_argument("--output-dir", default=None,
                    help="Output directory. Defaults to input file directory.")
    p.add_argument("--output-prefix", default="final_vla_xyzt",
                    help="Prefix for output files.")
    p.add_argument("--skip-existing", action="store_true",
                    help="Skip outputs that already exist.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = sorted(glob.glob(args.input_glob))

    if not input_paths:
        raise FileNotFoundError(f"No files matched: {args.input_glob!r}")

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))

    chunk = math.ceil(len(input_paths) / num_tasks)
    start = (task_id - 1) * chunk
    end = min(start + chunk, len(input_paths))
    my_paths = input_paths[start:end]

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Worker {task_id}/{num_tasks}] {len(my_paths)}/{len(input_paths)} files")

    grand = {
        "files": 0, "videos": 0, "no_agent_file": 0,
        "activities": 0, "avc_blocks": 0, "injected": 0, "misses": 0,
    }

    for in_path in my_paths:
        base = os.path.basename(in_path)
        out_dir = args.output_dir or os.path.dirname(in_path) or "."
        rank_match = re.search(r"_rank_(\d+)\.jsonl$", base)
        if rank_match:
            out_name = f"{args.output_prefix}_rank_{rank_match.group(1)}.jsonl"
        else:
            stem = os.path.splitext(base)[0]
            out_name = f"{args.output_prefix}_{stem}.jsonl"
        out_path = os.path.join(out_dir, out_name)

        if args.skip_existing and os.path.exists(out_path):
            tqdm.write(f"[SKIP] {out_path}")
            continue

        n_lines = count_lines(in_path)
        file_stats = {"videos": 0, "no_agent_file": 0, "activities": 0,
                       "avc_blocks": 0, "injected": 0, "misses": 0}

        with open(in_path, "r", encoding="utf-8") as fin, \
             open(out_path, "w", encoding="utf-8") as fout:
            pbar = tqdm(total=n_lines, desc=base, unit="vid")
            for line in fin:
                pbar.update(1)
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                st = process_video(record, args.agent_tokens_dir)
                file_stats["videos"] += 1
                file_stats["activities"] += st["activities"]
                file_stats["avc_blocks"] += st["avc_blocks"]
                file_stats["injected"] += st["injected"]
                file_stats["misses"] += st["misses"]
                if not st["agent_file_found"]:
                    file_stats["no_agent_file"] += 1

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            pbar.close()

        grand["files"] += 1
        for k in file_stats:
            grand[k] += file_stats[k]

        tqdm.write(f"[DONE] {out_path} | vids={file_stats['videos']} "
                   f"injected={file_stats['injected']} misses={file_stats['misses']}")

    print("=" * 70)
    print("Merge complete")
    for k, v in grand.items():
        print(f"  {k:20s}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    main()
