#!/usr/bin/env python3
"""
Merge Phase 5 adaptive PCHIP agent tokens + SNAC audio tokens into training_ready JSONL.

Reads agent tokens from phase5_adaptive_pchip output and (optionally) SNAC tokens
from snac_finevideo.py output, injecting both into video_tokens after each
<avc_lm>...</avc_lm> block.

Also adds a chunk_timing array and timing_meta to each activity so that
all 5 modalities (seed2, cosmos, avc_lm, agent, snac) have explicit timestamps.

Resulting token order per 8-frame chunk:
    <cosmos>...</cosmos> <avc_lm>...</avc_lm> [<agent>...</agent>] [<snac>...</snac>]

SNAC alignment:
    SNAC listen format = 37.5 tokens/sec. Each 8-frame chunk at 30fps = 0.267s.
    → ~9–10 SNAC tokens per chunk (3–4 base frames × 3 tokens/frame).
    snac_finevideo.py encodes the full activity once (preserving audio context)
    then splits tokens evenly across chunks, snapping to 3-token boundaries
    (1 base frame = 3 tokens: codes[0], codes[1][2i], codes[1][2i+1]).

Frame alignment
---------------
Phase 5 window_ids are absolute frame indices, always multiples of 8.
Training-ready avc_lm chunks start at activity.time_range_sec, which may
not be a multiple of 8.  We snap to the nearest multiple of 8 and check
+/-8 neighbours.
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


# ── Helpers ──────────────────────────────────────────────────────────────────

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


# ── Agent token loading ─────────────────────────────────────────────────────

def load_agent_dict(path: str) -> Dict[int, str]:
    """Load phase5 adaptive output -> {window_id: '<agent> <fps_30> ... </agent>'}."""
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
    nearest = round(abs_frame / CHUNK_SIZE) * CHUNK_SIZE
    for candidate in (nearest, nearest - CHUNK_SIZE, nearest + CHUNK_SIZE):
        if candidate in agent_dict:
            return agent_dict[candidate]
    return ""


# ── SNAC token loading ────────────────────────────────────────────────────────

def load_snac_dict(path: str) -> Dict[str, Dict]:
    """
    Load snac_finevideo output → {activity_id: snac_by_chunk}.
    snac_by_chunk: {"0": ["<snac_X>", ...], "1": [...], ...}
    """
    snac: Dict[str, Dict] = {}
    if not os.path.exists(path):
        return snac
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                act_id = obj.get("activity_id", "")
                by_chunk = obj.get("snac_by_chunk")
                if act_id and isinstance(by_chunk, dict):
                    snac[act_id] = by_chunk
    except OSError:
        pass
    return snac


def build_snac_insertion(snac_by_chunk: Dict, chunk_idx: int) -> str:
    """Return '<snac> <snac_N> ... </snac>' for chunk_idx, or '' if empty."""
    tokens = snac_by_chunk.get(str(chunk_idx), [])
    return ("<snac> " + " ".join(tokens) + " </snac>") if tokens else ""


# ── Injection ────────────────────────────────────────────────────────────────

def inject_chunk_tokens(
    video_tokens: str,
    agent_insertions: List[str],
    snac_insertions: List[str],
) -> Tuple[str, int, int]:
    """
    Inject agent and SNAC tokens after each <avc_lm> block in one pass.

    Per chunk, the resulting token order is:
        <cosmos>...</cosmos> <avc_lm>...</avc_lm> [<agent>...</agent>] [<snac>...</snac>]

    Both agent and snac are optional — empty string means nothing is injected.
    Returns (merged_tokens, n_agent_injected, n_snac_injected).
    """
    if not video_tokens:
        return video_tokens, 0, 0
    matches = list(AVC_PATTERN.finditer(video_tokens))
    if not matches:
        return video_tokens, 0, 0

    parts: List[str] = []
    cursor = 0
    inj_agent = inj_snac = 0
    for idx, m in enumerate(matches):
        parts.append(video_tokens[cursor:m.end()])
        agent_text = agent_insertions[idx] if idx < len(agent_insertions) else ""
        snac_text  = snac_insertions[idx]  if idx < len(snac_insertions)  else ""
        if agent_text:
            parts.append(" " + agent_text)
            inj_agent += 1
        if snac_text:
            parts.append(" " + snac_text)
            inj_snac += 1
        cursor = m.end()
    parts.append(video_tokens[cursor:])
    return "".join(parts), inj_agent, inj_snac


# ── Chunk timing builder ────────────────────────────────────────────────────

def build_chunk_timing(
    avc_count: int,
    abs_start: int,
    chunk_starts: List[int],
    agent_dict: Dict[int, str],
    video_tokens: str,
    snac_by_chunk: Dict = None,
) -> List[dict]:
    """Build per-chunk timing array with modality presence flags."""
    timing = []
    seed2_matches = list(re.finditer(r"<seed2>", video_tokens or ""))
    cosmos_matches = list(re.finditer(r"<cosmos>", video_tokens or ""))

    for i in range(avc_count):
        rel_start = chunk_starts[i] if i < len(chunk_starts) else i * CHUNK_SIZE
        abs_frame = abs_start + rel_start
        start_sec = round(abs_frame / TARGET_FPS, 4)
        end_sec = round((abs_frame + CHUNK_SIZE) / TARGET_FPS, 4)

        nearest = round(abs_frame / CHUNK_SIZE) * CHUNK_SIZE
        has_agent = any(
            c in agent_dict
            for c in (nearest, nearest - CHUNK_SIZE, nearest + CHUNK_SIZE)
        )
        has_snac = bool(snac_by_chunk and snac_by_chunk.get(str(i)))

        timing.append({
            "chunk_idx": i,
            "abs_frame": abs_frame,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "has_seed2": i < len(seed2_matches),
            "has_cosmos": i < len(cosmos_matches),
            "has_avc_lm": True,
            "has_agent": has_agent,
            "has_snac": has_snac,
        })

    return timing


def process_activity(
    activity: dict,
    agent_dict: Dict[int, str],
    snac_by_chunk: Dict = None,
) -> Tuple[int, int, int, int]:
    """Returns (avc_count, inj_agent, misses, inj_snac)."""
    start_sec, end_sec = activity.get("time_range_sec", [0.0, 0.0])[:2]
    start_sec = safe_float(start_sec)
    end_sec = safe_float(end_sec)
    video_tokens = activity.get("video_tokens", "")

    avc_count = len(AVC_PATTERN.findall(video_tokens or ""))
    if avc_count == 0:
        return 0, 0, 0, 0

    duration_frames = int(max(0.0, end_sec - start_sec) * TARGET_FPS)
    chunk_starts = list(range(0, duration_frames, CHUNK_SIZE)) if duration_frames > 0 else []
    abs_start = int(round(start_sec * TARGET_FPS))

    agent_insertions: List[str] = []
    snac_insertions: List[str] = []
    misses = 0
    for idx in range(avc_count):
        rel_start = chunk_starts[idx] if idx < len(chunk_starts) else idx * CHUNK_SIZE
        agent_text = find_agent_string(agent_dict, abs_start + rel_start)
        if not agent_text:
            misses += 1
        agent_insertions.append(agent_text)
        snac_text = build_snac_insertion(snac_by_chunk, idx) if snac_by_chunk else ""
        snac_insertions.append(snac_text)

    merged, inj_agent, inj_snac = inject_chunk_tokens(video_tokens, agent_insertions, snac_insertions)
    activity["video_tokens"] = merged

    chunk_timing = build_chunk_timing(
        avc_count, abs_start, chunk_starts, agent_dict, video_tokens, snac_by_chunk,
    )
    activity["chunk_timing"] = chunk_timing
    activity["timing_meta"] = {
        "video_fps": TARGET_FPS,
        "chunk_frames": CHUNK_SIZE,
        "seed2_rate": "1fps_keyframe",
        "cosmos_rate": "every_8_frames",
        "avc_lm_rate": "every_8_frames",
        "agent_rate": "every_8_frames_adaptive_pchip",
        "snac_rate": "37.5_tokens_per_sec_listen_format",
    }

    if inj_agent > 0:
        activity["agent_token_order"] = "image_first"
        activity["agent_fps"] = TARGET_FPS

    return avc_count, inj_agent, misses, inj_snac


def process_video(record: dict, agent_tokens_dir: str, snac_tokens_dir: str = "") -> dict:
    video_id = record.get("video_id", "")
    agent_path = os.path.join(agent_tokens_dir, f"{video_id}_tokens.jsonl")
    agent_dict = load_agent_dict(agent_path)

    # Load SNAC data keyed by activity_id (empty dict if dir not set or file missing)
    snac_file: Dict[str, Dict] = {}
    if snac_tokens_dir:
        snac_path = os.path.join(snac_tokens_dir, f"{video_id}_snac.jsonl")
        snac_file = load_snac_dict(snac_path)

    stats = {
        "video_id": video_id,
        "agent_file_found": os.path.exists(agent_path),
        "snac_file_found": bool(snac_tokens_dir and snac_file),
        "agent_windows": len(agent_dict),
        "activities": 0, "avc_blocks": 0, "injected": 0, "misses": 0, "snac_injected": 0,
    }

    for scene in record.get("scenes", []):
        if not isinstance(scene, dict):
            continue
        for activity in scene.get("activities", []):
            if not isinstance(activity, dict):
                continue
            act_id = activity.get("activity_id", "")
            snac_by_chunk = snac_file.get(act_id, {})
            stats["activities"] += 1
            avc, inj, miss, inj_snac = process_activity(activity, agent_dict, snac_by_chunk)
            stats["avc_blocks"] += avc
            stats["injected"] += inj
            stats["misses"] += miss
            stats["snac_injected"] += inj_snac

    return stats


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge Phase 5 adaptive PCHIP agent tokens into training_ready JSONL."
    )
    p.add_argument("--input-glob",
                    default="training_ready_rank_*.jsonl",
                    help="Glob for training_ready files.")
    p.add_argument("--agent-tokens-dir",
                    default=os.path.join("outputs", "agent_tokens_adaptive"),
                    help="Dir with <video_id>_tokens.jsonl from Phase 5 adaptive.")
    p.add_argument("--snac-tokens-dir",
                    default="",
                    help="Dir with <video_id>_snac.jsonl from snac_finevideo.py. "
                         "Leave empty to skip SNAC injection (backward-compatible).")
    p.add_argument("--output-dir", default=None,
                    help="Output directory. Defaults to input file directory.")
    p.add_argument("--output-prefix", default="final_vla_adaptive",
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
        "activities": 0, "avc_blocks": 0, "injected": 0, "misses": 0, "snac_injected": 0,
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
                       "avc_blocks": 0, "injected": 0, "misses": 0, "snac_injected": 0}

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

                st = process_video(record, args.agent_tokens_dir, args.snac_tokens_dir)
                file_stats["videos"] += 1
                file_stats["activities"] += st["activities"]
                file_stats["avc_blocks"] += st["avc_blocks"]
                file_stats["injected"] += st["injected"]
                file_stats["misses"] += st["misses"]
                file_stats["snac_injected"] += st["snac_injected"]
                if not st["agent_file_found"]:
                    file_stats["no_agent_file"] += 1

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            pbar.close()

        grand["files"] += 1
        for k in file_stats:
            grand[k] += file_stats[k]

        tqdm.write(f"[DONE] {out_path} | vids={file_stats['videos']} "
                   f"agent={file_stats['injected']} snac={file_stats['snac_injected']} "
                   f"misses={file_stats['misses']}")

    print("=" * 70)
    print("Merge complete")
    for k, v in grand.items():
        print(f"  {k:20s}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    main()
