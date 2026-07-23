#!/usr/bin/env python3
"""
Merge Phase 5 adaptive PCHIP agent tokens + SNAC audio tokens into training_ready JSONL.

Reads agent tokens from phase5_adaptive_pchip output and (optionally) SNAC tokens
from snac_finevideo.py output, injecting both into video_tokens after each
<avc_lm>...</avc_lm> block.

Also adds a chunk_timing array and timing_meta to each activity so that
all 5 modalities (seed2, cosmos, avc_lm, agent, snac) have explicit timestamps.

Resulting token order per 8-frame chunk:
    <cosmos>...</cosmos> <avc_lm>...</avc_lm> [<agent>...</agent>] [<listen>...</listen>]

2026-07-23: SNAC wrapper changed from generic <snac> to <listen> -- FineVideo's
own audio is always ambient/scene sound the model describes, never a reply, so
it's always tagged <listen> (contrast with the roleplay dataset, which is
always a reply and always tagged <speak> -- see
data_prep/laion_emotional_roleplay/tokenize_snac.py).

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
COSMOS_PATTERN = re.compile(r"<cosmos>\s*.*?\s*</cosmos>", re.DOTALL)


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
    """Return '<listen> <snac_N> ... </listen>' for chunk_idx, or '' if empty.

    2026-07-23 decision: FineVideo's own audio is ambient/scene sound the
    model is describing as part of an assistant turn, not a reply -- always
    <listen> format (snac_finevideo.py's production default is
    encode_listen(), 3 tok/frame). Contrast with the roleplay dataset
    (data_prep/laion_emotional_roleplay/tokenize_snac.py), which is always a
    reply and always tagged <speak> regardless of encode format used.
    """
    tokens = snac_by_chunk.get(str(chunk_idx), [])
    return ("<listen> " + " ".join(tokens) + " </listen>") if tokens else ""


# ── Caption token loading ───────────────────────────────────────────────────

def load_caption_dict(path: str) -> Dict[str, Dict[str, str]]:
    """Load build_caption_dict.py output for one video.

    Input file has exactly one JSON line:
        {"video_id": ..., "captions_by_activity": {activity_id: {chunk_idx_str: "<caption> ... </caption>"}}}
    Returns {activity_id: {chunk_idx_str: "<caption> ... </caption>"}}, already tag-wrapped.
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline()
    except OSError:
        return {}
    line = line.strip()
    if not line:
        return {}
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return {}
    by_activity = obj.get("captions_by_activity")
    return by_activity if isinstance(by_activity, dict) else {}


# ── Speech (inline) token loading ───────────────────────────────────────────

def load_speech_dict(path: str) -> Dict[str, Dict[str, str]]:
    """Load extract_speech_segments.py output for one video -> {activity_id: {chunk_idx_str: text}}.

    Same per-video, multi-line-per-activity shape convention as load_snac_dict,
    except values are already tag-wrapped strings ('<speech> ... </speech>'),
    not raw token lists -- no extra wrapping needed on lookup.
    """
    speech: Dict[str, Dict[str, str]] = {}
    if not os.path.exists(path):
        return speech
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
                by_chunk = obj.get("speech_by_chunk")
                if act_id and isinstance(by_chunk, dict):
                    speech[act_id] = by_chunk
    except OSError:
        pass
    return speech


# ── Injection ────────────────────────────────────────────────────────────────

def inject_chunk_tokens(
    video_tokens: str,
    agent_insertions: List[str],
    snac_insertions: List[str],
    speech_insertions: List[str] = None,
    caption_insertions: List[str] = None,
) -> Tuple[str, int, int, int, int]:
    """
    Inject agent/SNAC/speech tokens after each <avc_lm> block, and caption
    tokens before each <cosmos> block, in one pass.

    Per chunk, the resulting token order is:
        [<caption>...</caption>] <cosmos>...</cosmos> <avc_lm>...</avc_lm>
        [<agent>...</agent>] [<snac>...</snac>] [<speech>...</speech>]

    caption/agent/snac/speech are all optional — empty string means nothing
    is injected for that chunk.

    Caption placement uses <cosmos> block positions, found independently of
    <avc_lm>. cosmos and avc_lm are 1:1 per chunk for the overwhelming
    majority of activities (verified empirically), but a small number
    (~2,753/372,385 activities, trailing-chunk edge cases) have one fewer
    cosmos block than avc_lm blocks. When idx has no corresponding cosmos
    match, that chunk's caption is silently skipped (counted as a miss by
    the caller) rather than misplaced or crashing.

    Returns (merged_tokens, n_agent_injected, n_snac_injected, n_speech_injected, n_caption_injected).
    """
    if not video_tokens:
        return video_tokens, 0, 0, 0, 0
    avc_matches = list(AVC_PATTERN.finditer(video_tokens))
    if not avc_matches:
        return video_tokens, 0, 0, 0, 0
    cosmos_matches = list(COSMOS_PATTERN.finditer(video_tokens))

    speech_insertions = speech_insertions or []
    caption_insertions = caption_insertions or []

    # Collect (position, text) insertion events, then apply them in one
    # left-to-right pass. Positions come from two independent regexes
    # (avc_lm end / cosmos start), so events must be globally sorted —
    # they are not already in position order across the two sources.
    events: List[Tuple[int, str]] = []
    inj_agent = inj_snac = inj_speech = inj_caption = 0

    for idx, m in enumerate(avc_matches):
        after_text = ""
        agent_text  = agent_insertions[idx]  if idx < len(agent_insertions)  else ""
        snac_text   = snac_insertions[idx]   if idx < len(snac_insertions)   else ""
        speech_text = speech_insertions[idx] if idx < len(speech_insertions) else ""
        if agent_text:
            after_text += " " + agent_text
            inj_agent += 1
        if snac_text:
            after_text += " " + snac_text
            inj_snac += 1
        if speech_text:
            after_text += " " + speech_text
            inj_speech += 1
        if after_text:
            events.append((m.end(), after_text))

        caption_text = caption_insertions[idx] if idx < len(caption_insertions) else ""
        if caption_text and idx < len(cosmos_matches):
            events.append((cosmos_matches[idx].start(), caption_text + " "))
            inj_caption += 1

    events.sort(key=lambda e: e[0])

    parts: List[str] = []
    cursor = 0
    for pos, text in events:
        parts.append(video_tokens[cursor:pos])
        parts.append(text)
        cursor = pos
    parts.append(video_tokens[cursor:])
    return "".join(parts), inj_agent, inj_snac, inj_speech, inj_caption


# ── Chunk timing builder ────────────────────────────────────────────────────

def build_chunk_timing(
    avc_count: int,
    abs_start: int,
    chunk_starts: List[int],
    agent_dict: Dict[int, str],
    video_tokens: str,
    snac_by_chunk: Dict = None,
    caption_by_chunk: Dict = None,
    speech_by_chunk_inline: Dict = None,
) -> List[dict]:
    """Build per-chunk timing array with modality presence flags.

    has_seed2 is attributed by *string position*: a <seed2> tag belongs to
    chunk i if it appears between the end of chunk (i-1)'s <avc_lm> block and
    the end of chunk i's <avc_lm> block, matching the temporal interleaving
    order tokens are actually written in by pipeline_video/pipeline.py
    (seed2 checked once per frame, before the frame is added to the
    cosmos/avc_lm buffer). A prior version compared the chunk index
    positionally against the *total* seed2 tag count (`i < len(seed2_matches)`),
    which since seed2 fires at 1fps vs. avc_lm's 3.75/sec meant has_seed2 was
    true for an artificial prefix of chunks and false for the rest of the
    activity, rather than reflecting per-chunk presence.

    has_cosmos/has_avc_lm are both hardcoded True: cosmos and avc_lm are
    encoded together at the same 8-frame cadence as this chunk loop itself
    (pipeline_video/pipeline.py), so every iteration already corresponds to
    a chunk that has both — no position check needed (verified empirically:
    0 flips across 2,563 real activities before this fix too).
    """
    timing = []
    avc_ends = [m.end() for m in AVC_PATTERN.finditer(video_tokens or "")]
    seed2_positions = [m.start() for m in re.finditer(r"<seed2>", video_tokens or "")]

    prev_end = 0
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
        has_caption = bool(caption_by_chunk and caption_by_chunk.get(str(i)))
        has_speech_inline = bool(speech_by_chunk_inline and speech_by_chunk_inline.get(str(i)))

        this_end = avc_ends[i] if i < len(avc_ends) else len(video_tokens or "")
        has_seed2 = any(prev_end <= p < this_end for p in seed2_positions)
        prev_end = this_end

        timing.append({
            "chunk_idx": i,
            "abs_frame": abs_frame,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "has_seed2": has_seed2,
            "has_cosmos": True,
            "has_avc_lm": True,
            "has_agent": has_agent,
            "has_snac": has_snac,
            "has_caption": has_caption,
            "has_speech_inline": has_speech_inline,
        })

    return timing


def process_activity(
    activity: dict,
    agent_dict: Dict[int, str],
    snac_by_chunk: Dict = None,
    caption_by_chunk: Dict = None,
    speech_by_chunk_inline: Dict = None,
) -> Tuple[int, int, int, int, int, int]:
    """Returns (avc_count, inj_agent, misses, inj_snac, inj_speech, inj_caption)."""
    start_sec, end_sec = activity.get("time_range_sec", [0.0, 0.0])[:2]
    start_sec = safe_float(start_sec)
    end_sec = safe_float(end_sec)
    video_tokens = activity.get("video_tokens", "")

    avc_count = len(AVC_PATTERN.findall(video_tokens or ""))
    if avc_count == 0:
        return 0, 0, 0, 0, 0, 0

    duration_frames = int(max(0.0, end_sec - start_sec) * TARGET_FPS)
    chunk_starts = list(range(0, duration_frames, CHUNK_SIZE)) if duration_frames > 0 else []
    abs_start = int(round(start_sec * TARGET_FPS))

    # Idempotency guard: this script is designed to run a SECOND time on top
    # of final_dataset_adaptive_v3 (which already has <agent>/<snac> injected
    # from the first v2->v3 run) to add captions/speech only. agent_dict /
    # snac_by_chunk are still needed as inputs on that second run (so
    # chunk_timing's has_agent/has_snac stay accurate), but must NOT be
    # re-injected into video_tokens a second time. Detecting existing tags
    # in the string (rather than relying on the caller to omit
    # --agent-tokens-dir/--snac-tokens-dir) protects against double
    # injection even if the CLI is invoked incorrectly.
    already_has_agent = "<agent>" in video_tokens
    already_has_snac  = "<listen>" in video_tokens  # 2026-07-23: FineVideo's snac wrapper is now <listen>, not <snac>
    already_has_speech_inline = "<speech>" in video_tokens
    already_has_caption       = "<caption>" in video_tokens

    agent_insertions: List[str] = []
    snac_insertions: List[str] = []
    speech_insertions: List[str] = []
    caption_insertions: List[str] = []
    misses = 0
    for idx in range(avc_count):
        rel_start = chunk_starts[idx] if idx < len(chunk_starts) else idx * CHUNK_SIZE
        agent_text = find_agent_string(agent_dict, abs_start + rel_start)
        if not agent_text and not already_has_agent:
            misses += 1
        agent_insertions.append(agent_text)
        snac_text = build_snac_insertion(snac_by_chunk, idx) if snac_by_chunk else ""
        snac_insertions.append(snac_text)
        speech_text = speech_by_chunk_inline.get(str(idx), "") if speech_by_chunk_inline else ""
        speech_insertions.append(speech_text)
        caption_text = caption_by_chunk.get(str(idx), "") if caption_by_chunk else ""
        caption_insertions.append(caption_text)

    merged, inj_agent, inj_snac, inj_speech, inj_caption = inject_chunk_tokens(
        video_tokens,
        [] if already_has_agent else agent_insertions,
        [] if already_has_snac else snac_insertions,
        [] if already_has_speech_inline else speech_insertions,
        [] if already_has_caption else caption_insertions,
    )
    activity["video_tokens"] = merged

    chunk_timing = build_chunk_timing(
        avc_count, abs_start, chunk_starts, agent_dict, video_tokens, snac_by_chunk,
        caption_by_chunk, speech_by_chunk_inline,
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
        "caption_rate": "anchor_points_before_cosmos",
        "speech_inline_rate": "asr_segment_start_snapped_to_chunk",
    }

    if inj_agent > 0:
        activity["agent_token_order"] = "image_first"
        activity["agent_fps"] = TARGET_FPS

    return avc_count, inj_agent, misses, inj_snac, inj_speech, inj_caption


def process_video(
    record: dict,
    agent_tokens_dir: str,
    snac_tokens_dir: str = "",
    captions_dir: str = "",
    speech_segments_dir: str = "",
) -> dict:
    video_id = record.get("video_id", "")
    agent_path = os.path.join(agent_tokens_dir, f"{video_id}_tokens.jsonl")
    agent_dict = load_agent_dict(agent_path)

    # Load SNAC/caption/speech data keyed by activity_id (empty dict if dir not set or file missing)
    snac_file: Dict[str, Dict] = {}
    if snac_tokens_dir:
        snac_path = os.path.join(snac_tokens_dir, f"{video_id}_snac.jsonl")
        snac_file = load_snac_dict(snac_path)

    caption_file: Dict[str, Dict] = {}
    if captions_dir:
        caption_path = os.path.join(captions_dir, f"{video_id}_captions_dict.jsonl")
        caption_file = load_caption_dict(caption_path)

    speech_file: Dict[str, Dict] = {}
    if speech_segments_dir:
        speech_path = os.path.join(speech_segments_dir, f"{video_id}_speech.jsonl")
        speech_file = load_speech_dict(speech_path)

    stats = {
        "video_id": video_id,
        "agent_file_found": os.path.exists(agent_path),
        "snac_file_found": bool(snac_tokens_dir and snac_file),
        "captions_file_found": bool(captions_dir and caption_file),
        "speech_file_found": bool(speech_segments_dir and speech_file),
        "agent_windows": len(agent_dict),
        "activities": 0, "avc_blocks": 0, "injected": 0, "misses": 0,
        "snac_injected": 0, "speech_injected": 0, "caption_injected": 0,
    }

    for scene in record.get("scenes", []):
        if not isinstance(scene, dict):
            continue
        for activity in scene.get("activities", []):
            if not isinstance(activity, dict):
                continue
            act_id = activity.get("activity_id", "")
            snac_by_chunk = snac_file.get(act_id, {})
            caption_by_chunk = caption_file.get(act_id, {})
            speech_by_chunk_inline = speech_file.get(act_id, {})
            stats["activities"] += 1
            avc, inj, miss, inj_snac, inj_speech, inj_caption = process_activity(
                activity, agent_dict, snac_by_chunk, caption_by_chunk, speech_by_chunk_inline,
            )
            stats["avc_blocks"] += avc
            stats["injected"] += inj
            stats["misses"] += miss
            stats["snac_injected"] += inj_snac
            stats["speech_injected"] += inj_speech
            stats["caption_injected"] += inj_caption

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
    p.add_argument("--captions-dir",
                    default="",
                    help="Dir with <video_id>_captions_dict.jsonl from build_caption_dict.py. "
                         "Leave empty to skip caption injection (backward-compatible).")
    p.add_argument("--speech-segments-dir",
                    default="",
                    help="Dir with <video_id>_speech.jsonl from extract_speech_segments.py. "
                         "Leave empty to skip inline speech injection (backward-compatible).")
    p.add_argument("--output-dir", default=None,
                    help="Output directory. Defaults to input file directory.")
    p.add_argument("--output-prefix", default="final_vla_adaptive",
                    help="Prefix for output files.")
    p.add_argument("--skip-existing", action="store_true",
                    help="Skip outputs that already exist.")
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                    help=f"Must match Phase 3/4/5's --window-size/--window-frames and "
                         f"pipeline.py's WINDOW_FRAMES. Default: {CHUNK_SIZE}. 2026-07-22: "
                         f"use 24 to match the wider cosmos chunk window -- see REPORT.md #38.")
    return p.parse_args()


def main() -> None:
    global CHUNK_SIZE
    args = parse_args()
    CHUNK_SIZE = args.chunk_size
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
        "snac_injected": 0, "speech_injected": 0, "caption_injected": 0,
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
                       "avc_blocks": 0, "injected": 0, "misses": 0,
                       "snac_injected": 0, "speech_injected": 0, "caption_injected": 0}

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

                st = process_video(
                    record, args.agent_tokens_dir, args.snac_tokens_dir,
                    args.captions_dir, args.speech_segments_dir,
                )
                file_stats["videos"] += 1
                file_stats["activities"] += st["activities"]
                file_stats["avc_blocks"] += st["avc_blocks"]
                file_stats["injected"] += st["injected"]
                file_stats["misses"] += st["misses"]
                file_stats["snac_injected"] += st["snac_injected"]
                file_stats["speech_injected"] += st["speech_injected"]
                file_stats["caption_injected"] += st["caption_injected"]
                if not st["agent_file_found"]:
                    file_stats["no_agent_file"] += 1

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            pbar.close()

        grand["files"] += 1
        for k in file_stats:
            grand[k] += file_stats[k]

        tqdm.write(f"[DONE] {out_path} | vids={file_stats['videos']} "
                   f"agent={file_stats['injected']} snac={file_stats['snac_injected']} "
                   f"speech={file_stats['speech_injected']} caption={file_stats['caption_injected']} "
                   f"misses={file_stats['misses']}")

    print("=" * 70)
    print("Merge complete")
    for k, v in grand.items():
        print(f"  {k:20s}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    main()
