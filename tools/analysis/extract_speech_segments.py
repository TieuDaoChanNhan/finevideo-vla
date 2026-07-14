#!/usr/bin/env python3
"""
Speech-segment extraction: map FineVideo's pre-computed timecoded_text_to_speech
(YouTube-Commons ASR, no new ASR compute needed) onto the same 8-frame chunk
grid used by phase6_merge_adaptive.py, ready for direct consumption by a new
--speech-dir loader in that script (same per-video shape convention as SNAC's
{activity_id: {chunk_idx_str: text}} output).

Design decisions (agreed with the user this session):
  - Segment -> activity assignment: single activity, MAXIMUM overlap only
    (not Step A's old multi-assign-if-overlap>=0.2s logic, which could
    duplicate the same sentence into two adjacent activities).
  - Segment -> chunk snap: attach text ONCE, at the chunk containing the
    segment's start time (like a subtitle) -- not repeated across every
    chunk a multi-second segment spans (segments average ~3-4s; chunks are
    only ~0.267s wide, so a segment typically spans ~11 chunks).
  - Quality filter: skip segments where >garble-threshold fraction of
    characters are non-ASCII/non-Latin (default 15%) -- addresses a
    confirmed ASR-failure-on-non-English-audio garbling mode (video
    vd6hr_AtYtQ.mp4 producing Hindi/gibberish mixed text).
  - Same-chunk collisions: empirically ~0.03% and almost entirely a
    byproduct of the garble case above (already filtered) -- trivial
    space-join fallback, no sophisticated merge logic needed.

Fetch strategy: raw timecoded_text_to_speech is NOT cached anywhere locally
(activity["speech_transcript"] in final_dataset_adaptive_v3 already collapsed
away per-segment timestamps) -- must re-pull from the HF Hub parquet files
(json column only, no mp4 blobs). To avoid re-fetching the same shard
repeatedly, this script does a two-phase run per invocation: (1) load all of
this worker's assigned chunk_timing/time_range_sec data into memory and
resolve every video_id's target shard via the manifest from
build_video_shard_manifest.py, (2) fetch each distinct shard exactly once (in
shard-index order) and process every video_id that resolves to it.

Usage:
    python tools/analysis/extract_speech_segments.py --skip-existing
    python tools/analysis/extract_speech_segments.py --video-ids abc123,def456   # quick test
"""

import argparse
import glob
import json
import math
import os
import re

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

INPUT_GLOB_DEFAULT = (
    "/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/"
    "final_dataset_adaptive_v3/*.jsonl"
)
MANIFEST_DEFAULT = os.path.join("outputs", "speech_extraction", "video_id_to_shard.json")
OUTPUT_DIR_DEFAULT = os.path.join("outputs", "speech_extraction", "speech_segments")
NUM_SHARDS = 1357
REPO = "HuggingFaceFV/finevideo"


def get_token():
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_path):
        with open(token_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None


def time_to_seconds(time_str: str) -> float:
    """Mirror pipeline_video/pipeline.py's time_to_seconds() exactly."""
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def calculate_overlap(start1, end1, start2, end2) -> float:
    """Mirror pipeline_video/pipeline.py's calculate_overlap() exactly (overlap duration in seconds)."""
    return max(0.0, min(end1, end2) - max(start1, start2))


def derive_video_id(row: dict) -> str:
    """Mirror pipeline_video/pipeline.py's parse_video_metadata() id logic exactly."""
    video_id = (row.get("original_video_filename") or "unknown").replace(".mp4", "")
    if video_id == "unknown":
        video_id = (row.get("youtube_title") or "video").replace(" ", "_").lower()
    return video_id


def is_garbled(text: str, threshold: float, min_unique_ratio: float = 0.15) -> bool:
    """Two cheap heuristics for known ASR failure modes:
    1. Non-ASCII/non-Latin character fraction above threshold (script-garbling,
       e.g. video vd6hr_AtYtQ producing Devanagari/Tamil/Cyrillic mixed text).
    2. Degenerate repetition loops (e.g. "...adadadadadadad...", hundreds of
       characters of a repeated 2-4 char run) -- these are pure ASCII so the
       non-ASCII check above misses them; unique-character ratio catches them
       since a real spoken sentence has much higher character diversity.
    """
    if not text:
        return True
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if (non_ascii / len(text)) > threshold:
        return True
    if len(text) >= 40 and (len(set(text)) / len(text)) < min_unique_ratio:
        return True
    return False


def snap_to_chunk(chunk_timing, s_start: float) -> dict:
    """Find the chunk containing s_start; fallback to nearest by absolute diff."""
    for c in chunk_timing:
        if c["start_sec"] <= s_start < c["end_sec"]:
            return c
    return min(chunk_timing, key=lambda c: abs(c["start_sec"] - s_start))


def assign_segment_to_activity(s_start, s_end, activities):
    """Return the activity dict with maximum overlap duration, or None if zero overlap everywhere."""
    best_act, best_overlap = None, 0.0
    for act in activities:
        a_start, a_end = act.get("time_range_sec", [0.0, 0.0])[:2]
        ov = calculate_overlap(s_start, s_end, a_start, a_end)
        if ov > best_overlap:
            best_overlap = ov
            best_act = act
    return best_act


def load_activities_needing_speech(input_paths, allowlist=None):
    """Scan assigned final_dataset_adaptive_v3 shard files -> {video_id: [thin activity dicts with chunk_timing]}.

    Two memory-safety measures (each shard file is ~4.5GB, 160 files = 663GB total):
      1. Only keep the 3 fields actually used downstream (activity_id,
         chunk_timing, time_range_sec) -- NOT the full activity dict, which
         also carries `video_tokens` (the full token string, can be
         hundreds of KB per activity). Retaining full dicts for every
         video across all assigned files was observed to balloon RSS to
         90+ GB even for a 2-video `--video-ids` test against the default
         (all 160 file) glob.
      2. If an allowlist is given (quick `--video-ids` testing), skip
         non-matching videos before touching their activities at all.
    """
    by_video = {}
    for in_path in input_paths:
        with open(in_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                video_id = d.get("video_id", "")
                if allowlist is not None and video_id not in allowlist:
                    continue
                acts = []
                for scene in d.get("scenes", []):
                    for act in scene.get("activities", []):
                        ct = act.get("chunk_timing")
                        if ct:
                            acts.append({
                                "activity_id": act.get("activity_id", ""),
                                "chunk_timing": ct,
                                "time_range_sec": act.get("time_range_sec", [0.0, 0.0]),
                            })
                if acts:
                    by_video[video_id] = acts
    return by_video


def process_segments(raw_segments, activities, garble_threshold):
    """Returns ({activity_id: {chunk_idx_str: '<speech>...</speech>'}}, stats) for one video."""
    stats = {"segments": 0, "garbled": 0, "no_overlap": 0, "collisions": 0}
    by_activity = {}

    for seg in raw_segments:
        stats["segments"] += 1
        text = (seg.get("text") or "").strip()
        if not text or is_garbled(text, garble_threshold):
            stats["garbled"] += 1
            continue

        s_start = time_to_seconds(seg.get("start", ""))
        s_end = time_to_seconds(seg.get("end", ""))

        act = assign_segment_to_activity(s_start, s_end, activities)
        if act is None:
            stats["no_overlap"] += 1
            continue

        chunk = snap_to_chunk(act["chunk_timing"], s_start)
        chunk_key = str(chunk["chunk_idx"])
        act_id = act.get("activity_id", "")
        by_chunk = by_activity.setdefault(act_id, {})

        if chunk_key in by_chunk:
            stats["collisions"] += 1
            suffix = " </speech>"
            by_chunk[chunk_key] = by_chunk[chunk_key][: -len(suffix)] + " " + text + suffix
        else:
            by_chunk[chunk_key] = f"<speech> {text} </speech>"

    return by_activity, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract FineVideo speech segments, mapped onto chunk_timing.")
    p.add_argument("--input-glob", default=INPUT_GLOB_DEFAULT)
    p.add_argument("--manifest", default=MANIFEST_DEFAULT)
    p.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--garble-threshold", type=float, default=0.15)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--video-ids", default="", help="Comma-separated video_id allowlist, for quick testing.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = sorted(glob.glob(args.input_glob))
    if not input_paths:
        raise FileNotFoundError(f"No files matched: {args.input_glob!r}")

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    chunk_n = math.ceil(len(input_paths) / num_tasks)
    start = (task_id - 1) * chunk_n
    end = min(start + chunk_n, len(input_paths))
    my_paths = input_paths[start:end]
    print(f"[Worker {task_id}/{num_tasks}] {len(my_paths)}/{len(input_paths)} shard files")

    os.makedirs(args.output_dir, exist_ok=True)

    # Phase 1: load all activities needing speech for this worker's shard files.
    # Allowlist (if any) is applied *during* the scan, not after, so a quick
    # `--video-ids` test doesn't have to materialize every other video's
    # activities first (see load_activities_needing_speech's docstring).
    allowlist = set(x.strip() for x in args.video_ids.split(",") if x.strip()) or None
    by_video = load_activities_needing_speech(my_paths, allowlist=allowlist)
    print(f"[Worker {task_id}/{num_tasks}] {len(by_video)} videos with chunk_timing"
          + (f" (allowlist={sorted(allowlist)})" if allowlist else ""))

    if args.skip_existing:
        by_video = {
            vid: acts for vid, acts in by_video.items()
            if not os.path.exists(os.path.join(args.output_dir, f"{vid}_speech.jsonl"))
        }
        print(f"  after --skip-existing: {len(by_video)} videos left")

    # Phase 2: resolve video_id -> shard_idx, group so each shard is fetched once.
    videos_by_shard = {}
    missing_from_manifest = 0
    for vid in by_video:
        shard_idx = manifest.get(vid)
        if shard_idx is None:
            missing_from_manifest += 1
            continue
        videos_by_shard.setdefault(shard_idx, []).append(vid)
    print(f"[Worker {task_id}/{num_tasks}] {len(videos_by_shard)} distinct shards to fetch "
          f"({missing_from_manifest} videos missing from manifest)")

    token = get_token()
    grand = {"videos": 0, "activities_with_speech": 0, "segments": 0,
             "garbled": 0, "no_overlap": 0, "collisions": 0}

    for shard_idx in sorted(videos_by_shard):
        filename = f"data/train-{shard_idx:05d}-of-{NUM_SHARDS:05d}.parquet"
        # Download to local HF cache first (respects $HF_HOME) instead of reading
        # via HfFileSystem's remote streaming, which was observed to balloon RSS
        # to 90+ GB on a single ~490MB shard (unbounded growth, no sign of leveling
        # off) -- likely inefficient buffering in the fsspec/HTTP layer. A cached
        # local file also means repeat runs / other videos in the same shard are free.
        local_path = hf_hub_download(
            repo_id=REPO, repo_type="dataset", filename=filename, token=token,
        )
        tbl = pq.read_table(local_path, columns=["json"])
        rows = tbl.column("json").to_pylist()
        raw_by_id = {derive_video_id(r): r for r in rows}

        for vid in videos_by_shard[shard_idx]:
            raw_row = raw_by_id.get(vid)
            if raw_row is None:
                continue
            raw_segments = raw_row.get("timecoded_text_to_speech", [])
            if not raw_segments:
                continue

            by_activity, stats = process_segments(raw_segments, by_video[vid], args.garble_threshold)
            if not by_activity:
                continue

            out_path = os.path.join(args.output_dir, f"{vid}_speech.jsonl")
            with open(out_path, "w", encoding="utf-8") as fout:
                for act_id, speech_by_chunk in by_activity.items():
                    fout.write(json.dumps({
                        "video_id": vid,
                        "activity_id": act_id,
                        "speech_by_chunk": speech_by_chunk,
                    }) + "\n")

            grand["videos"] += 1
            grand["activities_with_speech"] += len(by_activity)
            for k in ("segments", "garbled", "no_overlap", "collisions"):
                grand[k] += stats[k]

    print(f"[Worker {task_id}/{num_tasks}] DONE: {grand['videos']} videos, "
          f"{grand['activities_with_speech']} activities with speech, "
          f"{grand['segments']} segments ({grand['garbled']} garbled/skipped, "
          f"{grand['no_overlap']} no-overlap, {grand['collisions']} same-chunk collisions)")


if __name__ == "__main__":
    main()
