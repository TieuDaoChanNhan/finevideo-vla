#!/usr/bin/env python3
"""
Build a per-video, per-segment caption+speech manifest from OmniVideo-100K's
scripts.jsonl, ready to be joined against Step A's video tokens once Step A
has actually been run (Step A determines real chunk/window boundaries in
seconds; this script only needs to run once, independent of that).

Verified before writing this (18/07/2026, 200-record sample):
  - segments[i].visual: list[{start_time, end_time, text}] -- always present
    when a segment exists (this IS the caption source)
  - segments[i].transcription: list[{start_time, end_time, text, speaker}] --
    present in 1613/1671 sampled segments (96.5%); empty on segments with no
    speech (e.g. a silent title-card intro) -- this IS the speech source,
    already speaker-labeled (same shape as the top-level label_speaker field)
  - all timestamps are "MM:SS" strings (OmniVideo-100K videos sampled here
    are all under a few minutes; an HH:MM:SS fallback is included for safety
    but not expected to trigger)

Does NOT touch train_oe_70k.jsonl/train_mcq_30k.jsonl (that's the separate
QA-only text stream, already flattened by flatten_qa_text.py -- no video
needed there).

Output: one JSONL, one line per video:
    {"video_id": ..., "video_path": ..., "duration": ...,
     "video_summary": ...,
     "segments": [{"start_sec": float, "end_sec": float,
                    "caption": "<joined visual text>",
                    "speech": "<joined 'Speaker: text' lines>"}, ...]}

Usage:
    python3 data_prep/omnivideo_100k/dataset_prep/build_segment_captions.py
"""
import json
import os
import re

SRC_FILE = "/p/data1/mmlaion/shared/vla/omnivideo_100k/scripts.jsonl"
OUT_PATH = "/p/data1/mmlaion/shared/vla/omnivideo_100k_flat/omnivideo_100k_segment_captions.jsonl"

TIME_RE = re.compile(r"^(?:(\d+):)?(\d+):(\d+)$")


def parse_time(s):
    """'MM:SS' or 'HH:MM:SS' -> float seconds. Returns None if unparseable."""
    if not s:
        return None
    m = TIME_RE.match(s.strip())
    if not m:
        return None
    h, mi, sec = m.groups()
    h = int(h) if h else 0
    return h * 3600 + int(mi) * 60 + int(sec)


def build_segment(seg):
    start_sec = parse_time(seg.get("start_time"))
    end_sec = parse_time(seg.get("end_time"))

    visual_parts = [v["text"] for v in seg.get("visual", []) if v.get("text")]
    caption = " ".join(visual_parts)

    speech_lines = []
    for t in seg.get("transcription", []):
        text = t.get("text")
        if not text:
            continue
        speaker = t.get("speaker")
        speech_lines.append(f"{speaker}: {text}" if speaker else text)
    speech = " ".join(speech_lines)

    return {
        "start_sec": start_sec,
        "end_sec": end_sec,
        "caption": caption,
        "speech": speech,
    }


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    n_videos = 0
    n_segments = 0
    n_bad_timestamps = 0

    with open(SRC_FILE) as in_f, open(OUT_PATH, "w") as out_f:
        for line in in_f:
            rec = json.loads(line)
            segments_out = []
            for seg in rec.get("segments", []):
                built = build_segment(seg)
                if built["start_sec"] is None or built["end_sec"] is None:
                    n_bad_timestamps += 1
                    continue
                segments_out.append(built)
                n_segments += 1

            out_f.write(json.dumps({
                "video_id": rec["video_id"],
                "video_path": rec["video_path"],
                "duration": rec.get("duration"),
                "video_summary": rec.get("video_summary"),
                "segments": segments_out,
            }, ensure_ascii=False) + "\n")
            n_videos += 1

    print(f"videos: {n_videos}, segments: {n_segments}, "
          f"unparseable timestamps skipped: {n_bad_timestamps} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
