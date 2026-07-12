#!/usr/bin/env python3
"""
Compare Qwen2.5-VL-3B vs Florence-2-base for the captioning pipeline:
speed and caption quality, on the same real frames, selected using the
final anchor design (select_anchor_points: activity start + has_agent flips).

Usage:
    python tools/analysis/caption_model_compare.py --video-ids eLPe7xp0jRw rieb8cBb2z8
"""

import argparse
import glob
import json
import os

from caption_prototype import (
    extract_frame, select_anchor_points,
    load_model, caption_frame,
    load_model_florence2, caption_frame_florence2,
)

MERGED_DIR = "/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/final_dataset_adaptive_v3"
VIDEOS_DIR = "/p/data1/mmlaion/shared/nguyen38/data/videos_staging"


def find_activity(video_id):
    for fp in sorted(glob.glob(os.path.join(MERGED_DIR, "*.jsonl"))):
        with open(fp) as f:
            for line in f:
                d = json.loads(line)
                if d["video_id"] != video_id:
                    continue
                for scene in d["scenes"]:
                    for act in scene["activities"]:
                        if act.get("chunk_timing"):
                            return act
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-ids", nargs="+", required=True)
    args = parser.parse_args()

    samples = []
    for vid in args.video_ids:
        video_path = os.path.join(VIDEOS_DIR, f"{vid}.mp4")
        act = find_activity(vid)
        if act is None or not os.path.exists(video_path):
            print(f"[{vid}] SKIP: missing activity or video")
            continue
        pts = select_anchor_points(act["chunk_timing"])
        samples.append((vid, video_path, act, pts))

    print("=== Loading Qwen2.5-VL-3B ===")
    qwen_model, qwen_proc = load_model()
    print("\n=== Loading Florence-2-base ===")
    flor_model, flor_proc = load_model_florence2()

    qwen_times, flor_times = [], []
    for vid, video_path, act, pts in samples:
        print(f"\n[{vid}] text_prompt: {act.get('text_prompt', '')[:70]}")
        print(f"  {len(act['chunk_timing'])} chunks -> {len(pts)} anchor points (start + agent-flip)")
        for pt in pts:
            ts = pt["start_sec"]
            frame = extract_frame(video_path, ts)
            q_cap, q_t = caption_frame(qwen_model, qwen_proc, frame)
            f_cap, f_t = caption_frame_florence2(flor_model, flor_proc, frame)
            qwen_times.append(q_t)
            flor_times.append(f_t)
            print(f"  t={ts:.2f}s [has_agent={pt['has_agent']}]")
            print(f"    Qwen2.5-VL   ({q_t:.1f}s): {q_cap}")
            print(f"    Florence-2   ({f_t:.1f}s): {f_cap}")

    n = len(qwen_times)
    print(f"\n=== Summary ({n} captions) ===")
    print(f"Qwen2.5-VL-3B avg: {sum(qwen_times)/n:.1f}s/caption")
    print(f"Florence-2-base avg: {sum(flor_times)/n:.1f}s/caption")
    print(f"Speedup: {sum(qwen_times)/sum(flor_times):.1f}x")


if __name__ == "__main__":
    main()
