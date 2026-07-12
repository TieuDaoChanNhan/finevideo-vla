#!/usr/bin/env python3
"""
Final comparison: Qwen2.5-VL-3B vs SmolVLM2-2.2B, on the same real frames
(final anchor design: activity start + has_agent flip, min_gap_sec=5.0),
to decide the captioning model for production.

Usage:
    python tools/analysis/caption_final_compare.py --video-ids rieb8cBb2z8 BbVjTgLJYVQ eLPe7xp0jRw
"""

import argparse
import glob
import json
import os
import time

from caption_prototype import (
    extract_frame, select_anchor_points,
    load_model, caption_frame,
    load_model_smolvlm2, caption_frame_smolvlm2,
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
        pts = select_anchor_points(act["chunk_timing"], min_gap_sec=5.0)
        samples.append((vid, video_path, act, pts))

    print("=== Loading Qwen2.5-VL-3B ===")
    qwen_model, qwen_proc = load_model()
    print("\n=== Loading SmolVLM2-2.2B ===")
    smol_model, smol_proc = load_model_smolvlm2()

    qwen_times, smol_times = [], []
    for vid, video_path, act, pts in samples:
        print(f"\n[{vid}] text_prompt: {act.get('text_prompt', '')[:70]}")
        print(f"  {len(act['chunk_timing'])} chunks -> {len(pts)} anchor points")
        for pt in pts:
            ts = pt["start_sec"]
            frame = extract_frame(video_path, ts)
            q_cap, q_t = caption_frame(qwen_model, qwen_proc, frame)
            s_cap, s_t = caption_frame_smolvlm2(smol_model, smol_proc, frame)
            qwen_times.append(q_t)
            smol_times.append(s_t)
            print(f"  t={ts:.2f}s [has_agent={pt['has_agent']}]")
            print(f"    Qwen2.5-VL ({q_t:.1f}s): {q_cap}")
            print(f"    SmolVLM2   ({s_t:.1f}s): {s_cap}")

    n = len(qwen_times)
    print(f"\n=== Summary ({n} captions) ===")
    print(f"Qwen2.5-VL-3B avg: {sum(qwen_times)/n:.1f}s/caption")
    print(f"SmolVLM2-2.2B avg: {sum(smol_times)/n:.1f}s/caption")
    print(f"Speedup: {sum(qwen_times)/sum(smol_times):.1f}x")


if __name__ == "__main__":
    main()
