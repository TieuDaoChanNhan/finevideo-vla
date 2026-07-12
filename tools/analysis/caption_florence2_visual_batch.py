#!/usr/bin/env python3
"""
Larger Florence-2 QA batch: saves frames + captions for review, using the
final anchor design (select_anchor_points: activity start + has_agent flip).
Deliberately over-samples activities that DO have an agent-flip (real
person appear/disappear event) so the sample isn't dominated by "just the
opening frame" cases.

Usage:
    python tools/analysis/caption_florence2_visual_batch.py --num-videos 10
"""

import argparse
import glob
import json
import os

from PIL import Image

from caption_prototype import (
    extract_frame, select_anchor_points, load_model_florence2, caption_frame_florence2,
)

MERGED_DIR = "/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/final_dataset_adaptive_v3"
VIDEOS_DIR = "/p/data1/mmlaion/shared/nguyen38/data/videos_staging"
OUT_DIR = "/p/data1/mmlaion/nguyen38/3d-human-pose/logs/caption_frames_florence2"


def collect_samples(num_videos, min_gap_sec=2.0):
    """Prefer activities with an agent-flip (real content event), fall back
    to start-only activities to fill the quota."""
    with_agent_flip = []
    start_only = []
    seen_videos = set()

    files = sorted(glob.glob(os.path.join(MERGED_DIR, "*.jsonl")))
    for fp in files:
        if len(with_agent_flip) >= num_videos:
            break
        with open(fp) as f:
            for line in f:
                d = json.loads(line)
                vid = d["video_id"]
                if vid in seen_videos:
                    continue
                video_path = os.path.join(VIDEOS_DIR, f"{vid}.mp4")
                if not os.path.exists(video_path):
                    continue
                for scene in d["scenes"]:
                    for act in scene["activities"]:
                        ct = act.get("chunk_timing") or []
                        if not ct:
                            continue
                        pts = select_anchor_points(ct, min_gap_sec=min_gap_sec)
                        seen_videos.add(vid)
                        entry = {
                            "video_id": vid, "video_path": video_path,
                            "text_prompt": act.get("text_prompt", ""), "pts": pts,
                        }
                        if len(pts) > 1:
                            with_agent_flip.append(entry)
                        else:
                            start_only.append(entry)
                        break
                    break
    random_fill = start_only[: max(0, num_videos - len(with_agent_flip))]
    return with_agent_flip[:num_videos] + random_fill


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-videos", type=int, default=10)
    parser.add_argument("--min-gap-sec", type=float, default=2.0)
    args = parser.parse_args()

    samples = collect_samples(args.num_videos, min_gap_sec=args.min_gap_sec)
    total_pts = sum(len(s["pts"]) for s in samples)
    print(f"Collected {len(samples)} videos, {total_pts} anchor points total")

    model, processor = load_model_florence2()

    for s in samples:
        out_dir = os.path.join(OUT_DIR, s["video_id"])
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n[{s['video_id']}] text_prompt: {s['text_prompt'][:70]}")
        manifest = []
        for i, pt in enumerate(s["pts"]):
            ts = pt["start_sec"]
            frame = extract_frame(s["video_path"], ts)
            caption, gen_time = caption_frame_florence2(model, processor, frame)
            img_path = os.path.join(out_dir, f"{i:02d}_t{ts:.2f}s.png")
            Image.fromarray(frame).save(img_path)
            manifest.append({
                "index": i, "start_sec": ts, "has_agent": pt["has_agent"],
                "caption": caption, "image": img_path,
            })
            print(f"  [{i:02d}] t={ts:.2f}s [has_agent={pt['has_agent']}] ({gen_time:.1f}s): {caption}")
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
