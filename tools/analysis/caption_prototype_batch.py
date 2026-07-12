#!/usr/bin/env python3
"""
Batch prototype for the FineVideo keyframe-captioning pipeline.

Extends caption_prototype.py from single-frame validation to a small real
batch: for N real activities (from final_dataset_adaptive_v2), find their
modality-transition points in chunk_timing, extract the frame at each from
the matching video in videos_staging/, and caption it with Qwen2.5-VL.

"Modality transition point" = a chunk whose (has_seed2, has_cosmos,
has_avc_lm, has_agent, has_snac) tuple differs from the previous chunk's.
This targets language anchors exactly where the model needs to learn to
switch modality, instead of captioning every 8-frame chunk (~98 chunks/
activity on average vs. ~2.8 transition points/activity).

Usage:
    python tools/analysis/caption_prototype_batch.py --num-videos 15
"""

import argparse
import glob
import json
import os
import time

MERGED_DIR = "/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/final_dataset_adaptive_v2"
VIDEOS_DIR = "/p/data1/mmlaion/shared/nguyen38/data/videos_staging"
FLAGS = ["has_seed2", "has_cosmos", "has_avc_lm", "has_agent", "has_snac"]


def transition_points(chunk_timing):
    pts = []
    prev = None
    for c in chunk_timing:
        sig = tuple(c[f] for f in FLAGS)
        if sig != prev:
            pts.append(c)
            prev = sig
    return pts


def collect_samples(num_videos, activities_per_video=1):
    """Walk merged files, yield (video_id, activity, transition_points) for
    activities whose video actually exists in videos_staging/."""
    samples = []
    seen_videos = set()
    files = sorted(glob.glob(os.path.join(MERGED_DIR, "*.jsonl")))
    for fp in files:
        if len(samples) >= num_videos:
            break
        with open(fp) as f:
            for line in f:
                if len(samples) >= num_videos:
                    break
                d = json.loads(line)
                vid = d["video_id"]
                if vid in seen_videos:
                    continue
                video_path = os.path.join(VIDEOS_DIR, f"{vid}.mp4")
                if not os.path.exists(video_path):
                    continue
                seen_videos.add(vid)
                acts_taken = 0
                for scene in d["scenes"]:
                    for act in scene["activities"]:
                        ct = act.get("chunk_timing") or []
                        if not ct:
                            continue
                        pts = transition_points(ct)
                        samples.append({
                            "video_id": vid,
                            "video_path": video_path,
                            "text_prompt": act.get("text_prompt", ""),
                            "transition_points": pts,
                        })
                        acts_taken += 1
                        if acts_taken >= activities_per_video:
                            break
                    if acts_taken >= activities_per_video:
                        break
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-videos", type=int, default=15)
    args = parser.parse_args()

    from caption_prototype import extract_frame, load_model, caption_frame

    print(f"=== Collecting up to {args.num_videos} activities with videos on disk ===")
    samples = collect_samples(args.num_videos)
    total_pts = sum(len(s["transition_points"]) for s in samples)
    print(f"Found {len(samples)} activities, {total_pts} total transition points "
          f"({total_pts / len(samples):.1f} avg/activity)")

    print("\n=== Loading Qwen2.5-VL ===")
    model, processor = load_model()

    print("\n=== Captioning ===")
    t_start = time.time()
    n_ok, n_fail = 0, 0
    for s in samples:
        print(f"\n[{s['video_id']}] text_prompt: {s['text_prompt'][:80]}")
        for pt in s["transition_points"]:
            ts = pt["start_sec"]
            try:
                frame = extract_frame(s["video_path"], ts)
            except Exception as e:
                print(f"  t={ts}s FRAME EXTRACT FAILED: {e}")
                n_fail += 1
                continue
            active = [f.replace("has_", "") for f in FLAGS if pt[f]]
            caption, gen_time = caption_frame(model, processor, frame)
            print(f"  t={ts}s [{'+'.join(active)}] ({gen_time:.1f}s): {caption}")
            n_ok += 1

    elapsed = time.time() - t_start
    print(f"\n=== Done: {n_ok} captioned, {n_fail} failed, {elapsed:.0f}s total "
          f"({elapsed / max(n_ok,1):.1f}s/caption) ===")


if __name__ == "__main__":
    main()
