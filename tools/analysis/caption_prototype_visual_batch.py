#!/usr/bin/env python3
"""
Multi-video version of caption_prototype_visual.py: loads Qwen2.5-VL once,
then for each given video_id saves every modality-transition frame as PNG
+ caption + manifest.json, for eyeball QA across a larger sample (e.g. to
gauge how common burned-in subtitles are across FineVideo videos).

Usage:
    python tools/analysis/caption_prototype_visual_batch.py --video-ids id1 id2 id3 ...
"""

import argparse
import glob
import json
import os

from PIL import Image

from caption_prototype import extract_frame, load_model, caption_frame

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


def find_activities_for_videos(video_ids):
    wanted = set(video_ids)
    found = {}
    for fp in sorted(glob.glob(os.path.join(MERGED_DIR, "*.jsonl"))):
        if not wanted:
            break
        with open(fp) as f:
            for line in f:
                d = json.loads(line)
                vid = d["video_id"]
                if vid not in wanted:
                    continue
                for scene in d["scenes"]:
                    for act in scene["activities"]:
                        ct = act.get("chunk_timing") or []
                        if ct:
                            found[vid] = act
                            wanted.discard(vid)
                            break
                    if vid not in wanted:
                        break
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-ids", nargs="+", required=True)
    parser.add_argument("--out-dir", default="/p/data1/mmlaion/nguyen38/3d-human-pose/logs/caption_frames")
    args = parser.parse_args()

    activities = find_activities_for_videos(args.video_ids)
    print(f"Resolved {len(activities)}/{len(args.video_ids)} video_ids to activities with chunk_timing")

    model, processor = load_model()

    for vid, act in activities.items():
        video_path = os.path.join(VIDEOS_DIR, f"{vid}.mp4")
        if not os.path.exists(video_path):
            print(f"[{vid}] SKIP: no video file")
            continue
        pts = transition_points(act["chunk_timing"])
        out_dir = os.path.join(args.out_dir, vid)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n[{vid}] text_prompt: {act.get('text_prompt', '')[:80]}")
        print(f"  {len(act['chunk_timing'])} chunks -> {len(pts)} transition points")

        manifest = []
        for i, pt in enumerate(pts):
            ts = pt["start_sec"]
            frame = extract_frame(video_path, ts)
            active = [f.replace("has_", "") for f in FLAGS if pt[f]]
            caption, gen_time = caption_frame(model, processor, frame)

            img_path = os.path.join(out_dir, f"{i:02d}_t{ts:.2f}s.png")
            Image.fromarray(frame).save(img_path)
            manifest.append({
                "index": i, "chunk_idx": pt["chunk_idx"], "start_sec": ts,
                "active_modalities": active, "caption": caption, "image": img_path,
            })
            print(f"  [{i:02d}] t={ts:.2f}s [{'+'.join(active)}] -> {caption}")

        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
