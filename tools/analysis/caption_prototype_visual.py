#!/usr/bin/env python3
"""
Visual QA for the captioning prototype: for ONE video, save every selected
frame (modality-transition points from chunk_timing) as a PNG plus its
Qwen2.5-VL caption, so results can be checked by eye instead of just reading
printed text.

Usage:
    python tools/analysis/caption_prototype_visual.py --video-id sqCFQy0Cdmo
    python tools/analysis/caption_prototype_visual.py --video-id sqCFQy0Cdmo --out-dir logs/caption_frames
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


def find_activity(video_id):
    for fp in sorted(glob.glob(os.path.join(MERGED_DIR, "*.jsonl"))):
        with open(fp) as f:
            for line in f:
                d = json.loads(line)
                if d["video_id"] != video_id:
                    continue
                for scene in d["scenes"]:
                    for act in scene["activities"]:
                        ct = act.get("chunk_timing") or []
                        if ct:
                            return act
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--out-dir", default="logs/caption_frames")
    args = parser.parse_args()

    video_path = os.path.join(VIDEOS_DIR, f"{args.video_id}.mp4")
    if not os.path.exists(video_path):
        raise SystemExit(f"No video at {video_path}")

    act = find_activity(args.video_id)
    if act is None:
        raise SystemExit(f"No activity with chunk_timing found for {args.video_id}")

    pts = transition_points(act["chunk_timing"])
    print(f"video_id: {args.video_id}")
    print(f"text_prompt: {act.get('text_prompt', '')}")
    print(f"activity time_range_sec: {act.get('time_range_sec')}")
    print(f"{len(act['chunk_timing'])} chunks total -> {len(pts)} transition points selected\n")

    out_dir = os.path.join(args.out_dir, args.video_id)
    os.makedirs(out_dir, exist_ok=True)

    model, processor = load_model()

    manifest = []
    for i, pt in enumerate(pts):
        ts = pt["start_sec"]
        frame = extract_frame(video_path, ts)
        active = [f.replace("has_", "") for f in FLAGS if pt[f]]
        caption, gen_time = caption_frame(model, processor, frame)

        img_path = os.path.join(out_dir, f"{i:02d}_t{ts:.2f}s.png")
        Image.fromarray(frame).save(img_path)

        entry = {
            "index": i,
            "chunk_idx": pt["chunk_idx"],
            "start_sec": ts,
            "active_modalities": active,
            "caption": caption,
            "image": img_path,
        }
        manifest.append(entry)
        print(f"[{i:02d}] t={ts:.2f}s [{'+'.join(active)}] -> {img_path}")
        print(f"      caption: {caption}")

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved {len(pts)} frames + captions to {out_dir}/ (manifest.json)")


if __name__ == "__main__":
    main()
