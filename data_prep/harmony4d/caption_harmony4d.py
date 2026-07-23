#!/usr/bin/env python3
"""
Caption 1 representative frame per Harmony4D (category, seq_id) sequence,
same model/method as pipeline_pose/caption_finevideo.py (Qwen2.5-VL-3B-
Instruct) -- 2026-07-23, replacing flatten_harmony4d.py's generic
per-category instruction text with a real per-sequence caption.

Reads data_prep/harmony4d/caption_frame_manifest.json (built by scanning
cached `unzip -l` listings of all 22 Harmony4D zips -- see that build step
in this session's history) -- each entry says which zip + member path holds
1 usable frame (either an mp4 to grab a middle frame from, or a single JPG)
for that sequence's third-person "exo" camera.

Output: outputs/harmony4d_captions.jsonl, one line per sequence:
    {"category", "seq_id", "caption"}

Usage:
    python3 data_prep/harmony4d/caption_harmony4d.py
"""
import json
import os
import sys
import tempfile
import zipfile

# 2026-07-23: must run before numpy/torch import -- caption_finevideo.py sets
# these for the same reason (unconstrained OMP/MKL/OpenBLAS thread pools on a
# shared SLURM allocation cause severe contention/instability). Missing this
# is the likely cause of a CPU segfault ("BLAS: Bad memory unallocation")
# hit before a single caption completed on a 16-cpu allocation.
_NUM_THREADS = os.environ.get("SLURM_CPUS_PER_TASK", "4")
os.environ.setdefault("OMP_NUM_THREADS", _NUM_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _NUM_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NUM_THREADS)

import cv2
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "analysis"))
from caption_prototype import load_model, caption_frame  # noqa: E402

PROMPT = (
    "This photo is from a multi-camera motion-capture studio -- ignore any "
    "tripods, cameras, or rigging visible in the frame. Describe in one "
    "short sentence what the two people are doing together."
)

MANIFEST_PATH = os.path.join(REPO_ROOT, "data_prep", "harmony4d", "caption_frame_manifest.json")
OUTPUT_PATH = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs/harmony4d_captions.jsonl"


def extract_frame_rgb(entry: dict, workdir: str):
    member = entry["member"]
    with zipfile.ZipFile(entry["zip"]) as zf:
        with zf.open(member) as src:
            local_path = os.path.join(workdir, os.path.basename(member))
            with open(local_path, "wb") as dst:
                dst.write(src.read())

    if entry["type"] == "mp4":
        cap = cv2.VideoCapture(local_path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open extracted mp4: {local_path}")
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n_frames // 2))
        ok, frame_bgr = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"cannot read mid-frame from {local_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    else:
        frame_bgr = cv2.imread(local_path)
        if frame_bgr is None:
            raise RuntimeError(f"cannot read jpg: {local_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    os.remove(local_path)
    return frame_rgb


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Only caption first N sequences (smoke test)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default=OUTPUT_PATH)
    args = ap.parse_args()

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    print(f"{len(manifest)} sequences to caption")

    done_keys = set()
    if os.path.exists(args.output):
        for line in open(args.output):
            r = json.loads(line)
            done_keys.add((r["category"], r["seq_id"]))
    manifest = [e for e in manifest if (e["category"], e["seq_id"]) not in done_keys]
    print(f"{len(manifest)} remaining after skip-existing")

    if args.limit > 0:
        manifest = manifest[: args.limit]
        print(f"--limit applied: {len(manifest)} sequences")

    if not manifest:
        return

    model, processor = load_model(device=args.device)

    n_ok = n_fail = 0
    with open(args.output, "a", encoding="utf-8") as out:
        for i, entry in enumerate(manifest, 1):
            tag = f"{entry['category']}/{entry['seq_id']}"
            try:
                with tempfile.TemporaryDirectory() as workdir:
                    frame_rgb = extract_frame_rgb(entry, workdir)
                caption, gen_time = caption_frame(model, processor, frame_rgb, prompt=PROMPT)
                out.write(json.dumps({
                    "category": entry["category"], "seq_id": entry["seq_id"], "caption": caption,
                }, ensure_ascii=False) + "\n")
                out.flush()
                n_ok += 1
                print(f"[{i}/{len(manifest)}] {tag} ({gen_time:.1f}s) -> {caption}")
            except Exception as e:
                n_fail += 1
                print(f"[{i}/{len(manifest)}] {tag} FAILED: {e}")

    print(f"\nDone: {n_ok} ok, {n_fail} failed")


if __name__ == "__main__":
    main()
