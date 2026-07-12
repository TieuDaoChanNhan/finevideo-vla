#!/usr/bin/env python3
"""
Prototype for the FineVideo keyframe-captioning pipeline (Huu, Jul 11 2026 request).

Validates the two building blocks needed before designing the real pipeline:
  1. Extract a frame from a source video at an arbitrary timestamp (mirrors
     how the real pipeline will use each activity's `chunk_timing` start_sec).
  2. Caption that frame with Qwen2.5-VL.

Runs on CPU against local sample videos in `videos/` — no JUPITER dependency.

Usage:
    python tools/analysis/caption_prototype.py
    python tools/analysis/caption_prototype.py --video videos/sample1.mp4 --timestamps 1.0 5.0 10.0
"""

import argparse
import time

import cv2

DEFAULT_VIDEO = "videos/sample1.mp4"
DEFAULT_TIMESTAMPS = [1.0, 5.0, 10.0]
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
FLORENCE_MODEL_ID = "microsoft/Florence-2-base"
SMOLVLM2_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"


def select_anchor_points(chunk_timing, min_gap_sec=2.0):
    """Caption anchor points, final design (agreed with Van Khue):
    the first chunk of the activity (opening context) plus every chunk where
    has_agent flips relative to the previous chunk (a person genuinely
    appears/disappears). has_seed2/has_cosmos/has_avc_lm are NOT used here:
    cosmos/avc_lm never vary within an activity, and has_seed2 (even after
    the phase6 chunk_timing fix) flips ~54x/activity purely because seed2
    fires at a fixed 1fps rate -- a technical cadence, not a content change.

    min_gap_sec debounces has_agent itself: in busy/multi-person scenes
    (sports, crowds) YOLO detection flickers frame-to-frame even though a
    person is continuously present, producing many spurious flips close
    together. Candidate points within min_gap_sec of the last KEPT point are
    dropped (person's real presence/absence hasn't had time to change
    meaningfully). This only affects which points this script chooses to
    caption -- it does not alter has_agent or any stored chunk_timing data.
    """
    if not chunk_timing:
        return []
    pts = [chunk_timing[0]]
    prev_agent = chunk_timing[0]["has_agent"]
    last_kept_sec = chunk_timing[0]["start_sec"]
    for c in chunk_timing[1:]:
        if c["has_agent"] != prev_agent:
            prev_agent = c["has_agent"]
            if c["start_sec"] - last_kept_sec < min_gap_sec:
                continue
            pts.append(c)
            last_kept_sec = c["start_sec"]
    return pts


def extract_frame(video_path: str, timestamp_sec: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = int(round(timestamp_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame at {timestamp_sec}s (idx {frame_idx}) in {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def load_model():
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print(f"Loading {MODEL_ID} (CPU, this will take a while)...")
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, device_map="cpu"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return model, processor


def caption_frame(model, processor, frame_rgb):
    from PIL import Image

    image = Image.fromarray(frame_rgb)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe what the person is doing in one short sentence."},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")

    t0 = time.time()
    generated = model.generate(**inputs, max_new_tokens=48)
    gen_time = time.time() - t0

    output_ids = generated[:, inputs["input_ids"].shape[1]:]
    caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return caption, gen_time


def load_model_florence2():
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    print(f"Loading {FLORENCE_MODEL_ID} (CPU)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        FLORENCE_MODEL_ID, torch_dtype=torch.float32, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(FLORENCE_MODEL_ID, trust_remote_code=True)
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return model, processor


def caption_frame_florence2(model, processor, frame_rgb, task="<CAPTION>"):
    from PIL import Image

    image = Image.fromarray(frame_rgb)
    inputs = processor(text=task, images=image, return_tensors="pt")

    t0 = time.time()
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=64,
        num_beams=3,
    )
    gen_time = time.time() - t0

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    parsed = processor.post_process_generation(text, task=task, image_size=image.size)
    caption = parsed.get(task, text).strip()
    return caption, gen_time


def load_model_smolvlm2():
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print(f"Loading {SMOLVLM2_MODEL_ID} (CPU)...")
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        SMOLVLM2_MODEL_ID, torch_dtype=torch.float32
    )
    processor = AutoProcessor.from_pretrained(SMOLVLM2_MODEL_ID)
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return model, processor


def caption_frame_smolvlm2(model, processor, frame_rgb):
    from PIL import Image

    image = Image.fromarray(frame_rgb)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe what the person is doing in one short sentence."},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")

    t0 = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens=64)
    gen_time = time.time() - t0

    output_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
    caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return caption, gen_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--timestamps", type=float, nargs="+", default=DEFAULT_TIMESTAMPS)
    args = parser.parse_args()

    print(f"=== Step 1: frame extraction from {args.video} ===")
    frames = []
    for ts in args.timestamps:
        frame = extract_frame(args.video, ts)
        frames.append((ts, frame))
        print(f"  t={ts}s -> frame shape {frame.shape}")

    print("\n=== Step 2: captioning with Qwen2.5-VL ===")
    model, processor = load_model()
    for ts, frame in frames:
        caption, gen_time = caption_frame(model, processor, frame)
        print(f"  t={ts}s ({gen_time:.1f}s gen): {caption}")


if __name__ == "__main__":
    main()
