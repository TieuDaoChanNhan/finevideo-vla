#!/usr/bin/env python3
"""
SNAC audio encoder -- turns a real audio/video file into `<snac_N>` tokens
(listen format, 3 tokens/base-frame), the reverse of tools/decode/decode_snac.py.
Reuses the exact encode_listen() logic already in pipeline_pose/snac_finevideo.py
(unchanged since inception) rather than re-deriving it.

This model (vla-1.7b-qwen3-v2) only ever saw listen-format audio wrapped in
the generic <snac> tag -- NOT the newer (2026-07-23) <listen>/<speak>
convention or the speak-format L2 tokens. Output here matches that: always
listen-format, always <snac> wrapper.

Usage:
    python tools/encode/encode_snac.py --input clip.wav --output tokens.txt
    python tools/encode/encode_snac.py --input video.mp4 --output tokens.txt
    # any format ffmpeg can read (audio extracted automatically, works on
    # video files too -- just uses the audio track)
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pipeline_pose"))

SAMPLE_RATE = 24000
SNAC_MODEL = "hubertsiuzdak/snac_24khz"


def _get_ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio(input_path: str):
    import numpy as np

    cmd = [_get_ffmpeg(), "-y", "-i", input_path, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(f"ffmpeg failed to extract audio from {input_path}: {result.stderr.decode(errors='replace')[:500]}")
    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if len(audio) == 0:
        raise RuntimeError(f"No audio extracted from {input_path} -- does it have an audio track?")
    return audio


def encode_file(input_path: str) -> list:
    import torch
    from snac import SNAC
    from snac_finevideo import encode_listen  # reused verbatim, not re-derived

    audio = extract_audio(input_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SNAC.from_pretrained(SNAC_MODEL).eval().to(device)
    tokens = encode_listen(audio, model, device)  # already "<snac_N>" strings, listen-format
    return tokens


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Audio or video file (any ffmpeg-readable format)")
    ap.add_argument("--output", help="Optional: write the <snac> block to this file")
    args = ap.parse_args()

    tokens = encode_file(args.input)
    duration_s = len(tokens) / 3 / 12.5
    print(f"{len(tokens)} snac tokens ({len(tokens) // 3} base frames, ~{duration_s:.2f}s @ 12.5Hz base rate)")
    block = "<snac> " + " ".join(tokens) + " </snac>"
    print(block)
    if args.output:
        with open(args.output, "w") as f:
            f.write(block)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
