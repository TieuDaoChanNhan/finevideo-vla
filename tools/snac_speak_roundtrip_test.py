#!/usr/bin/env python3
"""
One-off verification: encode 1 real English audio clip with the corrected
speak-format SNAC scheme (2026-07-23, matches Leo/Orpheus layout -- see
data_prep/laion_emotional_roleplay/tokenize_snac.py's OFFSET_L2 docstring),
then decode it back purely from the token STRINGS (not from raw codes) using
tools/decode/decode_snac.py's decode_speak_tokens(). This is a true
token-layer round-trip check, not just a raw-SNAC-codec check.

Usage:
    python tools/snac_speak_roundtrip_test.py
"""
import os
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
import pandas as pd
import torch
from snac import SNAC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_prep", "laion_emotional_roleplay"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "decode"))
import tokenize_snac  # noqa: E402
import decode_snac  # noqa: E402

DATA_DIR = "/p/data1/mmlaion/shared/vla/laion_emotional_roleplay/data"
OUTPUT_DIR = "samples/snac_l2_leo_scheme_fix"
SAMPLE_RATE = 24000
_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def decode_mp3_bytes(mp3_bytes: bytes) -> np.ndarray:
    cmd = [_FFMPEG, "-y", "-i", "pipe:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"]
    result = subprocess.run(cmd, input=mp3_bytes, capture_output=True, timeout=60)
    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
    return audio


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    parquet_files = sorted(os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".parquet"))
    df = pd.read_parquet(parquet_files[0])
    row = df[(df["language"] == "English") & (df["adherence_score"] == 5)].iloc[0]
    rid = row["id"]
    print(f"Record: {rid}")
    print(f"Language: {row['language']}  |  duration: {row['duration']:.2f}s")
    print(f"Text: {row['text']}")

    audio = decode_mp3_bytes(row["audio"]["bytes"])
    sf_path_orig = os.path.join(OUTPUT_DIR, f"{rid}__original.wav")
    import soundfile as sf
    sf.write(sf_path_orig, audio, SAMPLE_RATE)
    print(f"Saved original -> {sf_path_orig}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SNAC on {device}...")
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)

    print("Encoding with corrected encode_speak() (Leo-matched offsets + order)...")
    tokens = tokenize_snac.encode_speak(audio, model, device)
    print(f"  {len(tokens)} tokens ({len(tokens)//7} base frames)")

    tokens_path = os.path.join(OUTPUT_DIR, f"{rid}__speak_tokens.txt")
    with open(tokens_path, "w") as f:
        f.write(" ".join(tokens))
    print(f"Saved raw token stream -> {tokens_path}")

    print("Decoding back from token STRINGS only (not raw codes) via decode_speak_tokens()...")
    token_ids = [int(t[len("<snac_"):-1]) for t in tokens]
    decoded_path = os.path.join(OUTPUT_DIR, f"{rid}__decoded_speak_roundtrip.wav")
    decode_snac.decode_speak_tokens(token_ids, decoded_path)
    print(f"Saved round-trip decode -> {decoded_path}")

    print("\nDone. 3 files in", OUTPUT_DIR)
    print(f"  {rid}__original.wav              (real source audio)")
    print(f"  {rid}__decoded_speak_roundtrip.wav (encode->tokens->decode, corrected Leo scheme)")
    print(f"  {rid}__speak_tokens.txt            (raw token stream)")


if __name__ == "__main__":
    main()
