#!/usr/bin/env python3
"""
One-off experiment (not part of the production pipeline): pick a few real
laion/emotional-roleplay audio clips, encode with SNAC's full 3-level
codebook (L0+L1+L2, "speak" format) instead of the current production
"listen" format (L0+L1 only, data_prep/laion_emotional_roleplay/tokenize_snac.py),
measure the real token-count increase, and decode both versions plus the
original real audio so a human can A/B them directly.

Per Huu's 2026-07-22 Discord idea (REPORT.md #33 item 3): L0+L1 = <listen>,
full L0+L1+L2 = <speak> (higher quality). This script measures whether that's
worth the token cost and whether it's audibly better, before committing to
adding L2 tokens to the vocab/pipeline (a prior attempt at this,
add_snac_l2_tokens.py, was written then deleted 2026-07-22 specifically
because there was no L2-encoded data yet to justify it -- this script
produces that data on a small scale to inform the decision).

Usage:
    python tools/snac_l2_experiment.py --n 3 --output-dir samples/snac_l2_experiment
"""
import argparse
import os
import subprocess

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import imageio_ffmpeg
from snac import SNAC

DATA_ROOT = "/p/data1/mmlaion/shared/vla/laion_emotional_roleplay"
INPUT_DIR = os.path.join(DATA_ROOT, "data")
SAMPLE_RATE = 24000
SNAC_MODEL = "hubertsiuzdak/snac_24khz"

OFFSET_L0 = 128266
OFFSET_L1A = 128266 + 4096
OFFSET_L1B = 128266 + 4 * 4096
# L2 offsets: not part of the production vocab (never added -- see REPORT.md
# #32 point 1). Used here only to label tokens for the printed count/preview;
# decode goes straight from raw codebook indices, bypassing any vocab.
OFFSET_L2_BASE = 128266 + 8 * 4096

_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def decode_mp3_bytes(mp3_bytes: bytes) -> np.ndarray:
    cmd = [_FFMPEG, "-y", "-i", "pipe:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"]
    result = subprocess.run(cmd, input=mp3_bytes, capture_output=True, timeout=60)
    if result.returncode != 0 or not result.stdout:
        return None
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def encode_full(audio: np.ndarray, model, device: str):
    """Returns (c0, c1, c2) real SNAC codes -- all 3 hierarchical levels, no dropping."""
    tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        codes = model.encode(tensor)
    return codes[0], codes[1], codes[2]  # shapes: (1,n0) (1,2*n0) (1,4*n0)


def listen_tokens(c0, c1) -> list:
    """Reproduces tokenize_snac.py's encode_listen(): 3 tokens/base-frame (L0,L1a,L1b)."""
    n0 = c0.shape[1]
    tokens = []
    for i in range(n0):
        i1a, i1b = 2 * i, 2 * i + 1
        if i1b >= c1.shape[1]:
            break
        tokens.append(f"<snac_{c0[0, i].item() + OFFSET_L0}>")
        tokens.append(f"<snac_{c1[0, i1a].item() + OFFSET_L1A}>")
        tokens.append(f"<snac_{c1[0, i1b].item() + OFFSET_L1B}>")
    return tokens


def speak_tokens(c0, c1, c2) -> list:
    """Full 3-level: 7 tokens/base-frame (L0, L1a, L1b, L2[0..3])."""
    n0 = c0.shape[1]
    tokens = []
    for i in range(n0):
        i1a, i1b = 2 * i, 2 * i + 1
        i2 = [4 * i + k for k in range(4)]
        if i1b >= c1.shape[1] or i2[-1] >= c2.shape[1]:
            break
        tokens.append(f"<snac_{c0[0, i].item() + OFFSET_L0}>")
        tokens.append(f"<snac_{c1[0, i1a].item() + OFFSET_L1A}>")
        tokens.append(f"<snac_{c1[0, i1b].item() + OFFSET_L1B}>")
        for k, idx in enumerate(i2):
            tokens.append(f"<snac_{c2[0, idx].item() + OFFSET_L2_BASE + k * 4096}>")
    return tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="Number of real records to test")
    ap.add_argument("--output-dir", default="samples/snac_l2_experiment")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    parquet_files = sorted(
        os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith(".parquet")
    )
    print(f"Loading first parquet shard: {parquet_files[0]}")
    df = pd.read_parquet(parquet_files[0])
    df = df[df["adherence_score"].isin({1, 2, 3, 4, 5})].reset_index(drop=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading SNAC model on {device}...")
    model = SNAC.from_pretrained(SNAC_MODEL).eval().to(device)

    rows = df.iloc[: args.n]
    summary = []
    for i, row in rows.iterrows():
        rid = row["id"]
        audio = decode_mp3_bytes(row["audio"]["bytes"])
        if audio is None:
            print(f"  {rid}: mp3 decode FAILED, skipping")
            continue

        c0, c1, c2 = encode_full(audio, model, device)
        lt = listen_tokens(c0, c1)
        st = speak_tokens(c0, c1, c2)

        rec_dir = os.path.join(args.output_dir, rid)
        os.makedirs(rec_dir, exist_ok=True)

        sf.write(os.path.join(rec_dir, "original_real.wav"), audio, SAMPLE_RATE)

        n0 = c0.shape[1]
        c2_zero = torch.zeros(1, 4 * n0, dtype=torch.long, device=device)
        with torch.inference_mode():
            audio_listen = model.decode([c0.to(device), c1.to(device), c2_zero])
        sf.write(os.path.join(rec_dir, "decoded_listen_L0L1.wav"),
                 audio_listen.squeeze().float().cpu().numpy(), SAMPLE_RATE)

        with torch.inference_mode():
            audio_speak = model.decode([c0.to(device), c1.to(device), c2.to(device)])
        sf.write(os.path.join(rec_dir, "decoded_speak_L0L1L2.wav"),
                 audio_speak.squeeze().float().cpu().numpy(), SAMPLE_RATE)

        ratio = len(st) / len(lt) if lt else float("nan")
        print(f"  {rid}: listen={len(lt)} tok, speak={len(st)} tok (+{ratio - 1:.1%}) "
              f"-> {rec_dir}/")
        summary.append((rid, len(lt), len(st), ratio))

    print("\n=== SUMMARY ===")
    print(f"{'id':40s} {'listen tok':>10s} {'speak tok':>10s} {'increase':>10s}")
    for rid, lt, st, ratio in summary:
        print(f"{rid:40s} {lt:10d} {st:10d} {ratio - 1:9.1%}")
    if summary:
        avg_ratio = sum(r for _, _, _, r in summary) / len(summary)
        print(f"\nAverage token increase (L0+L1 -> L0+L1+L2): {avg_ratio - 1:.1%}")
        print("(theoretical: 7 tokens/frame vs 3 tokens/frame = +133.3%, since L2 adds 4 tokens/base-frame)")


if __name__ == "__main__":
    main()
