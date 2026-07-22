"""SNAC tokenize + interleave + flatten for laion/emotional-roleplay-finetuning-dataset.

Reads the 6 parquet shards (67,491 rows, downloaded to
$VLA/laion_emotional_roleplay/data/*.parquet), per Huu's instruction:
"concatenate the text and interleave with snac and/or moss tokens."

Format decided after review with Van Khue (session 2026-07-20), validated against
the dataset's own README recommendation ("For training a voice-direction TTS, use
voice_description (+ text) -- it describes the audio... req_* capture intent but
the model does not always comply"):

    USER: <text> [Voice: <voice_description>] ASSISTANT:
    <snac> <snac_N> <snac_N> ... </snac>

`instruction`/`req_*` are deliberately dropped -- they encode generation *intent*,
which the README's own Limitations section says the model does not reliably
follow (male/calm-biased default; req_* overstate female/loud versus what was
realized). `voice_description` is the judge-verified description of the audio
that actually exists, so it is the reliable field to condition on.

Does not import pipeline_pose/snac_finevideo.py -- that module has `X | None`
type hints (PEP 604) evaluated eagerly at import time, which breaks under
Python 3.9 (env_motion_final). Its listen-format encoding math (SNAC offsets,
3-tokens-per-base-frame) is dependency-free and reproduced here directly
instead of fixing/importing across the Python-version boundary.

No chunk alignment needed (unlike FineVideo's snac_finevideo.py): each row is
one independent audio clip, not a window into a shared 8-frame video grid, so
the whole clip's SNAC tokens go into one flat block.

Output: {OUTPUT_DIR}/roleplay_snac_flat_{shard:05d}.jsonl (one shard per 5,000
rows), one line per row: {"id": ..., "text": <flattened training record>}.
Resumable: skips a shard file that already exists.

Usage:
    python data_prep/laion_emotional_roleplay/tokenize_snac.py [--limit N] [--rows-per-shard N]
"""
import argparse
import json
import os
import subprocess

import numpy as np
import pandas as pd
import torch
import imageio_ffmpeg
from snac import SNAC

DATA_ROOT = "/p/data1/mmlaion/shared/vla/laion_emotional_roleplay"
INPUT_DIR = os.path.join(DATA_ROOT, "data")
OUTPUT_DIR = os.path.join(DATA_ROOT, "flattened")

SAMPLE_RATE = 24000
SNAC_MODEL = "hubertsiuzdak/snac_24khz"

# Same Orpheus/MixtureVitae-Omni-compatible offsets as pipeline_pose/snac_finevideo.py
OFFSET_L0 = 128266
OFFSET_L1A = 128266 + 4096
OFFSET_L1B = 128266 + 4 * 4096

# L2 ("speak" format, 2026-07-22): the finest 50Hz level, 4 sub-positions per
# base frame, never allocated in the tokenizer vocab before -- same offset
# scheme originally computed for the now-deleted add_snac_l2_tokens.py
# (REPORT.md #32 point 1), placed right after L1B's band ends (148745) so it
# doesn't collide with anything already in use. Real ids per band: L2_0
# 148746-152841, L2_1 152842-156937, L2_2 156938-161033, L2_3 161034-165129.
# These are NOT yet in any tokenizer's vocab -- must be added (mirroring
# add_snac_l2_tokens.py's approach) before speak-format data is atomic in
# training. Decided 2026-07-22 after a real A/B (tools/snac_l2_experiment.py)
# showed audibly better reconstruction, at a real +133% token cost.
OFFSET_L2 = [148746 + k * 4096 for k in range(4)]

VALID_ADHERENCE = {1, 2, 3, 4, 5}  # drops ~32/67,491 rows with out-of-range values (8/9/10/80/0)

_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def decode_mp3_bytes(mp3_bytes: bytes) -> np.ndarray:
    """MP3 bytes -> float32 mono 24kHz PCM, via ffmpeg piped through stdin (no temp file)."""
    cmd = [_FFMPEG, "-y", "-i", "pipe:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"]
    result = subprocess.run(cmd, input=mp3_bytes, capture_output=True, timeout=60)
    if result.returncode != 0 or not result.stdout:
        return None
    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
    return audio if len(audio) > 0 else None


def encode_listen(audio: np.ndarray, model, device: str) -> list[str]:
    """SNAC listen-format encode: 3 tokens per base frame (12.5Hz base -> 37.5 tok/s).
    Identical math to pipeline_pose/snac_finevideo.py's encode_listen(), reproduced
    standalone here (see module docstring for why)."""
    tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        codes = model.encode(tensor)
    c0, c1 = codes[0], codes[1]
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


def encode_speak(audio: np.ndarray, model, device: str) -> list[str]:
    """SNAC speak-format encode: full 3-level codebook, 7 tokens per base frame
    (L0, L1a, L1b, L2_0..L2_3 -- 12.5Hz base -> 87.5 tok/s). +133% tokens vs
    encode_listen() (7/3 ratio), confirmed both in theory and on real audio
    via tools/snac_l2_experiment.py. Real L2 tokens are NOT yet in any
    tokenizer vocab -- see OFFSET_L2's docstring above before using this in
    a real training run."""
    tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        codes = model.encode(tensor)
    c0, c1, c2 = codes[0], codes[1], codes[2]
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
            tokens.append(f"<snac_{c2[0, idx].item() + OFFSET_L2[k]}>")
    return tokens


def flatten_record(text: str, voice_description: str, tokens: list[str]) -> str:
    return (
        f"USER: {text.strip()} [Voice: {voice_description.strip()}] ASSISTANT:\n"
        f"<snac> " + " ".join(tokens) + " </snac>"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Only process first N rows total (0 = all).")
    ap.add_argument("--rows-per-shard", type=int, default=5000)
    ap.add_argument("--input-dir", default=INPUT_DIR)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--format", choices=["listen", "speak"], default="listen",
                     help="listen = L0+L1 only (current production, 3 tok/frame); "
                          "speak = full L0+L1+L2 (2026-07-22, 7 tok/frame, +133%% tokens). "
                          "speak requires L2 tokens to be added to the tokenizer vocab first.")
    args = ap.parse_args()

    encode_fn = encode_speak if args.format == "speak" else encode_listen
    # Separate filename prefix so speak-format shards never collide with
    # already-produced listen-format ones in the same output dir.
    shard_prefix = "roleplay_snac_speak_flat_" if args.format == "speak" else "roleplay_snac_flat_"

    os.makedirs(args.output_dir, exist_ok=True)

    parquet_files = sorted(
        os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith(".parquet")
    )
    print(f"Loading {len(parquet_files)} parquet shards...")
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    print(f"Loaded {len(df)} rows total")

    before = len(df)
    df = df[df["adherence_score"].isin(VALID_ADHERENCE)].reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows with out-of-range adherence_score (kept {len(df)})")

    if args.limit > 0:
        df = df.iloc[: args.limit]
        print(f"--limit applied: processing first {len(df)} rows")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading SNAC model ({SNAC_MODEL}) on {device}...")
    model = SNAC.from_pretrained(SNAC_MODEL).eval().to(device)
    print("SNAC loaded.")

    n_shards = (len(df) + args.rows_per_shard - 1) // args.rows_per_shard
    n_done = n_skip_shard = n_decode_fail = n_snac_fail = 0
    total_tokens = 0

    for shard_idx in range(n_shards):
        out_path = os.path.join(args.output_dir, f"{shard_prefix}{shard_idx:05d}.jsonl")
        if os.path.exists(out_path):
            print(f"[shard {shard_idx}/{n_shards}] SKIP (already exists): {out_path}")
            n_skip_shard += 1
            continue

        start = shard_idx * args.rows_per_shard
        end = min(start + args.rows_per_shard, len(df))
        shard_rows = []

        for i in range(start, end):
            row = df.iloc[i]
            audio = decode_mp3_bytes(row["audio"]["bytes"])
            if audio is None:
                n_decode_fail += 1
                continue
            try:
                tokens = encode_fn(audio, model, device)
            except Exception as e:
                print(f"  SNAC failed for {row['id']}: {e}")
                n_snac_fail += 1
                continue
            if not tokens:
                n_snac_fail += 1
                continue

            flat = flatten_record(row["text"], row["voice_description"], tokens)
            shard_rows.append({"id": row["id"], "text": flat})
            n_done += 1
            total_tokens += len(tokens)

            if (i + 1) % 500 == 0:
                print(f"  [shard {shard_idx}] {i + 1 - start}/{end - start} rows in shard, "
                      f"{n_done} total ok, {total_tokens:,} snac tokens so far")

        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for r in shard_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.rename(tmp_path, out_path)
        print(f"[shard {shard_idx}/{n_shards}] wrote {len(shard_rows)} rows -> {out_path}")

    print(f"\nDONE. ok={n_done} skipped_shards={n_skip_shard} "
          f"decode_fail={n_decode_fail} snac_fail={n_snac_fail} total_snac_tokens={total_tokens:,}")


if __name__ == "__main__":
    main()
