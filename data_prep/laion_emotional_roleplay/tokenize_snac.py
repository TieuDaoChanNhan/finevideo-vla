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

# L2 ("speak" format): the finest 50Hz level, 4 sub-positions per base frame.
# 2026-07-23: corrected to match the REAL scheme in Huu/Chien's production
# snac_gpu.py on Leonardo (pipeline_video/snac_gpu.py, byte-identical copy
# confirmed 2026-07-23) -- the Orpheus-standard SNAC packing layout. The
# first two sub-codes (136458, 140554) sit in what was previously assumed to
# be an unused gap between L1A and L1B -- it is not a gap, Leo's L2 lives
# there. The 2026-07-22 first attempt at this (148746/152842/156938/161034,
# all placed after L1B) was wrong: it happened to coincide with Leo's last
# two sub-code offsets (148746, 152842) by arithmetic accident, but put them
# in the wrong position in the per-frame sequence and invented two extra
# bands (156938, 161034) that don't exist in Leo's scheme at all. Never used
# in any completed training run, so free to correct. Real ids per band:
# L2_0 136458-140553, L2_1 140554-144649, L2_2 148746-152841, L2_3 152842-156937.
OFFSET_L2 = [136458, 140554, 148746, 152842]

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
    -- order L0, L1a, L2_0, L2_1, L1b, L2_2, L2_3 (12.5Hz base -> 87.5 tok/s).
    2026-07-23: token order corrected to interleave L2 between L1a/L1b,
    matching Huu/Chien's production snac_gpu.py on Leonardo exactly (both
    offsets and position within the group -- see OFFSET_L2's docstring
    above). +133% tokens vs encode_listen() (7/3 ratio), confirmed both in
    theory and on real audio via tools/snac_l2_experiment.py."""
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
        tokens.append(f"<snac_{c2[0, i2[0]].item() + OFFSET_L2[0]}>")
        tokens.append(f"<snac_{c2[0, i2[1]].item() + OFFSET_L2[1]}>")
        tokens.append(f"<snac_{c1[0, i1b].item() + OFFSET_L1B}>")
        tokens.append(f"<snac_{c2[0, i2[2]].item() + OFFSET_L2[2]}>")
        tokens.append(f"<snac_{c2[0, i2[3]].item() + OFFSET_L2[3]}>")
    return tokens


def flatten_record(text: str, voice_description: str, tokens: list[str]) -> str:
    """2026-07-23: always wrapped <speak>...</speak>, never <snac>. This
    dataset is entirely ASSISTANT replies (a character voice-acting a line),
    never ambient/scene audio -- role decides the tag, not whether the
    source voice has been cloned to a single identity yet (that's a later,
    separate upgrade to the audio Chien's voice-clone pipeline feeds in;
    the tag/format stays <speak> either way). Contrast with FineVideo's own
    audio, always <listen> -- see pipeline_pose/phase6_merge_adaptive.py's
    build_snac_insertion()."""
    return (
        f"USER: {text.strip()} [Voice: {voice_description.strip()}] ASSISTANT:\n"
        f"<speak> " + " ".join(tokens) + " </speak>"
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
