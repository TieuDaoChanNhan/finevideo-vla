#!/usr/bin/env python3
"""
General-purpose Cosmos detokenizer -- turns `<cosmos_N>` tokens (from any of
our flattened datasets: FineVideo-VLA, OmniVideo-100K, ...) back into an
actual video.

Verified 20/07/2026 (see PROGRESS_VI.md entry) against a real 200-token
chunk from omnivideo_100k_final: for the DV8x16x16 checkpoint, encode() of
an 8-frame/160x160 chunk produces indices of shape (1, 2, 10, 10) == 200
flat tokens -- this is CHUNK_TOKENS/CHUNK_GRID below, hardcoded for this
checkpoint (would need updating if a different Cosmos-Tokenizer-* variant
is ever used).

This is a lossy, generative-but-deterministic *reconstruction* (the neural
decoder's output, not the original pixels) -- expect visible blur/artifacts,
this is the accepted tradeoff for a much lower token count than storing raw
pixels. Contrast with decode_avclm.py in this same directory, which is
byte-exact (BPE over raw H.264 bytes, not a neural codec).

Token values in `<cosmos_N>` are the RAW encoder codebook indices with no
vocab offset added (verified: flatten scripts across the project just wrap
the same digit string in `<cosmos_{n}>`, never add an offset) -- so N can be
fed straight into decode() after reshaping.

Usage:
    # From a raw list of ints:
    python tools/decode/decode_cosmos.py --tokens 18697,55801,44451,... --output out.mp4

    # From a flattened JSONL record (extracts the Nth 200-token chunk):
    python tools/decode/decode_cosmos.py --input-jsonl path.jsonl --record-id VIDEO_ID \
        --chunk-index 0 --output out.mp4
"""
import argparse
import json
import os
import re
import subprocess
import sys

PROTOTYPE_DIR = "/e/project1/reformo/nguyen38/prototype"
CHECKPOINT_DEC = os.path.join(
    PROTOTYPE_DIR, "pretrained_ckpts/Cosmos-Tokenizer-DV8x16x16/decoder.jit"
)
CHUNK_GRID = (2, 10, 10)  # (T', H', W') per 8-frame/160x160 input chunk, this checkpoint
CHUNK_TOKENS = CHUNK_GRID[0] * CHUNK_GRID[1] * CHUNK_GRID[2]  # 200

_COSMOS_ATOMIC_RE = re.compile(r"<cosmos_(\d+)>")
_COSMOS_RAW_BLOCK_RE = re.compile(r"<cosmos>(.*?)</cosmos>", re.DOTALL)


def extract_chunk_tokens(text: str, chunk_index: int) -> list:
    """Pull out the Nth CHUNK_TOKENS-sized slice of cosmos ids from a record.
    Supports both formats found in this project:
      - flattened/atomic: `<cosmos_N> <cosmos_N> ...` (post-flatten output,
        e.g. omnivideo_100k_final, FineVideo's megatron_dataset_*)
      - raw pre-flatten block: `<cosmos>N N N...</cosmos>` (Step A's own
        output before any flatten script runs, e.g. FineVideo's
        training_ready_rank_*.jsonl activity.video_tokens, or OmniVideo's
        omnivideo_100k_video_flat) -- chunk_index selects which <cosmos>
        block (each block is already exactly one chunk, verified 200 tokens
        both for FineVideo and OmniVideo real data).

    In the flattened/atomic format, chunks with <50% cosmos dropout keep-rate
    have gaps (some chunks entirely missing cosmos), so chunk_index there
    means "Nth cosmos chunk present in the stream", not "Nth temporal chunk
    of the video" -- the two only coincide if dropout happened to keep every
    chunk up to that point.
    """
    raw_blocks = _COSMOS_RAW_BLOCK_RE.findall(text)
    if raw_blocks:
        if chunk_index >= len(raw_blocks):
            raise ValueError(f"Requested chunk {chunk_index} but only {len(raw_blocks)} <cosmos> blocks in this record")
        chunk = [int(x) for x in raw_blocks[chunk_index].split() if x.isdigit()]
        if len(chunk) != CHUNK_TOKENS:
            raise ValueError(f"<cosmos> block {chunk_index} has {len(chunk)} tokens, expected {CHUNK_TOKENS}")
        return chunk

    all_ids = [int(x) for x in _COSMOS_ATOMIC_RE.findall(text)]
    start = chunk_index * CHUNK_TOKENS
    end = start + CHUNK_TOKENS
    chunk = all_ids[start:end]
    if len(chunk) != CHUNK_TOKENS:
        raise ValueError(
            f"Requested chunk {chunk_index} needs tokens [{start}:{end}) but only "
            f"{len(all_ids)} cosmos tokens total in this record."
        )
    return chunk


def _record_text(rec: dict) -> str:
    """Flat records (OmniVideo-100K, Megatron-flattened FineVideo): {"text": ...}.
    Raw FineVideo Step A records (training_ready_rank_*.jsonl) instead nest
    per-activity `video_tokens` under scenes[].activities[] -- concatenate
    all of them for the record so chunk_index can walk the whole video."""
    if "text" in rec:
        return rec["text"]
    if "scenes" in rec:
        return "".join(
            act.get("video_tokens", "")
            for scene in rec["scenes"]
            for act in scene.get("activities", [])
        )
    raise KeyError("Record has neither 'text' nor 'scenes' -- unrecognized schema")


def load_tokens_from_jsonl(path: str, record_id: str, chunk_index: int) -> list:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = rec.get("video_id", rec.get("id"))
            if rid == record_id:
                return extract_chunk_tokens(_record_text(rec), chunk_index)
    raise KeyError(f"record_id={record_id!r} not found in {path}")


def decode_cosmos_chunk(token_ids: list, output_path: str, fps: int = 6) -> None:
    """token_ids: exactly CHUNK_TOKENS (200) raw cosmos codebook indices."""
    if len(token_ids) != CHUNK_TOKENS:
        raise ValueError(f"Expected exactly {CHUNK_TOKENS} tokens, got {len(token_ids)}")
    output_path = os.path.abspath(output_path)  # must resolve before os.chdir() below

    sys.path.insert(0, PROTOTYPE_DIR)
    os.chdir(PROTOTYPE_DIR)
    import imageio_ffmpeg
    import torch
    import torchvision.transforms as T
    from cosmos_tokenizer.video_lib import CausalVideoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dec = CausalVideoTokenizer(checkpoint_dec=CHECKPOINT_DEC).to(device)

    indices = torch.tensor(token_ids, dtype=torch.int64, device=device).view(1, *CHUNK_GRID)
    with torch.no_grad():
        out = dec.decode(indices)  # (1, 3, T, H, W), range ~[-1, 1]

    out = ((out.float() + 1.0) / 2.0).clamp(0, 1).squeeze(0)  # (3, T, H, W)
    n_frames = out.shape[1]

    frame_dir = f"/tmp/cosmos_decode_{os.getpid()}"
    os.makedirs(frame_dir, exist_ok=True)
    to_pil = T.ToPILImage()
    for i in range(n_frames):
        to_pil(out[:, i, :, :].cpu()).save(f"{frame_dir}/frame_{i:02d}.png")

    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg_bin, "-y", "-framerate", str(fps), "-i", f"{frame_dir}/frame_%02d.png",
         "-vf", "scale=320:320:flags=neighbor", "-pix_fmt", "yuv420p", output_path],
        check=True, capture_output=True,
    )
    for i in range(n_frames):
        os.remove(f"{frame_dir}/frame_{i:02d}.png")
    os.rmdir(frame_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", help="Comma-separated list of exactly 200 raw cosmos ids")
    ap.add_argument("--input-jsonl", help="Flattened JSONL file to pull tokens from")
    ap.add_argument("--record-id", help="video_id/id field to select within --input-jsonl")
    ap.add_argument("--chunk-index", type=int, default=0,
                     help="Which 200-token chunk (in order of appearance) to decode, 0-indexed")
    ap.add_argument("--fps", type=int, default=6, help="Output mp4 framerate (decoded frames are few, slow fps for visibility)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.tokens:
        token_ids = [int(x) for x in args.tokens.split(",")]
    elif args.input_jsonl and args.record_id:
        token_ids = load_tokens_from_jsonl(args.input_jsonl, args.record_id, args.chunk_index)
    else:
        ap.error("Provide either --tokens or (--input-jsonl and --record-id)")

    decode_cosmos_chunk(token_ids, args.output, fps=args.fps)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
