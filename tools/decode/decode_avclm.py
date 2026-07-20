#!/usr/bin/env python3
"""
General-purpose AVC-LM detokenizer -- turns `<avc_lm>` tokens back into a
real, byte-exact H.264 video segment (unlike decode_cosmos.py/seed2, this is
NOT a lossy/generative neural reconstruction -- AVC-LM is plain BPE over the
raw bytes of an H.264 elementary stream, so decoding is exact reconstruction
of the original bitstream, which ffmpeg can then decode/transcode normally).

Real bug found + fixed here (20/07/2026, see PROGRESS_VI.md): the vendored
`avc_lm_v2/tokenizer.json` has no `decoder` component configured
(`tok.decoder is None`). Calling the library's own `Tokenizer.decode(ids)`
therefore falls back to naively joining each token's piece WITH A LITERAL
SPACE CHARACTER -- which corrupts arbitrary binary data (verified: ffmpeg
then fails with "non-existing PPS 0 referenced", i.e. a mangled H.264
stream). Each vocab entry here is a literal latin-1 substring of the
original bytes (no ByteLevel unicode remapping, no pre-tokenizer) -- so the
fix is to concatenate `tok.id_to_token(id)` for every id directly, with NO
separator, then `.encode("latin-1")` to get the exact original bytes back.
Verified round-trip on a real 1,327-token OmniVideo-100K avc_lm chunk:
ffmpeg decodes it cleanly (returncode 0, 8 real frames).

NOTE: this project's flatten scripts (pipeline_pose/phase7_flatten.py,
flatten_step_a_video.py, ...) always DISCARD the avc_lm payload in their
final output -- by design, to keep token counts down (see those scripts'
docstrings). So avc_lm tokens only exist in *pre-flatten* raw Step A output
(e.g. omnivideo_100k_video_flat/step_a_rank_*.jsonl, or FineVideo's
equivalent raw Step A files) -- not in any of the datasets we've published.

Usage:
    # From a raw list of ints:
    python tools/decode/decode_avclm.py --tokens 263,107,70,196,... --output out.mp4

    # From a raw (pre-flatten) Step A JSONL record (extracts the Nth <avc_lm> block):
    python tools/decode/decode_avclm.py --input-jsonl step_a_rank_0.jsonl --record-id VIDEO_ID \
        --chunk-index 0 --output out.mp4
"""
import argparse
import json
import re
import subprocess

from tokenizers import Tokenizer

PROTOTYPE_DIR = "/e/project1/reformo/nguyen38/prototype"
TOKENIZER_JSON = f"{PROTOTYPE_DIR}/avc_lm_v2/tokenizer.json"

_AVC_LM_BLOCK_RE = re.compile(r"<avc_lm>(.*?)</avc_lm>", re.DOTALL)


def extract_chunk_tokens(text: str, chunk_index: int) -> list:
    """Pull out the Nth <avc_lm>...</avc_lm> block's raw digit-token ids
    from a *pre-flatten* Step A record (final/flattened records never have
    avc_lm -- see module docstring)."""
    blocks = _AVC_LM_BLOCK_RE.findall(text)
    if chunk_index >= len(blocks):
        raise ValueError(f"Requested chunk {chunk_index} but only {len(blocks)} <avc_lm> blocks in this record")
    return [int(x) for x in blocks[chunk_index].split() if x.isdigit()]


def _record_text(rec: dict) -> str:
    """Flat records (OmniVideo-100K): {"text": ...}. Raw FineVideo Step A
    records (training_ready_rank_*.jsonl) instead nest per-activity
    `video_tokens` under scenes[].activities[] -- concatenate all of them."""
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


def decode_avclm_chunk(token_ids: list, output_path: str) -> None:
    import imageio_ffmpeg

    tok = Tokenizer.from_file(TOKENIZER_JSON)
    # NOT tok.decode(token_ids) -- see module docstring for why that's broken here.
    decoded_str = "".join(tok.id_to_token(i) for i in token_ids)
    h264_bytes = decoded_str.encode("latin-1")

    h264_path = f"/tmp/avclm_decode_{__import__('os').getpid()}.h264"
    with open(h264_path, "wb") as f:
        f.write(h264_bytes)

    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [ffmpeg_bin, "-y", "-i", h264_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path],
        capture_output=True, text=True,
    )
    __import__("os").remove(h264_path)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (returncode {result.returncode}):\n{result.stderr[-2000:]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", help="Comma-separated list of avc_lm BPE ids")
    ap.add_argument("--input-jsonl", help="Pre-flatten Step A JSONL file to pull tokens from")
    ap.add_argument("--record-id", help="video_id/id field to select within --input-jsonl")
    ap.add_argument("--chunk-index", type=int, default=0,
                     help="Which <avc_lm> block (0-indexed, in appearance order) to decode")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.tokens:
        token_ids = [int(x) for x in args.tokens.split(",")]
    elif args.input_jsonl and args.record_id:
        token_ids = load_tokens_from_jsonl(args.input_jsonl, args.record_id, args.chunk_index)
    else:
        ap.error("Provide either --tokens or (--input-jsonl and --record-id)")

    decode_avclm_chunk(token_ids, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
