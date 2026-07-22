#!/usr/bin/env python3
"""
SNAC detokenizer -- turns `<snac_N>` "listen format" tokens back into a real
audio waveform (24kHz mono), using the same `hubertsiuzdak/snac_24khz` model
used to encode them (data_prep/laion_emotional_roleplay/tokenize_snac.py,
pipeline_pose/snac_finevideo.py).

Listen format encodes only 2 of the model's 3 hierarchical codebook levels
(base 12.5Hz level 0, and one 25Hz level 1 -- the finest 50Hz level 2 is
dropped entirely to save tokens). 3 tokens per base frame, in fixed order
(L0, L1_even, L1_odd), with these offsets added to the raw codebook index
(0-4095 each):

    OFFSET_L0  = 128266  (raw + 128266)
    OFFSET_L1A = 132362  (raw + 128266 + 4096)
    OFFSET_L1B = 144650  (raw + 128266 + 4*4096)

Since level 2 was never encoded, this decoder reconstructs it as all-zero
codes (index 0 in each of the 4x-oversampled slots) -- `SNAC.decode()` needs
all 3 levels present, so this is the only reconstruction available without
retraining/re-deriving the missing fine detail. Expect this to sound coarser
than the original clip (level 2 carries the finest timbral detail), same
lossy tradeoff already documented for `decode_cosmos.py`'s video reconstruction.

Usage:
    # From a raw list of ints (must be a multiple of 3):
    python tools/decode/decode_snac.py --tokens 128266,132850,145181,... --output out.wav

    # From a flattened JSONL record, pulling every <snac>...</snac> block:
    python tools/decode/decode_snac.py --input-jsonl path.jsonl --record-id ID --output out.wav
"""
import argparse
import json
import re
import sys

OFFSET_L0 = 128266
OFFSET_L1A = 128266 + 4096
OFFSET_L1B = 128266 + 4 * 4096
SAMPLE_RATE = 24000
SNAC_MODEL = "hubertsiuzdak/snac_24khz"

_SNAC_ATOMIC_RE = re.compile(r"<snac_(\d+)>")
_SNAC_BLOCK_RE = re.compile(r"<snac>(.*?)</snac>", re.DOTALL)


def extract_snac_tokens(text: str) -> list:
    """Pull every <snac_N> id inside every <snac>...</snac> block, in order.
    Falls back to scanning the whole text if no <snac>...</snac> wrapper is present."""
    blocks = _SNAC_BLOCK_RE.findall(text)
    source = " ".join(blocks) if blocks else text
    return [int(x) for x in _SNAC_ATOMIC_RE.findall(source)]


def load_tokens_from_jsonl(path: str, record_id: str) -> list:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = rec.get("video_id", rec.get("id"))
            if rid == record_id:
                return extract_snac_tokens(rec.get("text", ""))
    raise KeyError(f"record_id={record_id!r} not found in {path}")


def decode_snac_tokens(token_ids: list, output_path: str) -> None:
    """token_ids: raw <snac_N> ids (with offsets), length must be a multiple of 3."""
    if len(token_ids) % 3 != 0:
        raise ValueError(f"Expected a multiple of 3 tokens (L0,L1a,L1b triplets), got {len(token_ids)}")
    if not token_ids:
        raise ValueError("No snac tokens to decode")

    import torch
    import soundfile as sf
    from snac import SNAC

    n0 = len(token_ids) // 3
    c0 = torch.zeros(1, n0, dtype=torch.long)
    c1 = torch.zeros(1, 2 * n0, dtype=torch.long)
    c2 = torch.zeros(1, 4 * n0, dtype=torch.long)  # level 2 was never encoded -- zero-fill

    # This function assumes strict positional cycling (index%3==0 -> L0,
    # ==1 -> L1a, ==2 -> L1b) matches the actual band each id belongs to --
    # true for well-formed <snac>...</snac> blocks, but NOT guaranteed for a
    # free-floating/unwrapped token span assembled by extract_snac_tokens()'s
    # whole-text fallback (e.g. a generation that never closes </snac>,
    # spliced with earlier/later snac fragments elsewhere in the text). A
    # mismatched id produces an out-of-[0,4095) codebook index, which SNAC's
    # embedding lookup on GPU turns into an opaque
    # "CUDA error: device-side assert triggered" instead of a real error
    # message (hit for real 2026-07-22, roleplay_speech/sample eval run --
    # see samples/qwen3_1.7b_vla_v2_eval/2026-07-22_full_eval/SUMMARY.md).
    # Validate up front so the failure is legible.
    for i in range(n0):
        raw_l0, raw_l1a, raw_l1b = token_ids[3 * i], token_ids[3 * i + 1], token_ids[3 * i + 2]
        r0, r1a, r1b = raw_l0 - OFFSET_L0, raw_l1a - OFFSET_L1A, raw_l1b - OFFSET_L1B
        for pos, (name, tok, raw) in enumerate([("L0", raw_l0, r0), ("L1a", raw_l1a, r1a), ("L1b", raw_l1b, r1b)]):
            if not (0 <= raw < 4096):
                raise ValueError(
                    f"Triplet {i} position {pos} ({name}): token <snac_{tok}> decodes to raw "
                    f"codebook index {raw}, outside valid [0, 4096). This id likely belongs to a "
                    f"different band than its position implies (offsets: L0={OFFSET_L0}, "
                    f"L1a={OFFSET_L1A}, L1b={OFFSET_L1B}) -- often means the input tokens aren't "
                    f"one clean <snac>...</snac> block (e.g. spliced fragments from an unwrapped "
                    f"or unclosed generation)."
                )
        c0[0, i] = r0
        c1[0, 2 * i] = r1a
        c1[0, 2 * i + 1] = r1b

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SNAC.from_pretrained(SNAC_MODEL).eval().to(device)

    with torch.inference_mode():
        audio = model.decode([c0.to(device), c1.to(device), c2.to(device)])  # (1, 1, samples)

    waveform = audio.squeeze().float().cpu().numpy()
    sf.write(output_path, waveform, SAMPLE_RATE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", help="Comma-separated list of raw <snac_N> ids (must be multiple of 3)")
    ap.add_argument("--input-jsonl", help="Flattened JSONL file to pull tokens from")
    ap.add_argument("--record-id", help="video_id/id field to select within --input-jsonl")
    ap.add_argument("--text-file", help="Plain text file containing <snac_N> tokens anywhere in it")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.tokens:
        token_ids = [int(x) for x in args.tokens.split(",")]
    elif args.input_jsonl and args.record_id:
        token_ids = load_tokens_from_jsonl(args.input_jsonl, args.record_id)
    elif args.text_file:
        token_ids = extract_snac_tokens(open(args.text_file, encoding="utf-8").read())
    else:
        ap.error("Provide --tokens, --text-file, or (--input-jsonl and --record-id)")

    print(f"Decoding {len(token_ids)} snac tokens ({len(token_ids) // 3} base frames, "
          f"~{len(token_ids) / 3 / 12.5:.2f}s @ 12.5Hz base rate)...")
    decode_snac_tokens(token_ids, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
