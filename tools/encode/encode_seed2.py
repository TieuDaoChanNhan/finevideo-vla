#!/usr/bin/env python3
"""
Seed2 encoder -- turns a real image into `<seed2_N>` tokens (0-8191, no
offset), the reverse of tools/decode/decode_seed2.py. Reuses that module's
_load_seed2_tokenizer() (same runtime patches: the transformers import move
+ BertLMHeadModel.cls=None guard) rather than re-deriving them -- the public
ontocord/seed2 repo's own seed2_tokenizer.py still has both bugs unpatched
(verified 2026-07-23), so any fresh download needs these regardless of
whether you're encoding or decoding.

Preprocessing: Seed2Tokenizer.encode_image() does its own internal resize to
224x224 (CLIP-style Resize+Normalize, see seed2_tokenizer.py's `self.processor`)
-- pass a PIL image straight through, no manual resize needed first.

Usage:
    python tools/encode/encode_seed2.py --image photo.jpg
    # prints 32 raw ids; wrap as <seed2> <seed2_N> ... </seed2> to splice
    # into a prompt for this model (v2's convention -- no offset needed)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "decode"))
from decode_seed2 import _load_seed2_tokenizer, NUM_IMAGE_TOKENS  # noqa: E402


def encode_image(image_path: str) -> list:
    from PIL import Image

    # _load_seed2_tokenizer() os.chdir()s -- resolve a relative image_path
    # against the original cwd *before* that happens, same class of bug
    # already fixed once in decode_seed2.py's own --output handling.
    image_path = os.path.abspath(image_path)

    Seed2Tokenizer, seed2_dir = _load_seed2_tokenizer()
    tokenizer = Seed2Tokenizer.from_pretrained(seed2_dir).eval()
    if hasattr(tokenizer, "cuda") and __import__("torch").cuda.is_available():
        tokenizer = tokenizer.cuda()

    image = Image.open(image_path).convert("RGB")
    ids = tokenizer.encode_image(image_pil=image)
    ids = ids.view(-1).tolist()

    bad = [t for t in ids if not (0 <= t < NUM_IMAGE_TOKENS)]
    if bad:
        raise ValueError(f"encode_image produced out-of-range ids: {bad[:5]}... (expected [0, {NUM_IMAGE_TOKENS}))")
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="Path to a real image file")
    args = ap.parse_args()

    ids = encode_image(args.image)
    print(f"{len(ids)} seed2 tokens:")
    print(",".join(str(i) for i in ids))
    print()
    print("As a prompt fragment:")
    print("<seed2> " + " ".join(f"<seed2_{i}>" for i in ids) + " </seed2>")


if __name__ == "__main__":
    main()
