#!/usr/bin/env python3
"""
Cosmos video-chunk encoder -- turns 8 real video frames into `<cosmos_N>`
tokens (200 raw ids, no offset), the reverse of tools/decode/decode_cosmos.py.
Reproduces the OLD (vla-1.7b-qwen3-v2 training-era) preprocessing convention
EXACTLY, not the current/newer aspect-preserving pipeline convention
(2026-07-23 pivot, 896 tokens/chunk) -- recovered from git history
(commit edf25393, before 38d8e5f2 switched to aspect-preserving):

    Resize((160,160))          # direct squash/stretch to 160x160, NOT an
                                # aspect-preserving crop -- distorts aspect
                                # ratio on purpose (this model's own training
                                # convention, don't "fix" it here)
    ToTensor()
    Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])   # -> range [-1, 1]

8 frames -> stack -> permute(1,0,2,3) -> (3,8,160,160) -> unsqueeze(0) ->
(1,3,8,160,160) -> CausalVideoTokenizer.encode() -> (1,2,10,10) == 200 ids,
checkpoint nvidia/Cosmos-Tokenizer-DV8x16x16 (encoder.jit -- same repo
decode_cosmos.py already downloads decoder.jit from).

Usage:
    python tools/encode/encode_cosmos.py --frames f0.png f1.png ... f7.png --output tokens.txt
    # exactly 8 frame image paths, in temporal order
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "decode", "vendor"))

TARGET_SIZE = 160
N_FRAMES = 8
COSMOS_HF_REPO = "nvidia/Cosmos-Tokenizer-DV8x16x16"
_LOCAL_CHECKPOINT_ENC = "/e/project1/reformo/nguyen38/prototype/pretrained_ckpts/Cosmos-Tokenizer-DV8x16x16/encoder.jit"


def _resolve_checkpoint_enc() -> str:
    if os.path.exists(_LOCAL_CHECKPOINT_ENC):
        return _LOCAL_CHECKPOINT_ENC
    from huggingface_hub import hf_hub_download
    print(f"Local checkpoint not found -- downloading encoder.jit from {COSMOS_HF_REPO} "
          f"(~350MB, cached for future runs)...")
    return hf_hub_download(repo_id=COSMOS_HF_REPO, filename="encoder.jit")


def encode_frames(frame_paths: list) -> list:
    if len(frame_paths) != N_FRAMES:
        raise ValueError(f"Expected exactly {N_FRAMES} frame paths, got {len(frame_paths)}")

    import torch
    import torchvision.transforms as T
    from PIL import Image
    from cosmos_tokenizer.video_lib import CausalVideoTokenizer

    transform = T.Compose([
        T.Resize((TARGET_SIZE, TARGET_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    frames = [transform(Image.open(p).convert("RGB")) for p in frame_paths]
    video = torch.stack(frames, dim=0)          # (T, 3, H, W)
    video = video.permute(1, 0, 2, 3)           # (3, T, H, W)
    video = video.unsqueeze(0)                  # (1, 3, T, H, W)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = CausalVideoTokenizer(checkpoint_enc=_resolve_checkpoint_enc()).to(device)
    with torch.no_grad():
        indices = enc.encode(video.to(device))[0]  # (1, 2, 10, 10)

    ids = indices.reshape(-1).tolist()
    if len(ids) != 200:
        raise ValueError(f"Expected 200 raw ids, got {len(ids)} -- checkpoint/shape mismatch")
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", nargs=8, required=True, metavar="FRAME",
                     help="Exactly 8 frame image paths, in temporal order")
    ap.add_argument("--output", help="Optional: write comma-separated ids to this file")
    args = ap.parse_args()

    ids = encode_frames(args.frames)
    out = ",".join(str(i) for i in ids)
    print(f"{len(ids)} cosmos tokens:")
    print(out)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
