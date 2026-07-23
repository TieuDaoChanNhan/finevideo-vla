#!/usr/bin/env python3
"""
Seed2 detokenizer -- turns `<seed2_N>` tokens back into an actual image.

Unlike decode_cosmos.py/decode_avclm.py (both have a dedicated neural video
decoder), Seed2Tokenizer (prototype/seed2/, vendored `ontocord/seed2`) is a
Q-Former/BLIP-2-style *understanding* tokenizer -- it has no pixel decoder of
its own. Its own README documents the only reconstruction path: look up each
token's codebook embedding, then condition a diffusion img2img pipeline
(`stabilityai/stable-diffusion-2-1-unclip`) on those embeddings to *generate*
a plausible image. This is fundamentally different from cosmos/avc_lm's
reconstruction (same lossy-but-deterministic neural codec used to encode) --
here the "decode" is itself a generative sample, not a round-trip of the
original pixels. Two different tokens can plausibly decode to visually
different images even if they'd caption similarly, and the same tokens can
decode to slightly different images across runs unless a fixed seed/latent
is used (this script fixes the latent, matching the tokenizer's own
`self.latents` buffer, so decoding is reproducible run-to-run).

Token ids are raw codebook indices, 0-8191, no vocab offset (verified:
tools/tokenizer/build_tokenizers.py's `<seed2_{i}> for i in range(8192)`
and Seed2Tokenizer.num_image_tokens == 8192 agree; unlike snac's +128266
offset or cosmos's chunking, seed2 tokens map directly).

First run downloads `stabilityai/stable-diffusion-2-1-unclip` (~5GB) from
HF -- needs internet access (compute nodes reportedly do not have it, see
REPORT.md's checkpoint-conversion note; run this on a node that does, or
pre-populate $HF_HOME first).

Usage:
    # From a raw list of ints:
    python tools/decode/decode_seed2.py --tokens 6750,2157,4657,... --output out.png

    # From a flattened JSONL / text file containing <seed2_N> tokens anywhere:
    python tools/decode/decode_seed2.py --text-file path.txt --output out.png
"""
import argparse
import os
import re
import sys

PROTOTYPE_DIR = "/e/project1/reformo/nguyen38/prototype"
# 2026-07-23: prototype/ is NOT part of the public github.com/TieuDaoChanNhan/
# finevideo-vla repo (0 files git-tracked), so external users can't reach
# _LOCAL_SEED2_DIR below. Found that Seed2Tokenizer's own vendored README
# (prototype/seed2/README.md) already documents its true public home --
# `git clone https://huggingface.co/ontocord/seed2` -- confirmed for real via
# HfApi().model_info("ontocord/seed2"): same 2 checkpoint files
# (ae.safetensors, model.safetensors) + seed2_tokenizer.py are already public
# there. No redistribution/licensing decision needed on our end (unlike a
# vendor-our-own-checkpoint approach) -- just point at the existing repo.
SEED2_HF_REPO = "ontocord/seed2"
_LOCAL_SEED2_DIR = os.path.join(PROTOTYPE_DIR, "seed2")


def _resolve_seed2_dir() -> str:
    if os.path.isdir(_LOCAL_SEED2_DIR):
        return _LOCAL_SEED2_DIR
    from huggingface_hub import snapshot_download
    print(f"Local seed2 checkpoint not found -- downloading from {SEED2_HF_REPO} "
          f"(~2.6GB, cached for future runs)...")
    return snapshot_download(repo_id=SEED2_HF_REPO)


# stabilityai/stable-diffusion-2-1-unclip returns a genuine 404 (page title
# literally "404 - Hugging Face", not a gated-access page) as of 2026-07-22 --
# confirmed via HF search API that it no longer appears under the stabilityai
# org at all (Stability AI removed it, not access-gated). Using the community
# re-upload instead: same weights/pipeline class (StableUnCLIPImg2ImgPipeline,
# safetensors, openrail++ license), created 2025-11-14 specifically as a mirror.
DIFFUSION_NAME = "sd2-community/stable-diffusion-2-1-unclip"
NUM_IMAGE_TOKENS = 8192
SEED2_QUERY_LEN = 32  # fixed Q-former query length trained into Seed2Tokenizer, 1 image's worth

_SEED2_ATOMIC_RE = re.compile(r"<seed2_(\d+)>")
_SEED2_BLOCK_RE = re.compile(r"<seed2>(.*?)</seed2>", re.DOTALL)


def extract_seed2_tokens(text: str) -> list:
    """Pull every <seed2_N> id inside every <seed2>...</seed2> block, in order,
    concatenated into one flat list. Falls back to scanning the whole text if
    no <seed2>...</seed2> wrapper is present. Only meaningful as one image's
    worth of tokens if the text has exactly one block (or zero, unwrapped) --
    for multi-block text (e.g. a multi-activity generation), use
    extract_seed2_blocks() instead so each image decodes separately."""
    blocks = _SEED2_BLOCK_RE.findall(text)
    source = " ".join(blocks) if blocks else text
    return [int(x) for x in _SEED2_ATOMIC_RE.findall(source)]


def extract_seed2_blocks(text: str) -> list:
    """Like extract_seed2_tokens(), but keeps each <seed2>...</seed2> block's
    ids separate (one list per image) instead of concatenating them -- needed
    because Seed2Tokenizer only ever decodes exactly SEED2_QUERY_LEN=32 tokens
    as one image (see decode_seed2_tokens()). Falls back to treating the whole
    text as one block if no wrapper is present."""
    blocks = _SEED2_BLOCK_RE.findall(text)
    if not blocks:
        return [[int(x) for x in _SEED2_ATOMIC_RE.findall(text)]] if _SEED2_ATOMIC_RE.search(text) else []
    return [[int(x) for x in _SEED2_ATOMIC_RE.findall(b)] for b in blocks]


def _load_seed2_tokenizer():
    """Same import/shim sequence as data_prep/synth_llava/tokenize_seed2.py --
    reused verbatim rather than re-derived (see that file's docstring for why
    each shim exists: a transformers-version helper-function move, and a
    Qformer.cls=None crash).

    NOTE: tokenize_seed2.py only ever calls .encode_image() (only needs
    prototype/pipeline.py's thin Seed2Tokenizer wrapper), but decode needs
    .decode()/.from_pretrained(), which only exist on the real HF PreTrainedModel
    class in seed2_tokenizer.py itself -- pipeline.py's wrapper doesn't have
    them. Copying the import verbatim silently returned the wrong class here
    (caught 2026-07-22: AttributeError, no from_pretrained) -- must return
    _seed2_tokenizer.Seed2Tokenizer directly, not prototype/pipeline.py's.

    2026-07-23: no longer hardcodes PROTOTYPE_DIR -- uses whatever
    _resolve_seed2_dir() finds (local cluster copy, or a fresh download from
    the public ontocord/seed2 HF repo). Returns that dir alongside the class
    so callers know where to point Seed2Tokenizer.from_pretrained().

    chdir target differs by branch: init_tokenizer() inside seed2_tokenizer.py
    does BertTokenizer.from_pretrained("./seed2/bert-base-uncased") -- a
    relative lookup that expects cwd to be seed2_dir's PARENT (matching how
    the local cluster copy is laid out: PROTOTYPE_DIR/seed2/bert-base-uncased).
    Chdir'ing into seed2_dir itself (one level too deep) breaks that lookup --
    caught 2026-07-23 testing the new encode_seed2.py against the local
    branch specifically (OSError: can't load './seed2/bert-base-uncased').
    The downloaded ontocord/seed2 snapshot has no bert-base-uncased subfolder
    at all (verified via its real file listing) yet works anyway -- empirically
    that branch's relative lookup resolves some other way (not fully
    root-caused), so only the local branch needs the parent-dir chdir fix."""
    seed2_dir = _resolve_seed2_dir()
    os.chdir(PROTOTYPE_DIR if seed2_dir == _LOCAL_SEED2_DIR else seed2_dir)

    import transformers.modeling_utils as _modeling_utils
    import transformers.pytorch_utils as _pytorch_utils
    for _name in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"):
        if not hasattr(_modeling_utils, _name):
            setattr(_modeling_utils, _name, getattr(_pytorch_utils, _name))

    sys.path.insert(0, seed2_dir)
    import seed2_tokenizer as _seed2_tokenizer

    def _safe_get_output_embeddings(self):
        return None if self.cls is None else self.cls.predictions.decoder

    def _safe_set_output_embeddings(self, new_embeddings):
        if self.cls is not None:
            self.cls.predictions.decoder = new_embeddings

    for _cls in (_seed2_tokenizer.BertLMHeadModel, _seed2_tokenizer.BertForMaskedLM):
        _cls.get_output_embeddings = _safe_get_output_embeddings
        _cls.set_output_embeddings = _safe_set_output_embeddings

    return _seed2_tokenizer.Seed2Tokenizer, seed2_dir


def decode_seed2_tokens(token_ids: list, output_path: str, guidance_scale: float = 10.0,
                         num_inference_steps: int = 20) -> None:
    if not token_ids:
        raise ValueError("No seed2 tokens to decode")
    bad = [t for t in token_ids if not (0 <= t < NUM_IMAGE_TOKENS)]
    if bad:
        raise ValueError(f"Token ids out of range [0, {NUM_IMAGE_TOKENS}): {bad[:5]}...")
    # Seed2Tokenizer's Q-former was trained with a fixed 32 query tokens/image
    # (see pos_embed_image.repeat(query_output_up.shape[0], ...) in
    # seed2_tokenizer.py -- shape[0] must be 1 image's worth of queries, i.e.
    # len(token_ids)==32). Any other count desyncs from the diffusion
    # pipeline's own batch-size assumption deep in the UNet (hit for real
    # 2026-07-22: 3-token and 96-token spans both crashed with a "tensor a
    # must match tensor b" RuntimeError -- see
    # samples/qwen3_1.7b_vla_v2_eval/2026-07-22_full_eval/SUMMARY.md, tests
    # 02_agent_continuation and 07_full_chain_from_scratch). Fail clearly
    # instead of letting that opaque error surface from inside the UNet.
    if len(token_ids) != SEED2_QUERY_LEN:
        raise ValueError(
            f"Got {len(token_ids)} seed2 tokens, but Seed2Tokenizer only decodes exactly "
            f"{SEED2_QUERY_LEN} tokens at a time (1 image's fixed Q-former query length). "
            f"If this span was extracted from text containing multiple <seed2>...</seed2> blocks "
            f"(e.g. a multi-activity generation), decode each block separately."
        )

    import torch
    from diffusers import StableUnCLIPImg2ImgPipeline

    Seed2Tokenizer, seed2_dir = _load_seed2_tokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading {DIFFUSION_NAME} (first run downloads ~5GB)...")
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(DIFFUSION_NAME, torch_dtype=dtype).to(device)

    print("Loading Seed2Tokenizer...")
    tokenizer = Seed2Tokenizer.from_pretrained(seed2_dir, torch_dtype=dtype).to(device)

    # get_codebook_entry() does self.embedding(indices) with no batch handling of
    # its own -- dim 0 of `indices` becomes the batch dim downstream (repeat(),
    # etc.). A flat (N,) tensor is silently read as N separate 1-token images
    # instead of 1 image made of N query tokens, which desyncs from the diffusion
    # pipeline's own (batch=1, x2 for CFG) conditioning shape and crashes deep in
    # the UNet (caught 2026-07-22: "tensor a (2) must match tensor b (64)" for a
    # 32-token input, i.e. 32 x 2 leaking through). Needs an explicit batch dim.
    indices = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    image = tokenizer.decode(pipe, indices, guidance_scale=guidance_scale,
                              num_inference_steps=num_inference_steps)[0]
    image.save(output_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", help="Comma-separated list of raw <seed2_N> ids (0-8191)")
    ap.add_argument("--text-file", help="Plain text / JSONL file containing <seed2_N> tokens anywhere in it")
    ap.add_argument("--guidance-scale", type=float, default=10.0)
    ap.add_argument("--num-inference-steps", type=int, default=20)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # _load_seed2_tokenizer() os.chdir()s into PROTOTYPE_DIR (needs to be cwd for
    # its own relative "./seed2" lookups) -- resolve a relative --output against
    # the original cwd *before* that happens, or it silently lands in prototype/
    # instead (caught 2026-07-22: full generation succeeded, only image.save() failed).
    output_path = os.path.abspath(args.output)

    if args.tokens:
        blocks = [[int(x) for x in args.tokens.split(",")]]
    elif args.text_file:
        blocks = extract_seed2_blocks(open(args.text_file, encoding="utf-8").read())
    else:
        ap.error("Provide --tokens or --text-file")

    if not blocks:
        ap.error("No <seed2_N> tokens found")

    if len(blocks) == 1:
        print(f"Decoding {len(blocks[0])} seed2 tokens...")
        decode_seed2_tokens(blocks[0], output_path, args.guidance_scale, args.num_inference_steps)
        print(f"Saved: {output_path}")
        return

    # Multiple <seed2>...</seed2> blocks (e.g. a multi-activity generation) --
    # decode each as its own image rather than concatenating and erroring
    # (added 2026-07-22 after a 96-token/3-block span crashed decode_seed2_tokens).
    print(f"Found {len(blocks)} separate <seed2>...</seed2> blocks -- decoding each as its own image.")
    stem, ext = os.path.splitext(output_path)
    for i, block in enumerate(blocks):
        block_output = f"{stem}_{i}{ext}"
        print(f"\nBlock {i}: {len(block)} tokens -> {block_output}")
        try:
            decode_seed2_tokens(block, block_output, args.guidance_scale, args.num_inference_steps)
            print(f"Saved: {block_output}")
        except ValueError as e:
            print(f"Skipped block {i}: {e}")


if __name__ == "__main__":
    main()
