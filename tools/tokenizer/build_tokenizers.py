#!/usr/bin/env python3
"""
Build VLA tokenizers with SNAC support.

Two outputs:
  current  — load existing tokenizer_vla_adaptive (GPT-NeoX-20b, 144,215 vocab)
             + add 12,290 SNAC tokens → tokenizer_vla_adaptive_v2
  qwen3    — load Qwen3 base tokenizer
             + add ALL VLA tokens (93,938 existing + 12,290 SNAC = 106,228)
             → tokenizer_vla_qwen3
  all      — both

Usage:
    python tools/tokenizer/build_tokenizers.py --mode current
    python tools/tokenizer/build_tokenizers.py --mode qwen3
    python tools/tokenizer/build_tokenizers.py --mode all
"""

import argparse
import os
from transformers import AutoTokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR          = "/p/data1/mmlaion/nguyen38/3d-human-pose"
EXISTING_TOK_DIR  = "/p/data1/mmlaion/shared/vla/tokenizer_vla_adaptive"
QWEN3_BASE_DIR    = os.path.join(BASE_DIR, "vocab/qwen3_tokenizer")
OUTPUT_CURRENT    = "/p/data1/mmlaion/shared/vla/tokenizer_vla_adaptive_v2"
OUTPUT_QWEN3      = "/p/data1/mmlaion/shared/vla/tokenizer_vla_qwen3"

# ── Token generation ──────────────────────────────────────────────────────────

JOINT_NAMES = [
    "pelvis", "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "spine", "thorax", "nose", "head_top",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_shoulder", "r_elbow", "r_wrist",
]

def snac_tokens() -> list[str]:
    """12,290 SNAC tokens: 2 wrappers + 12,288 <snac_N>."""
    toks = ["<snac>", "</snac>"]
    # Level 0:    codes[0][i] + 128266  →  128266..132361
    toks += [f"<snac_{i + 128266}>" for i in range(4096)]
    # Level 1 even: codes[1][2i] + 132362  →  132362..136457
    toks += [f"<snac_{i + 132362}>" for i in range(4096)]
    # Level 1 odd:  codes[1][2i+1] + 144650  →  144650..148745
    toks += [f"<snac_{i + 144650}>" for i in range(4096)]
    return toks


def all_vla_tokens() -> list[str]:
    """Full VLA token list: 93,938 existing + 12,290 SNAC = 106,228 total."""
    toks = []

    # ── Modality wrapper tags ──────────────────────────────────────────────────
    toks += [
        "<seed2>", "</seed2>",
        "<cosmos>", "</cosmos>",
        "<avc_lm>", "</avc_lm>",
        "<agent>", "</agent>",
        "<start_cosmo>", "</start_cosmo>",
        "<start_avclm>", "</start_avclm>",
    ]
    # Joint open/close wrappers (17 × 2 = 34)
    for name in JOINT_NAMES:
        toks += [f"<{name}>", f"</{name}>"]

    # ── Video tokens ───────────────────────────────────────────────────────────
    toks += [f"<agent_{i}>"  for i in range(256)]       # legacy agent tokens
    toks += [f"<avclm_{i}>"  for i in range(8192)]
    toks += [f"<seed2_{i}>"  for i in range(8192)]
    toks += [f"<cosmos_{i}>" for i in range(64000)]

    # ── Pose tokens ───────────────────────────────────────────────────────────
    toks += [f"<fps_{i}>" for i in range(1, 61)]        # <fps_1> .. <fps_60>
    for name in JOINT_NAMES:
        toks += [f"<{name}_x_{n}>" for n in range(256)]
        toks += [f"<{name}_y_{n}>" for n in range(256)]
        toks += [f"<{name}_z_{n}>" for n in range(256)]
        toks += [f"<{name}_t_{n}>" for n in range(8)]

    # ── SNAC tokens ───────────────────────────────────────────────────────────
    toks += snac_tokens()

    return toks


# ── Verification ──────────────────────────────────────────────────────────────

SPOT_CHECK = [
    "<seed2_1137>",
    "<cosmos_58567>",
    "<pelvis_x_128>",
    "<fps_30>",
    "<agent>",
    "<snac_128266>",   # L0 first
    "<snac_132362>",   # L1A first
    "<snac_144650>",   # L1B first
    "<snac_132361>",   # L0 last
    "<snac_148745>",   # L1B last
    "<snac>",
    "</snac>",
]

def verify(tok, label: str):
    print(f"\n── Spot check: {label} (vocab size {len(tok)}) ──")
    all_ok = True
    for token in SPOT_CHECK:
        ids = tok.encode(token, add_special_tokens=False)
        atomic = (len(ids) == 1)
        status = "✓" if atomic else "✗ SPLIT"
        print(f"  {status}  {token:30s} → {ids}")
        if not atomic:
            all_ok = False
    if all_ok:
        print("  All tokens atomic ✓")
    else:
        print("  WARNING: some tokens are being split!")


# ── Build functions ───────────────────────────────────────────────────────────

def build_current(output_dir: str):
    """
    Load existing tokenizer_vla_adaptive (144,215 vocab) and add SNAC tokens.
    Only adds tokens not already present — safe to re-run.
    """
    print(f"\n=== Building tokenizer_vla_adaptive_v2 ===")
    print(f"  Base:   {EXISTING_TOK_DIR}")
    print(f"  Output: {output_dir}")

    tok = AutoTokenizer.from_pretrained(EXISTING_TOK_DIR)
    print(f"  Loaded base vocab size: {len(tok)}")

    new_toks = snac_tokens()
    # Filter out any already present (safe re-run)
    to_add = [t for t in new_toks if t not in tok.get_vocab()]
    print(f"  SNAC tokens to add: {len(to_add)} / {len(new_toks)}")

    if to_add:
        added = tok.add_tokens(to_add, special_tokens=True)
        print(f"  Actually added: {added}")

    print(f"  Final vocab size: {len(tok)}")
    os.makedirs(output_dir, exist_ok=True)
    tok.save_pretrained(output_dir)
    print(f"  Saved → {output_dir}")

    verify(tok, "tokenizer_vla_adaptive_v2")


def build_qwen3(output_dir: str):
    """
    Load Qwen3 base tokenizer and add ALL VLA tokens (existing + SNAC).
    """
    print(f"\n=== Building tokenizer_vla_qwen3 ===")
    print(f"  Base:   {QWEN3_BASE_DIR}")
    print(f"  Output: {output_dir}")

    tok = AutoTokenizer.from_pretrained(QWEN3_BASE_DIR)
    print(f"  Loaded Qwen3 base vocab size: {len(tok)}")

    new_toks = all_vla_tokens()
    to_add = [t for t in new_toks if t not in tok.get_vocab()]
    print(f"  VLA tokens to add: {len(to_add)} / {len(new_toks)}")

    if to_add:
        added = tok.add_tokens(to_add, special_tokens=True)
        print(f"  Actually added: {added}")

    print(f"  Final vocab size: {len(tok)}")
    os.makedirs(output_dir, exist_ok=True)
    tok.save_pretrained(output_dir)
    print(f"  Saved → {output_dir}")

    verify(tok, "tokenizer_vla_qwen3")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["current", "qwen3", "all"], default="all")
    p.add_argument("--output-current", default=OUTPUT_CURRENT)
    p.add_argument("--output-qwen3",   default=OUTPUT_QWEN3)
    args = p.parse_args()

    if args.mode in ("current", "all"):
        build_current(args.output_current)

    if args.mode in ("qwen3", "all"):
        build_qwen3(args.output_qwen3)

    print("\nDone.")


if __name__ == "__main__":
    main()
