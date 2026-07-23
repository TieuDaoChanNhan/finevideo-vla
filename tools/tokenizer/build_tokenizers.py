#!/usr/bin/env python3
"""
Build VLA tokenizers with SNAC + caption/speech support.

Two outputs:
  current  — load existing tokenizer_vla_adaptive (GPT-NeoX-20b, 144,215 vocab)
             + add 12,290 SNAC tokens + 4 caption/speech wrapper tokens
             → tokenizer_vla_adaptive_v2 (rebuilt in place, not yet used by any
             completed retokenization/training run as of this change — see
             PROGRESS.md's still-unchecked "Megatron re-tokenize ... v0.3" item)
  qwen3    — load Qwen3 base tokenizer
             + add ALL VLA tokens (93,938 existing + 12,290 SNAC + 4 caption/speech)
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


def caption_speech_tokens() -> list[str]:
    """4 wrapper tokens for inline caption/speech interleaving (chunk-anchored)."""
    return ["<caption>", "</caption>", "<speech>", "</speech>"]


def snac_l2_tokens() -> list[str]:
    """16,384 new SNAC level-2 ("speak" format) tokens: 4 bands x 4096,
    2026-07-23 -- offsets corrected to match the REAL scheme used by Huu/
    Chien's production snac_gpu.py on Leonardo (pipeline_video/snac_gpu.py,
    byte-identical copy confirmed 2026-07-23), which is itself the Orpheus-
    standard SNAC packing layout. Two of these bands (136458, 140554) sit in
    what was previously assumed to be an unused gap between L1a and L1b --
    it is not a gap, it's where Leo's L2 sub-codes 0 and 1 live. The other
    two (148746, 152842) coincide numerically with the original (wrong)
    scheme's first two bands but now correctly represent Leo's L2 sub-codes
    2 and 3, not sub-codes 0 and 1. Must match
    data_prep/laion_emotional_roleplay/tokenize_snac.py's OFFSET_L2 exactly.
    MUST only ever be appended at the very end of all_vla_tokens()'s
    returned list, never inserted earlier -- see that function's docstring."""
    toks = []
    for base in (136458, 140554, 148746, 152842):
        toks += [f"<snac_{i + base}>" for i in range(4096)]
    return toks


def listen_speak_tokens() -> list[str]:
    """4 new wrapper tokens, 2026-07-23: <listen>/</listen>/<speak>/</speak>,
    replacing the generic <snac>/</snac> wrapper going forward. <listen> =
    ambient/scene audio the model is describing (FineVideo's own audio,
    always -- pipeline_pose/phase6_merge_adaptive.py's build_snac_insertion());
    <speak> = the model's own reply (laion/emotional-roleplay, always --
    data_prep/laion_emotional_roleplay/tokenize_snac.py's flatten_record()).
    Role decides the tag, not audio format (3 vs 7 tok/frame) or whether the
    voice has been cloned yet. MUST only ever be appended at the very end of
    all_vla_tokens()'s returned list, never inserted earlier."""
    return ["<listen>", "</listen>", "<speak>", "</speak>"]


def agent_t_extended_tokens() -> list[str]:
    """<{joint}_t_8> .. <{joint}_t_23> for all 17 joints, 272 tokens total --
    2026-07-22 (REPORT.md #38), needed once the agent/pose window widened
    from 8 to 24 frames ("Option 2": PCHIP control points may now land on
    any of the 24 real frames, not just a fixed 0-7 grid). MUST only ever be
    appended at the very end of all_vla_tokens()'s returned list, never
    inserted into the existing per-joint loop below (which would shift the
    id of every token generated after it -- x/y/z of later joints, all of
    snac_tokens() -- silently breaking every already-trained model's
    embedding table)."""
    toks = []
    for name in JOINT_NAMES:
        toks += [f"<{name}_t_{n}>" for n in range(8, 24)]
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
        "<caption>", "</caption>",
        "<speech>", "</speech>",
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

    # ── 2026-07-22 additions (REPORT.md #37, #38) ──────────────────────────────
    # APPENDED ONLY, after everything else -- every token above this point
    # keeps the exact same id it already has in the published
    # tokenizer_vla_qwen3 / any already-trained checkpoint's embedding table.
    # Never move these two lines earlier or insert into an existing loop.
    toks += agent_t_extended_tokens()
    toks += snac_l2_tokens()
    toks += listen_speak_tokens()

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
    "<snac_136458>",   # L2 sub-code 0 first (Leo scheme, was the old "gap")
    "<snac_152841>",   # L2 sub-code 3 last
    "<snac>",
    "</snac>",
    "<caption>",
    "</caption>",
    "<speech>",
    "</speech>",
    "<listen>",
    "</listen>",
    "<speak>",
    "</speak>",
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

    new_toks = snac_tokens() + caption_speech_tokens()
    # Filter out any already present (safe re-run)
    to_add = [t for t in new_toks if t not in tok.get_vocab()]
    print(f"  SNAC + caption/speech tokens to add: {len(to_add)} / {len(new_toks)}")

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
