#!/usr/bin/env python3
"""
Flatten Phase 5 (adaptive PCHIP, window=24) output into training-ready flat
JSONL, one record per (category, seq_id, person_id) track -- text-only
instruction -> <agent> tokens, no video/cosmos (Harmony4D has no matching
video pipeline run; see 2026-07-23 discussion). Same flat {"id","text"}
schema as data_prep/laion_emotional_roleplay/tokenize_snac.py, not
FineVideo's hierarchical merge -- there's no other modality to merge with.

Instruction text = hand-written per-category sentence (guaranteed correct,
e.g. "Two people grappling on the ground.") + real per-sequence VLM caption
from caption_harmony4d.py (data_prep/harmony4d/caption_harmony4d.py,
Qwen2.5-VL-3B-Instruct, same method as pipeline_pose/caption_finevideo.py),
2026-07-23. Combined rather than caption-only: the VLM caption alone was
inconsistent (sometimes correctly says "practicing martial arts", sometimes
generic "motion-capture session" -- the frame doesn't always land on clear
action) -- category text guarantees the activity is never wrong, caption adds
real per-clip variety on top.

Oversampling (2026-07-23, Van Khue's decision): Harmony4D is ~2.8M tokens
vs FineVideo-VLA's ~32B -- at natural ratio it's essentially invisible in an
epoch despite being the highest-quality (multi-camera ground truth, not
monocular-estimated) agent-token source in the mix. Each record is repeated
OVERSAMPLE_FACTOR times (unique "id" suffix per copy, otherwise byte-
identical) so it gets meaningfully more exposure. 20x chosen as a starting
balance between "actually seen enough to matter" and "not so repeated it's
pure memorization risk" (416 unique tracks -> effectively see each ~20x/epoch,
same order of magnitude as normal multi-epoch training, not per-epoch spam).
Revisit once the final full-mix token budget is known.

Usage:
    python3 data_prep/harmony4d/flatten_harmony4d.py
"""
import json
import os

INPUT_DIR = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs/harmony4d_agent_tokens"
CAPTIONS_PATH = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs/harmony4d_captions.jsonl"
OUTPUT_PATH = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs/harmony4d_flat.jsonl"
OVERSAMPLE_FACTOR = 20

CATEGORY_TEXT = {
    "01_hugging": "Two people hugging each other.",
    "02_grappling": "Two people grappling on the ground.",
    "03_grappling2": "Two people grappling on the ground.",
    "04_sword_part1": "Two people sword fighting.",
    "04_sword_part2": "Two people sword fighting.",
    "05_sword2": "Two people sword fighting.",
    "06_sword3": "Two people sword fighting.",
    "07_ballroom": "Two people ballroom dancing together.",
    "08_ballroom2": "Two people ballroom dancing together.",
    "09_karate": "Two people sparring in karate.",
    "10_karate2": "Two people sparring in karate.",
    "11_karate3": "Two people sparring in karate.",
    "12_mma": "Two people sparring in mixed martial arts.",
    "13_mma2": "Two people sparring in mixed martial arts.",
    "14_mma3": "Two people sparring in mixed martial arts.",
    "15_mma4": "Two people sparring in mixed martial arts.",
    "16_mma5": "Two people sparring in mixed martial arts.",
}


def category_of(video_id: str) -> str:
    # video_id = "<category>_<seq_id>_<person_id>", category itself may
    # contain underscores (e.g. "04_sword_part1") -- match longest known
    # category prefix rather than naively splitting on "_".
    for cat in sorted(CATEGORY_TEXT, key=len, reverse=True):
        if video_id.startswith(cat + "_"):
            return cat
    raise ValueError(f"Unrecognized category prefix for video_id={video_id!r}")


def seq_id_of(video_id: str, cat: str) -> str:
    # video_id = "<cat>_<seq_id>_<person_id>" -- seq_id is everything between
    # cat and the trailing "_ariaNN" person suffix.
    rest = video_id[len(cat) + 1:]
    return rest.rsplit("_", 1)[0]


def load_captions() -> dict:
    captions = {}
    if not os.path.exists(CAPTIONS_PATH):
        print(f"WARNING: {CAPTIONS_PATH} not found -- using category text only, no VLM caption.")
        return captions
    for line in open(CAPTIONS_PATH):
        r = json.loads(line)
        captions[(r["category"], r["seq_id"])] = r["caption"]
    print(f"Loaded {len(captions)} captions")
    return captions


def main():
    captions = load_captions()

    files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith("_tokens.jsonl"))
    print(f"{len(files)} track files found")

    n_records = n_windows_total = n_tokens_total = n_with_caption = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for fname in files:
            video_id = fname[: -len("_tokens.jsonl")]
            cat = category_of(video_id)
            seq_id = seq_id_of(video_id, cat)
            text_instr = CATEGORY_TEXT[cat]

            caption = captions.get((cat, seq_id))
            if caption:
                text_instr = f"{text_instr} {caption}"
                n_with_caption += 1

            recs = [json.loads(l) for l in open(os.path.join(INPUT_DIR, fname))]
            recs.sort(key=lambda r: r["window_id"])
            if not recs:
                continue

            agent_blocks = " ".join(f"<agent> {r['token_str']} </agent>" for r in recs)
            flat_text = f"USER: {text_instr} ASSISTANT:\n{agent_blocks}"
            n_tok = sum(len(r["token_str"].split()) for r in recs)

            for copy_idx in range(OVERSAMPLE_FACTOR):
                out.write(json.dumps(
                    {"id": f"{video_id}_os{copy_idx}", "text": flat_text}, ensure_ascii=False
                ) + "\n")

            n_records += 1
            n_windows_total += len(recs)
            n_tokens_total += n_tok

    print(f"\n{n_records} unique tracks, {n_with_caption} with real VLM caption")
    print(f"Oversample factor: {OVERSAMPLE_FACTOR}x -> {n_records * OVERSAMPLE_FACTOR} records written -> {OUTPUT_PATH}")
    print(f"{n_windows_total} total windows/track, ~{n_tokens_total} unique agent tokens "
          f"(x{OVERSAMPLE_FACTOR} effective = ~{n_tokens_total * OVERSAMPLE_FACTOR:,} tokens seen/epoch, whitespace-split estimate)")


if __name__ == "__main__":
    main()
