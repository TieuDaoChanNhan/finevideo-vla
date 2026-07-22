#!/usr/bin/env python3
"""
One-off: test whether the full_chain_from_scratch results (identical seed2
blocks every time, agent never reached even at 4000 tokens) are specific to
that one prompt's wording -- possibly because it's close to a memorized
training example -- or a general pattern regardless of prompt content.

Uses a deliberately different scene (skateboarding, not cooking) in the same
Title/Context/Keywords format convention, greedy only, same 4000-token budget
as the previous long run.

Usage:
    python tools/eval/gen_full_chain_novel_prompt.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from eval_vla_v2_media import (
    MODEL_PATH, TOKENIZER_PATH, _patched_tokenizer_path, decode_media, REPO_ROOT,
)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = (
    "### Title: Backyard skateboard trick\n"
    "### Context: A teenager attempts a kickflip on a skateboard in a driveway.\n"
    "### Keywords: skateboarding, sports, outdoor\n"
)
MAX_NEW_TOKENS = 4000

OUT_DIR = os.path.join(REPO_ROOT, "samples", "qwen3_1.7b_vla_v2_eval",
                        "2026-07-22_full_chain_novel_prompt")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Tokenizer: {TOKENIZER_PATH}")
    print(f"Model:     {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(_patched_tokenizer_path(TOKENIZER_PATH))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device} (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    model.eval()

    input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)
    prompt_len = input_ids.shape[1]

    print(f"Generating (greedy, max_new_tokens={MAX_NEW_TOKENS})...")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, attention_mask=attention_mask, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=False)
    full_text = PROMPT + generated_text

    text_path = os.path.join(OUT_DIR, "input_output.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("CATEGORY: full_chain (novel prompt, not the cooking-tutorial one)\n")
        f.write("DESCRIPTION: Same Title/Context/Keywords format as full_prompt, deliberately "
                "different scene content, to check whether the cooking-tutorial run's identical-"
                "seed2-blocks/no-agent result is prompt-specific (possible memorization) or general.\n")
        f.write("MODE: greedy\n\n")
        f.write("INPUT PROMPT (fed to the model):\n")
        f.write(PROMPT + "\n\n")
        f.write("MODEL OUTPUT (everything generated after the prompt):\n")
        f.write(generated_text + "\n")
    print(f"Saved: {text_path}")

    media_status = decode_media(text_path, full_text, OUT_DIR)
    for modality, (ok, detail) in media_status.items():
        print(f"  decode[{modality}]: {'OK' if ok else 'FAIL'} - {detail}")

    import re
    n_seed2_blocks = len(re.findall(r"<seed2>(.*?)</seed2>", full_text, re.DOTALL))
    unique_seed2 = len(set(re.findall(r"<seed2>(.*?)</seed2>", full_text, re.DOTALL)))
    n_fps = len(re.findall(r"<fps_\d+>", full_text))
    print(f"\nseed2 blocks: {n_seed2_blocks} total, {unique_seed2} unique")
    print(f"agent/fps tokens: {n_fps}")


if __name__ == "__main__":
    main()
