#!/usr/bin/env python3
"""
Generalized version of gen_full_chain_novel_prompt.py -- takes an arbitrary
Title/Context/Keywords prompt via CLI instead of a hardcoded one, greedy,
same 4000-token budget (near this checkpoint's max_position_embeddings=4096
ceiling), decodes every modality present via eval_vla_v2_media.decode_media().

Usage:
    python tools/eval/gen_full_chain_prompt.py \
        --title "Morning walk" \
        --context "A man walks down a street." \
        --keywords "walking, street, pedestrian" \
        --out-name man_walking
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from eval_vla_v2_media import MODEL_PATH, TOKENIZER_PATH, _patched_tokenizer_path, decode_media, REPO_ROOT

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_NEW_TOKENS = 4000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--keywords", required=True)
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = ap.parse_args()

    prompt = f"### Title: {args.title}\n### Context: {args.context}\n### Keywords: {args.keywords}\n"
    out_dir = os.path.join(REPO_ROOT, "samples", "qwen3_1.7b_vla_v2_eval", f"2026-07-22_full_chain_{args.out_name}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Tokenizer: {TOKENIZER_PATH}")
    print(f"Model:     {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(_patched_tokenizer_path(TOKENIZER_PATH))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device} (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)
    prompt_len = input_ids.shape[1]

    print(f"Prompt:\n{prompt}")
    print(f"Generating (greedy, max_new_tokens={args.max_new_tokens})...")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, attention_mask=attention_mask, max_new_tokens=args.max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=False)
    full_text = prompt + generated_text

    text_path = os.path.join(out_dir, "input_output.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(f"CATEGORY: full_chain (custom prompt: {args.out_name})\n")
        f.write("MODE: greedy\n\n")
        f.write("INPUT PROMPT (fed to the model):\n")
        f.write(prompt + "\n\n")
        f.write("MODEL OUTPUT (everything generated after the prompt):\n")
        f.write(generated_text + "\n")
    print(f"Saved: {text_path}")

    media_status = decode_media(text_path, full_text, out_dir)
    for modality, (ok, detail) in media_status.items():
        print(f"  decode[{modality}]: {'OK' if ok else 'FAIL'} - {detail}")

    caps = re.findall(r"<caption>(.*?)</caption>", full_text, re.DOTALL)
    speeches = re.findall(r"<speech>(.*?)</speech>", full_text, re.DOTALL)
    n_fps = len(re.findall(r"<fps_\d+>", full_text))
    print(f"\nCAPTION: {caps}")
    print(f"SPEECH: {speeches}")
    print(f"agent/fps tokens: {n_fps}")

    if n_fps:
        try:
            from decode_agent_tokens import decode as decode_agent
            trajs = decode_agent(full_text)
            for i, t in enumerate(trajs):
                pelvis_range = t[:, 0, :].max(axis=0) - t[:, 0, :].min(axis=0)
                print(f"  pose window {i}: pelvis movement range (x,y,z) = {pelvis_range}")
        except Exception as e:
            print(f"  (could not compute pose movement range: {e})")


if __name__ == "__main__":
    main()
