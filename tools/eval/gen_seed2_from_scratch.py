#!/usr/bin/env python3
"""
Have qwen3_1.7b_vla_v2 generate its OWN <seed2_N> tokens + caption from scratch
(no ground-truth seed2 given, unlike eval_vla_v2_sanity.py's "image_caption"
test which feeds a REAL seed2 block and checks the caption). Here the model
invents the image tokens too -- this tests whether the model's own seed2
tokens actually decode to something that matches its own caption, i.e.
image-generation self-consistency, not just image->text binding.

Prompt is just "<seed2>", matching the synth_llava training format (record
starts directly with the seed2 block, no USER:/ASSISTANT: chat wrapper --
see data_prep/synth_llava/tokenize_seed2.py).

Saves one .txt with the prompt and raw model output (exactly one <seed2>...
</seed2> span across the two, so decode_seed2.py's --text-file block-scan
picks up exactly the model's own tokens, not a duplicate), then decodes those
tokens to a PNG via decode_seed2.py.

Usage:
    python tools/eval/gen_seed2_from_scratch.py
"""
import json
import os
import shutil
import sys
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from eval_vla_v2_sanity import MODEL_PATH, TOKENIZER_PATH


def _patched_tokenizer_path(src: str) -> str:
    """tokenizer_config.json ships extra_special_tokens as a list; transformers
    4.57.6 (this env) expects a dict and crashes on load. Same scratch-copy
    shim used elsewhere in this repo (tools/eval/eval_vla_v2_sanity.py callers,
    the deleted add_snac_l2_tokens.py) -- canonical tokenizer left untouched."""
    with open(os.path.join(src, "tokenizer_config.json")) as f:
        cfg = json.load(f)
    if not isinstance(cfg.get("extra_special_tokens"), list):
        return src
    tmp = tempfile.mkdtemp(prefix="tokenizer_vla_qwen3_patched_")
    for name in os.listdir(src):
        shutil.copy2(os.path.join(src, name), os.path.join(tmp, name))
    cfg["extra_special_tokens"] = {}
    with open(os.path.join(tmp, "tokenizer_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return tmp

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "samples", "qwen3_1.7b_vla_v2_eval")
TEXT_OUT = os.path.join(OUT_DIR, "seed2_from_scratch_gen.txt")
IMAGE_OUT = os.path.join(OUT_DIR, "seed2_from_scratch_decoded.png")

PROMPT = "<seed2>"
MAX_NEW_TOKENS = 350


def main():
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
    prompt_len = input_ids.shape[1]

    print(f"Generating (greedy, max_new_tokens={MAX_NEW_TOKENS})...")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    n_seed2 = generated_text.count("<seed2_")
    has_caption = "<caption>" in generated_text and "</caption>" in generated_text
    caption = ""
    if has_caption:
        caption = generated_text.split("<caption>", 1)[1].split("</caption>", 1)[0].strip()

    os.makedirs(OUT_DIR, exist_ok=True)
    # Deliberately exactly one <seed2>...</seed2> span across INPUT+OUTPUT combined
    # (prompt supplies the open tag, generated_text supplies the close tag) -- no
    # duplicate "full text" section, so decode_seed2.py's block-scan regex can't
    # double-count tokens by matching the same span twice.
    with open(TEXT_OUT, "w", encoding="utf-8") as f:
        f.write("INPUT PROMPT (fed to the model):\n")
        f.write(PROMPT + "\n\n")
        f.write("MODEL OUTPUT (everything generated after the prompt, greedy decoding):\n")
        f.write(generated_text + "\n\n")
        f.write(f"seed2 token count: {n_seed2}\n")
        f.write(f"caption found: {has_caption}\n")
        if caption:
            f.write(f"caption text: {caption}\n")
    print(f"Saved generation to: {TEXT_OUT}")
    print(f"  seed2 tokens generated: {n_seed2}")
    print(f"  caption found: {has_caption}")
    if caption:
        print(f"  caption: {caption}")

    if n_seed2 == 0:
        print("No seed2 tokens generated -- skipping decode.")
        return

    print(f"\nDecoding the model's own {n_seed2} generated seed2 tokens to an image...")
    decode_script = os.path.join(os.path.dirname(__file__), "..", "decode", "decode_seed2.py")
    ret = os.system(f'python "{decode_script}" --text-file "{TEXT_OUT}" --output "{IMAGE_OUT}"')
    if ret != 0:
        print(f"decode_seed2.py exited with code {ret}")


if __name__ == "__main__":
    main()
