#!/usr/bin/env python3
"""
VLA model sanity check — token atomicity + greedy generation.

Run on login node with one GPU:
    module --force purge
    module load Stages/2025 GCC/13.3.0 Python/3.12.3 CUDA/12 PyTorch/2.5.1 torchvision/0.20.1
    source /e/project1/reformo/nguyen38/env_stable_vla/bin/activate
    python tools/eval/eval_vla_sanity.py

Or specify a different checkpoint:
    python tools/eval/eval_vla_sanity.py --model-path output_vla/vla_adaptive/hf/iter_0001000
"""

import argparse
import json
import re
import sys
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from decode_agent_tokens import decode, to_json, JOINT_NAMES, JOINT_INDEX

# 2026-07-23: output_vla moved project1 -> data1 (freed inodes for the
# project1 quota crisis) -- verified byte-for-byte match before the project1
# copy was deleted, see PROGRESS_VI.md same-day entry.
MODEL_PATH = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/output_vla/vla_adaptive/hf/iter_0002032"
TOKENIZER_PATH = "/e/project1/reformo/nguyen38/3d-human-pose/vocab/tokenizer_vla_adaptive"


# ── Test 1: Token atomicity ──────────────────────────────────────────────────

ATOMICITY_TESTS = [
    "<seed2_1137>", "<seed2_0>", "<seed2_8191>",
    "<cosmos_58567>", "<cosmos_0>", "<cosmos_63999>",
    "<avclm_100>", "<avclm_0>", "<avclm_8191>",
    "<fps_30>",
    "<pelvis>", "</pelvis>",
    "<pelvis_t_0>", "<pelvis_t_7>",
    "<pelvis_x_0>", "<pelvis_x_128>", "<pelvis_x_255>",
    "<r_wrist_y_200>", "<l_shoulder_z_50>",
    "<r_hip>", "</r_hip>",
    "<head_top>", "</head_top>",
]


def test_atomicity(tokenizer):
    print("=" * 60)
    print("TEST 1: Token atomicity")
    print("=" * 60)

    passed = 0
    failed = 0
    for token_str in ATOMICITY_TESTS:
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        ok = len(ids) == 1
        status = "OK" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
            decoded = [tokenizer.decode([i]) for i in ids]
            print(f"  {status}  {token_str:30s} -> {len(ids)} ids: {decoded}")

    if failed == 0:
        print(f"  All {passed} tokens are atomic (single token ID each)")
    else:
        print(f"\n  {passed} passed, {failed} FAILED")

    print()
    return failed == 0


# ── Test 2: Greedy generation ────────────────────────────────────────────────

PROMPTS = [
    {
        "name": "full_prompt",
        "prompt": (
            "### Title: Home cooking tutorial\n"
            "### Context: A person chops vegetables on a cutting board in a kitchen.\n"
            "### Keywords: cooking, kitchen, food preparation\n"
        ),
        "max_new_tokens": 2000,
        "description": "Full training-like prompt, expect seed2 -> cosmos -> avclm -> agent sequence",
    },
    {
        "name": "agent_continuation",
        "prompt": (
            "### Context: Person raises both arms above head.\n"
            "<seed2_3758> <seed2_2157> <seed2_3402> <cosmos_58567> <cosmos_56071> "
            "<fps_30> <pelvis> <pelvis_t_0> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128>"
        ),
        "max_new_tokens": 500,
        "description": "Partial agent block, expect model completes the 17-joint sequence",
    },
    {
        "name": "agent_from_scratch",
        "prompt": (
            "### Context: Person stands up from a chair.\n"
            "<seed2_6750> <seed2_680> <seed2_5141> <seed2_7543> <seed2_680> "
            "<seed2_1940> <seed2_6707> <seed2_6258> <seed2_2900> <seed2_2157> "
            "<seed2_2157> <seed2_6707> <seed2_3488> <seed2_7543> <seed2_5141> "
            "<seed2_4815> <seed2_2315> <seed2_1940> <seed2_2157> <seed2_4682> "
            "<seed2_6707> <seed2_4773> <seed2_6707> <seed2_2157> <seed2_891> "
            "<seed2_3488> <seed2_6506> <seed2_7940> <seed2_1603> <seed2_3488> "
            "<seed2_6834> <seed2_6861>"
        ),
        "max_new_tokens": 2000,
        "description": "Real seed2 block from training data, expect model continues with cosmos/agent",
    },
]


def classify_token(token_str):
    if re.match(r"<seed2_\d+>", token_str):
        return "seed2"
    if re.match(r"<cosmos_\d+>", token_str):
        return "cosmos"
    if re.match(r"<avclm_\d+>", token_str):
        return "avclm"
    if re.match(r"<fps_\d+>", token_str):
        return "agent"
    if re.match(r"<\w+_[txyz]_\d+>", token_str):
        return "agent"
    if re.match(r"</?[a-z_]+>", token_str):
        return "agent"
    return "text"


def validate_agent_structure(tokens):
    """Check if generated agent tokens have valid structure."""
    errors = []
    warnings = []

    vla_tokens = [t for t in tokens if classify_token(t) == "agent"]
    if not vla_tokens:
        errors.append("No agent tokens generated")
        return errors, warnings

    fps_count = sum(1 for t in vla_tokens if t.startswith("<fps_"))
    if fps_count == 0:
        errors.append("No <fps_N> token found")

    found_joints = set()
    for t in vla_tokens:
        m = re.match(r"^<([a-z_]+)>$", t)
        if m and m.group(1) in JOINT_INDEX:
            found_joints.add(m.group(1))

    if found_joints:
        missing = set(JOINT_NAMES) - found_joints
        if missing:
            warnings.append(f"Missing joints: {sorted(missing)}")
    else:
        warnings.append("No joint open tags found")

    for t in vla_tokens:
        for dim in ("x", "y", "z"):
            m = re.match(rf"<\w+_{dim}_(\d+)>", t)
            if m and int(m.group(1)) > 255:
                errors.append(f"Out-of-range value in {t}")
        m = re.match(r"<\w+_t_(\d+)>", t)
        if m and int(m.group(1)) > 7:
            errors.append(f"Frame index > 7 in {t}")

    return errors, warnings


def test_generation(model, tokenizer, device, max_new_tokens=500):
    print("=" * 60)
    print("TEST 2: Greedy generation")
    print("=" * 60)

    all_ok = True

    for pinfo in PROMPTS:
        print(f"\n--- {pinfo['name']} ---")
        print(f"  {pinfo['description']}")

        input_ids = tokenizer.encode(pinfo["prompt"], return_tensors="pt").to(device)
        attention_mask = torch.ones_like(input_ids)
        prompt_len = input_ids.shape[1]
        gen_len = pinfo.get("max_new_tokens", max_new_tokens)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_len,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0, prompt_len:]
        generated_tokens = [tokenizer.decode([tid]).strip() for tid in generated_ids.tolist()]

        # Token type breakdown
        counts = {}
        for t in generated_tokens:
            cat = classify_token(t)
            counts[cat] = counts.get(cat, 0) + 1

        print(f"  Generated {len(generated_tokens)} tokens")
        print(f"  Breakdown: {counts}")

        # Show first 40 tokens
        preview = " ".join(generated_tokens[:40])
        if len(generated_tokens) > 40:
            preview += " ..."
        print(f"  First 40: {preview}")

        # Show where modality transitions happen
        transitions = []
        prev_cat = None
        for i, t in enumerate(generated_tokens):
            cat = classify_token(t)
            if cat != prev_cat:
                transitions.append((i, cat, t))
                prev_cat = cat
        if len(transitions) > 1:
            print(f"  Transitions: ", end="")
            print(" -> ".join(f"{cat}@{i}" for i, cat, _ in transitions[:10]))

        # Validate agent token structure
        errors, warnings = validate_agent_structure(generated_tokens)
        for e in errors:
            print(f"  ERROR: {e}")
            all_ok = False
        for w in warnings:
            print(f"  WARN:  {w}")

        # Try to decode agent tokens if present
        agent_str = " ".join(t for t in generated_tokens if classify_token(t) == "agent")
        if agent_str:
            try:
                trajectories = decode(agent_str)
                if trajectories:
                    result = to_json(trajectories)
                    print(f"  Decoded {result['n_windows']} pose windows, "
                          f"shape {result['shape']}, "
                          f"range {result['value_range_m']} m")
                    for w in result["windows"][:2]:
                        movers = ", ".join(f"{n} {d:.3f}m" for n, d in w["top_movers"][:3])
                        print(f"    Window {w['window']}: {movers}")
                else:
                    print("  Could not decode agent tokens into poses")
            except Exception as e:
                print(f"  Decode error: {e}")

    print()
    return all_ok


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="VLA model sanity check")
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--tokenizer-path", default=TOKENIZER_PATH)
    p.add_argument("--max-new-tokens", type=int, default=500)
    args = p.parse_args()

    print(f"Model:     {args.model_path}")
    print(f"Tokenizer: {args.tokenizer_path}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    atomicity_ok = test_atomicity(tokenizer)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device} (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {n_params / 1e9:.2f}B params, vocab {model.config.vocab_size}")
    print()

    generation_ok = test_generation(model, tokenizer, device, args.max_new_tokens)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Token atomicity: {'PASS' if atomicity_ok else 'FAIL'}")
    print(f"  Generation:      {'PASS' if generation_ok else 'NEEDS REVIEW'}")
    print()


if __name__ == "__main__":
    main()
