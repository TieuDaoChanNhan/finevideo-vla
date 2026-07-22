#!/usr/bin/env python3
"""
Full media eval for qwen3_1.7b_vla_v2: feed a fixed set of prompts (spanning
continuation / from-scratch / full-chain), generate with both greedy and
sampling, and decode every non-text modality span found in each output to
real media (png/mp4/wav) using the project's existing decoders -- all
verified working end-to-end as of 2026-07-22 (see REPORT.md #31 follow-ups):
    tools/decode/decode_seed2.py    (seed2 -> image, generative/diffusion)
    tools/decode/decode_cosmos.py   (cosmos -> video, deterministic codec)
    tools/decode/decode_snac.py     (snac -> audio, deterministic codec)
    tools/eval/decode_agent_tokens.py + tools/visualize/render_agent_pose.py
                                     (agent -> 3D pose video)

Each test gets its own folder with exactly one input_output.txt (full
prompt + full raw generation, nothing truncated) plus whatever media
decoded successfully next to it -- meant to be read/watched directly by
anyone on the team, not just parsed by a script. A SUMMARY.md at the end
links every test and records decode pass/fail per modality.

Usage:
    python tools/eval/eval_vla_v2_media.py
    python tools/eval/eval_vla_v2_media.py --run-dir samples/qwen3_1.7b_vla_v2_eval/my_run
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from eval_vla_v2_sanity import MODEL_PATH, TOKENIZER_PATH, PROMPTS as SANITY_PROMPTS

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DECODE_SEED2 = os.path.join(REPO_ROOT, "tools", "decode", "decode_seed2.py")
DECODE_COSMOS = os.path.join(REPO_ROOT, "tools", "decode", "decode_cosmos.py")
DECODE_SNAC = os.path.join(REPO_ROOT, "tools", "decode", "decode_snac.py")
RENDER_AGENT = os.path.join(REPO_ROOT, "tools", "visualize", "render_agent_pose.py")

COSMOS_CHUNK_TOKENS = 200


def _patched_tokenizer_path(src: str) -> str:
    """transformers==4.57.6 (this env) chokes on extra_special_tokens as a
    list instead of a dict -- same scratch-copy shim used elsewhere in this
    repo (canonical tokenizer left untouched)."""
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


def _by_name(name):
    return next(p for p in SANITY_PROMPTS if p["name"] == name)


# ── Prompt suite ─────────────────────────────────────────────────────────────
# category: continuation (real ground-truth tokens fed in, has a known-good
#   next step to compare against) / from_scratch (bare tag or text only, model
#   invents everything) / full_chain (one long open-ended prompt, model must
#   walk seed2->cosmos->snac->agent transitions on its own).

PROMPTS = [
    {**_by_name("image_caption"), "category": "continuation",
     "max_new_tokens": 300},
    {**_by_name("agent_continuation"), "category": "continuation",
     "max_new_tokens": 500},
    {**_by_name("roleplay_speech"), "category": "continuation",
     "max_new_tokens": 500},
    {
        "name": "seed2_from_scratch", "category": "from_scratch",
        "prompt": "<seed2>",
        "max_new_tokens": 350,
        "description": "Bare seed2 open tag, no real tokens -- model invents the image itself.",
    },
    {
        "name": "cosmos_from_scratch", "category": "from_scratch",
        "prompt": "### Context: A person walks across a room.\n<cosmos>",
        "max_new_tokens": 250,
        "description": "Text context + bare cosmos open tag, no real tokens -- model invents 1 video chunk.",
    },
    {
        "name": "agent_from_scratch", "category": "from_scratch",
        "prompt": "### Context: Person raises both arms above head.\n<fps_30>",
        "max_new_tokens": 400,
        "description": "Text context + bare fps tag, no real pose values -- model invents the full pose.",
    },
    {
        "name": "full_chain_from_scratch", "category": "full_chain",
        "prompt": _by_name("full_prompt")["prompt"],
        # 2026-07-22: a 2000-token greedy run never reached <agent> -- cosmos
        # alone burned 787/2000 tokens (~40%) before the budget ran out (see
        # 07_full_chain_from_scratch_greedy/COMPARISON_REPORT.md). Bumped as
        # high as the model actually supports: config.json's
        # max_position_embeddings is a hard 4096 (matches training seq_length),
        # so prompt_len + max_new_tokens must stay under that -- 4000 leaves
        # ~90 tokens of headroom for this prompt (~40-50 tokens).
        "max_new_tokens": 4000,
        "description": "Long open text prompt only -- model must walk seed2->cosmos->snac->agent transitions unaided.",
    },
]


def classify_present(text: str) -> dict:
    return {
        "seed2": bool(re.search(r"<seed2_\d+>", text)),
        "cosmos": len(re.findall(r"<cosmos_(\d+)>", text)),
        "snac": len(re.findall(r"<snac_(\d+)>", text)),
        "agent": bool(re.search(r"<fps_\d+>", text)),
    }


def run_subprocess(cmd: list) -> tuple:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return result.returncode == 0, (result.stdout[-2000:] + result.stderr[-2000:])


def decode_media(text_path: str, full_text: str, out_dir: str) -> dict:
    """Decode every modality present in full_text, writing media into out_dir.
    Returns {modality: (ok, detail)} for the SUMMARY table."""
    status = {}
    present = classify_present(full_text)

    if present["seed2"]:
        ok, detail = run_subprocess([
            "python", DECODE_SEED2, "--text-file", text_path,
            "--output", os.path.join(out_dir, "decoded_seed2.png"),
        ])
        if ok:
            # decode_seed2.py writes decoded_seed2.png for a single 32-token block,
            # or decoded_seed2_0.png/_1.png/... for multiple blocks (each image
            # decoded separately since Seed2Tokenizer only handles 32 at a time).
            multi = sorted(f for f in os.listdir(out_dir) if re.match(r"decoded_seed2_\d+\.png$", f))
            produced = multi if multi else (["decoded_seed2.png"] if os.path.exists(os.path.join(out_dir, "decoded_seed2.png")) else [])
            detail = ", ".join(produced) if produced else "no image files found despite exit 0"
        else:
            detail = detail.splitlines()[-1] if detail else "failed"
        status["seed2"] = (ok, detail)

    if present["cosmos"] >= COSMOS_CHUNK_TOKENS:
        all_ids = [int(x) for x in re.findall(r"<cosmos_(\d+)>", full_text)]
        n_chunks = len(all_ids) // COSMOS_CHUNK_TOKENS
        chunk_results = []
        for i in range(n_chunks):
            chunk = all_ids[i * COSMOS_CHUNK_TOKENS:(i + 1) * COSMOS_CHUNK_TOKENS]
            out_path = os.path.join(out_dir, f"decoded_cosmos_chunk{i}.mp4")
            ok, detail = run_subprocess([
                "python", DECODE_COSMOS, "--tokens", ",".join(map(str, chunk)), "--output", out_path,
            ])
            chunk_results.append((ok, f"decoded_cosmos_chunk{i}.mp4" if ok else detail.splitlines()[-1] if detail else "failed"))
        status["cosmos"] = (all(r[0] for r in chunk_results),
                             f"{sum(r[0] for r in chunk_results)}/{n_chunks} chunks: " + "; ".join(r[1] for r in chunk_results))
    elif present["cosmos"]:
        status["cosmos"] = (False, f"only {present['cosmos']} cosmos tokens, need >= {COSMOS_CHUNK_TOKENS} for 1 chunk")

    if present["snac"] >= 3:
        ok, detail = run_subprocess([
            "python", DECODE_SNAC, "--text-file", text_path,
            "--output", os.path.join(out_dir, "decoded_snac.wav"),
        ])
        status["snac"] = (ok, "decoded_snac.wav" if ok else detail.splitlines()[-1] if detail else "failed")

    if present["agent"]:
        ok, detail = run_subprocess([
            "python", RENDER_AGENT, "--input", text_path,
            "--output", os.path.join(out_dir, "decoded_agent_pose.mp4"), "--fps", "5",
        ])
        status["agent"] = (ok, "decoded_agent_pose.mp4" if ok else detail.splitlines()[-1] if detail else "failed")

    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--only", default=None, help="Comma-separated substrings; only run prompts whose name matches one")
    ap.add_argument("--modes", default="greedy,sample", help="Comma-separated subset of: greedy,sample")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="Override every matched prompt's max_new_tokens")
    args = ap.parse_args()

    prompts = PROMPTS
    if args.only:
        needles = args.only.split(",")
        prompts = [p for p in PROMPTS if any(n in p["name"] for n in needles)]
        if not prompts:
            raise SystemExit(f"--only {args.only!r} matched no prompt names (have: {[p['name'] for p in PROMPTS]})")
    if args.max_new_tokens:
        prompts = [{**p, "max_new_tokens": args.max_new_tokens} for p in prompts]
    modes = [m for m in (("greedy", False), ("sample", True)) if m[0] in args.modes.split(",")]

    run_dir = args.run_dir or os.path.join(
        REPO_ROOT, "samples", "qwen3_1.7b_vla_v2_eval", f"{date.today().isoformat()}_full_eval")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print(f"Tokenizer: {TOKENIZER_PATH}")
    print(f"Model:     {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(_patched_tokenizer_path(TOKENIZER_PATH))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device} (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    model.eval()

    summary_rows = []

    for idx, pinfo in enumerate(prompts, start=1):
        for mode, sample in modes:
            test_name = f"{idx:02d}_{pinfo['name']}_{mode}"
            test_dir = os.path.join(run_dir, test_name)
            os.makedirs(test_dir, exist_ok=True)
            print(f"\n{'=' * 60}\n{test_name}  [{pinfo['category']}]\n{'=' * 60}")
            print(f"  {pinfo['description']}")

            input_ids = tokenizer.encode(pinfo["prompt"], return_tensors="pt").to(device)
            attention_mask = torch.ones_like(input_ids)
            prompt_len = input_ids.shape[1]

            gen_kwargs = dict(max_new_tokens=pinfo["max_new_tokens"], pad_token_id=tokenizer.eos_token_id)
            if sample:
                torch.manual_seed(42)
                gen_kwargs.update(do_sample=True, temperature=0.8, top_p=0.9, repetition_penalty=1.3)
            else:
                gen_kwargs.update(do_sample=False)

            with torch.no_grad():
                output_ids = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)

            generated_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=False)
            full_text = pinfo["prompt"] + generated_text

            text_path = os.path.join(test_dir, "input_output.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(f"CATEGORY: {pinfo['category']}\n")
                f.write(f"DESCRIPTION: {pinfo['description']}\n")
                f.write(f"MODE: {mode}\n\n")
                f.write("INPUT PROMPT (fed to the model):\n")
                f.write(pinfo["prompt"] + "\n\n")
                if "ground_truth" in pinfo:
                    f.write("GROUND TRUTH (what real training data has next):\n")
                    f.write(pinfo["ground_truth"] + "\n\n")
                f.write("MODEL OUTPUT (everything generated after the prompt):\n")
                f.write(generated_text + "\n")

            print(f"  Saved: {text_path}")

            media_status = decode_media(text_path, full_text, test_dir)
            for modality, (ok, detail) in media_status.items():
                print(f"  decode[{modality}]: {'OK' if ok else 'FAIL'} - {detail}")

            summary_rows.append({
                "test": test_name, "category": pinfo["category"], "mode": mode,
                "media": media_status,
            })

    summary_path = os.path.join(run_dir, "SUMMARY.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# qwen3_1.7b_vla_v2 media eval -- {date.today().isoformat()}\n\n")
        f.write(f"Model: `{MODEL_PATH}`\n\n")
        f.write("| Test | Category | Mode | seed2 | cosmos | snac | agent |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for row in summary_rows:
            def cell(m):
                if m not in row["media"]:
                    return "-"
                ok, detail = row["media"][m]
                return f"{'PASS' if ok else 'FAIL'}: {detail}"
            f.write(f"| [{row['test']}]({row['test']}/input_output.txt) | {row['category']} | {row['mode']} | "
                     f"{cell('seed2')} | {cell('cosmos')} | {cell('snac')} | {cell('agent')} |\n")
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
