#!/usr/bin/env python3
"""
Upload VLA 1.7B Qwen3 v2 model to HuggingFace.

Usage:
    export HF_TOKEN=hf_...
    python tools/upload/upload_vla_v2_model.py
"""

import os
import tempfile
import shutil

from huggingface_hub import HfApi

REPO_ID = "EmpathicRobotics/vla-1.7b-qwen3-v2"
MODEL_DIR = "/e/project1/reformo/nguyen38/output_vla/qwen3_1.7b_vla_v2/hf/iter_0007632"

README = """\
---
license: apache-2.0
language:
  - en
tags:
  - robotics
  - vla
  - vision-language-action
  - 3d-pose
  - qwen3
  - megatron
  - multimodal
pipeline_tag: text-generation
library_name: transformers
---

# VLA 1.7B — Qwen3 v2

A 1.7B parameter Vision-Language-Action model, migrated to a **Qwen3** backbone and
trained on a **5-source, ~32B-token multimodal mix** (video, 3D pose, audio, image+caption).
This is the project's first Qwen3-based VLA model, and the first trained after fixing
the "stuck in one modality" failure mode found in the previous model.

## Key facts

| | |
|---|---|
| **Architecture** | Qwen3 (28 layers, hidden 2048, intermediate 6144, 16 attn heads / 8 KV heads (GQA), qk-layernorm, RoPE θ=1e6, tied embeddings) |
| **Parameters** | 1.94B (including embeddings for 257,920 vocab) |
| **Vocab size** | 257,920 (Qwen3 base ~151,669 + 106,232 VLA tokens, padded) |
| **Tokenizer** | [EmpathicRobotics/tokenizer-vla-qwen3](https://huggingface.co/EmpathicRobotics/tokenizer-vla-qwen3) |
| **Training data** | ~32.01B tokens across 5 sources: FineVideo-VLA v6, MixtureVitae-Omni, OmniVideo-100K, synth-llava, emotional-roleplay |
| **Training** | 7,632 iters (1 epoch), 64 nodes × 4 GH200 GPUs, global batch 1024, seq len 4096 |
| **Final loss** | Train: 1.694, Val: 1.7526 (PPL 5.77), Test: 1.7722 (PPL 5.88) |
| **Precision** | bf16 |
| **Context length** | 4,096 tokens |

## What this model does

Given a text prompt (activity description, image seed2 block, or partial modality
sequence), the model generates an interleaved multimodal token sequence spanning
6 categories it was trained on:

```
<seed2_N> ...                          # 1 FPS semantic image/video keyframes (vocab 8192)
<cosmos_N> ... </cosmos>               # 8-frame spatial video tokens (vocab 64000)
<snac_N> ... </snac>                   # SNAC audio codec tokens (12,288)
<speech> ... </speech>                 # inline spoken-dialogue text
<caption> ... </caption>               # inline visual caption text
<agent> <fps_30> <pelvis> ... </agent> # 3D human pose, 17 H36M joints
```

## Progress vs. the previous model

The first model ([vla-1.7b-pab-spline-adaptive](https://huggingface.co/EmpathicRobotics/vla-1.7b-pab-spline-adaptive))
passed agent-completion but **failed modality transitions**: it stayed in `seed2` mode and
never transitioned to `cosmos`/`avclm`/`agent` from text alone. This model no longer has
that failure mode — it transitions freely across all 6 trained categories, in both
greedy and sampled decoding.

Strongest evidence: given **only** 32 real `<seed2_N>` tokens from a held-out image record
(no other text hint), the model generated a topically-correct caption closely matching the
real ground truth, then closed `</caption><|im_end|>` cleanly — genuine image↔text
cross-modal binding, not template noise.

It can also produce full agent (3D pose) blocks that decode to valid, non-degenerate
coordinates, and — verified for the first time on this model's own generation, not just
training data — `cosmos` video tokens that decode to a real, playable video via
[`Cosmos-Tokenizer-DV8x16x16`](https://github.com/NVIDIA/Cosmos-Tokenizer), and `snac`
audio tokens that decode to a real, non-silent waveform via
[SNAC](https://github.com/hubertsiuzdak/snac) (`hubertsiuzdak/snac_24khz`).

## Known limitations

- **Greedy decoding can degenerate into repeated-token loops** inside long `cosmos`
  runs (e.g. the same token repeating 6-8 times), which can burn the generation budget
  before reaching `<fps_N>`/`agent`. Sampling with `repetition_penalty>1` mitigates this.
- **Sampling trades accuracy for diversity**: in the image-captioning test, sampled
  generation occasionally hallucinated details (e.g. an invented name) not present in
  the source image; greedy decoding did not.
- **`cosmos` tokens dominate generation**: aggregated across all test prompts, `cosmos`
  is 61-77% of all non-text VLA tokens produced (vs. a minority share for
  agent/seed2/snac combined). This is largely structural (one cosmos chunk costs a fixed
  200 tokens vs. ~1-4 tokens for the others), but it does mean cosmos runs can consume
  most of a generation's token budget before reaching `<fps_N>`/`agent`.
- **`avc_lm` tokens are essentially unused** — discarded at the data-flatten stage before
  training (to control token count), so the model rarely if ever produces them.
- **`seed2`→image reconstruction has not been demonstrated** — no decoder for that
  direction exists yet in the project repo. (`cosmos`→video and `snac`→audio decoders
  both exist and have been verified against this model's own generated output.)
- **Evaluation so far is qualitative** (manual inspection of generated tokens/decoded
  media) — no MPJPE, BLEU/CIDEr, or closed-loop task-success metric has been run yet.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "EmpathicRobotics/vla-1.7b-qwen3-v2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("EmpathicRobotics/tokenizer-vla-qwen3")

prompt = (
    "### Context: Person raises both arms above head.\\n"
    "<seed2_3758> <seed2_2157> <cosmos_58567> "
    "<fps_30> <pelvis> <pelvis_t_0> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128>"
)
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
output = model.generate(
    input_ids, max_new_tokens=500,
    do_sample=True, temperature=0.8, top_p=0.9, repetition_penalty=1.3,
)
print(tokenizer.decode(output[0]))
```

### Decoding agent tokens to 3D poses

```python
# pip install scipy
from decode_agent_tokens import decode  # from the 3d-human-pose repo, tools/eval/

generated_text = tokenizer.decode(output[0])
trajectories = decode(generated_text)  # list of (8, 17, 3) ndarrays
```

### Decoding cosmos tokens to video

```python
# from the 3d-human-pose repo, tools/decode/decode_cosmos.py
# python tools/decode/decode_cosmos.py --tokens 58345,57843,... --output out.mp4
# (requires exactly 200 raw cosmos ids per 8-frame chunk)
```

## Training details

### Loss curve

| Iter | Loss |
|---|---|
| 50 | 6.472 |
| 500 | 2.840 |
| 1000 | 2.154 |
| 2000 | 1.953 |
| 4000 | 1.826 |
| 6000 | 1.767 |
| 7600 | 1.694 |
| 7632 (val) | 1.7526 (PPL 5.77) |
| 7632 (test) | 1.7722 (PPL 5.88) |

### Config

- **Batch**: GBS 1024, seq_len 4096 → 32.01B tokens trained (exactly 1 epoch)
- **Infrastructure**: 64 nodes × 4 GH200 GPUs (256 total), ~284 TFLOP/s/GPU, ~21,800 tok/s/GPU
- **Framework**: Megatron-LM via oellm-autoexp

### Data mix

| Source | Tokens |
|---|---|
| MixtureVitae-Omni | 20.39B |
| FineVideo-VLA v6 | 10.93B |
| OmniVideo-100K (video) | 0.54B |
| synth-llava | 0.10B |
| emotional-roleplay (SNAC TTS) | 0.05B |
| **Total** | **~32.01B** |

## Citation

```bibtex
@misc{empathicrobotics2026vlaqwen3,
  title={VLA 1.7B Qwen3 v2: Multi-Source Multimodal Vision-Language-Action Pretraining},
  author={EmpathicRobotics},
  year={2026},
  url={https://huggingface.co/EmpathicRobotics/vla-1.7b-qwen3-v2}
}
```
"""


def main():
    api = HfApi()

    print(f"Creating repo: {REPO_ID}")
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        for f in os.listdir(MODEL_DIR):
            src = os.path.join(MODEL_DIR, f)
            dst = os.path.join(tmp, f)
            print(f"  Copying {f} ({os.path.getsize(src) / 1e6:.1f} MB)")
            shutil.copy2(src, dst)

        with open(os.path.join(tmp, "README.md"), "w") as f:
            f.write(README)

        print(f"\nUploading to {REPO_ID}...")
        api.upload_folder(
            folder_path=tmp,
            repo_id=REPO_ID,
            repo_type="model",
            create_pr=False,
            commit_message="Upload VLA 1.7B Qwen3 v2 with model card",
        )

    print(f"\nDone: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
