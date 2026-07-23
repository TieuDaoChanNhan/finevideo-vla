#!/usr/bin/env python3
"""
Upload VLA 1.7B adaptive model to HuggingFace.

Usage:
    export HF_TOKEN=hf_...
    python tools/upload/upload_vla_model.py
"""

import os
import tempfile
import shutil

from huggingface_hub import HfApi

REPO_ID = "EmpathicRobotics/vla-1.7b-pab-spline-adaptive"
# 2026-07-23: output_vla moved project1 -> data1 (freed inodes for the
# project1 quota crisis) -- verified byte-for-byte match before the project1
# copy was deleted, see PROGRESS_VI.md same-day entry.
MODEL_DIR = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/output_vla/vla_adaptive/hf/iter_0002032"

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
  - pchip
  - megatron
pipeline_tag: text-generation
library_name: transformers
---

# VLA 1.7B — PAB-Spline Adaptive

A 1.7B parameter Vision-Language-Action model trained on the FineVideo-VLA dataset.
This model generates interleaved video tokens (Seed2, Cosmos, AVC-LM) and **adaptive PCHIP 3D human pose tokens** from activity descriptions.

## Key facts

| | |
|---|---|
| **Architecture** | OpenSci-Ref 1.7B (Llama-like, RMSNorm, SwiGLU, RoPE, QK-LayerNorm) |
| **Parameters** | 1.91B (including embeddings for 144K vocab) |
| **Vocab size** | 144,256 (50,277 base GPT-NeoX-20b + 93,938 VLA tokens, padded to 128) |
| **Tokenizer** | [EmpathicRobotics/tokenizer-vla-adaptive](https://huggingface.co/EmpathicRobotics/tokenizer-vla-adaptive) |
| **Training data** | 2.84B tokens from ~40K FineVideo YouTube videos |
| **Training** | 2,032 iters (~3 epochs), 64 nodes × 4 GH200 GPUs, WSD schedule |
| **Final loss** | Train: 1.476, Val: 1.501 (PPL 4.49), Test: 1.494 (PPL 4.45) |
| **Precision** | bf16 |
| **Context length** | 4,096 tokens |

## What this model does

Given an activity description, the model generates a multimodal token sequence:

```
### Context: Person chops vegetables on a cutting board.
<seed2_6750> <seed2_680> ...          # 1 FPS semantic keyframes (Seed2, vocab 8192)
<cosmos_58567> <cosmos_56071> ...     # 8-frame spatial tokens (Cosmos, vocab 64000)
<avclm_100> <avclm_200> ...           # 8-frame H.264 BPE tokens (AVC-LM, vocab 8192)
<fps_30> <pelvis> <pelvis_t_0> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128>
         <pelvis_t_7> <pelvis_x_128> ... </pelvis>
<r_hip> <r_hip_t_0> <r_hip_x_115> ... </r_hip>
... (17 joints total)
```

## Agent token format (Adaptive PCHIP)

Each 8-frame pose window uses **variable-length** self-describing tokens:

- **`<fps_30>`** — frame rate
- **`<joint> ... </joint>`** — 17 H36M joints, each with 2, 4, or 8 control points based on motion curvature
- **`<joint_t_N>`** — frame index (0-7) within the window
- **`<joint_x_N>`**, **`<joint_y_N>`**, **`<joint_z_N>`** — quantized coordinates (uint8)

Dequantization: `coord_metres = N / 255.0 * 4.0 - 2.0` (range [-2, 2] m, precision ~15.7 mm)

Reconstruction: parse control points per joint, apply PCHIP interpolation → (8, 17, 3) trajectory in metres.

### 17 joints (H36M order)

| Index | Joint | Index | Joint | Index | Joint |
|---|---|---|---|---|---|
| 0 | pelvis | 7 | spine | 14 | r_shoulder |
| 1 | r_hip | 8 | thorax | 15 | r_elbow |
| 2 | r_knee | 9 | nose | 16 | r_wrist |
| 3 | r_ankle | 10 | head_top | | |
| 4 | l_hip | 11 | l_shoulder | | |
| 5 | l_knee | 12 | l_elbow | | |
| 6 | l_ankle | 13 | l_wrist | | |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "EmpathicRobotics/vla-1.7b-pab-spline-adaptive",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("EmpathicRobotics/tokenizer-vla-adaptive")

prompt = (
    "### Context: Person raises both arms above head.\\n"
    "<seed2_3758> <seed2_2157> <cosmos_58567> "
    "<fps_30> <pelvis> <pelvis_t_0> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128>"
)
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
output = model.generate(input_ids, max_new_tokens=500, do_sample=False)
print(tokenizer.decode(output[0]))
```

### Decoding agent tokens to 3D poses

```python
# pip install scipy
from decode_agent_tokens import decode  # from the 3d-human-pose repo

generated_text = tokenizer.decode(output[0])
trajectories = decode(generated_text)  # list of (8, 17, 3) ndarrays
```

## Training details

### Loss curve

| Iter | Loss | LR | Epoch |
|---|---|---|---|
| 50 | 6.158 | 1.0e-3 | 0.02 |
| 100 | 3.927 | 2.0e-3 | 0.05 |
| 200 | 2.982 | 4.0e-3 | 0.10 |
| 500 | 2.070 | 4.0e-3 | 0.25 |
| 1000 | 1.672 | 4.0e-3 | 0.49 |
| 1500 | 1.555 | 4.0e-3 | 0.74 |
| 2000 | 1.476 | 3.2e-4 | 0.99 |
| 2032 (val) | 1.501 | — | — |
| 2032 (test) | 1.494 | — | — |

### Config

- **Optimizer**: AdamW (β1=0.9, β2=0.95, wd=0.05, ε=1e-8, clip=1.0)
- **Schedule**: WSD (200 warmup, 400 linear decay at end, peak LR 4e-3)
- **Batch**: GBS 1024, MBS 4, seq_len 4096
- **Infrastructure**: 64 nodes × 4 GH200 GPUs (256 total), ~287 TFLOP/s/GPU
- **Wall time**: ~35 minutes
- **Framework**: Megatron-LM via oellm-autoexp

### Data pipeline

FineVideo (~40K YouTube videos) → Seed2/Cosmos/AVC-LM tokenization → HRNet 2D pose → MotionBERT 3D lift → kinematics → YOLO cleaning → adaptive PCHIP tokenization → merge → flatten → Megatron tokenization

See [EmpathicRobotics/FineVideo-Phase7-Flattened](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase7-Flattened) for the training data.

## Differences from first model

The previous model ([EmpathicRobotics/vla-1.7b-pab-spline-25b-test](https://huggingface.co/EmpathicRobotics/vla-1.7b-pab-spline-25b-test)) had a **broken tokenizer** — VLA tokens like `<seed2_1137>` were split into 7 sub-pieces by BPE. This model fixes that:

| | Previous (25b-test) | This model (adaptive) |
|---|---|---|
| Tokenizer | Broken (BPE splits VLA tokens) | Fixed (`add_tokens(special_tokens=True)`) |
| Agent format | Fixed 256 tokens per window | Adaptive 171-579 tokens (PCHIP, variable CPs) |
| Agent encoding | Scale + anchor + motion integers | Self-describing `<joint_t_N> <joint_x_N>` |
| Token atomicity | ❌ `<seed2_1137>` → 7 sub-pieces | ✅ `<seed2_1137>` → 1 token |

## Limitations

- **Small dataset** (2.84B tokens, ~3 epochs) — model memorizes well but generalises poorly to novel prompts
- **No vision encoder** — generates tokens from text descriptions only, not from actual video frames
- **Validation run** — proves the pipeline works end-to-end, not intended as a final model
- **Next steps**: Rich augmentation pipeline (4x data multiplier), additional datasets (SenseNova-SI-8M, stera-10m), Qwen3 architecture migration

## Citation

```bibtex
@misc{empathicrobotics2025vla,
  title={PAB-Spline VLA: Adaptive PCHIP Tokenization for Vision-Language-Action Models},
  author={EmpathicRobotics},
  year={2025},
  url={https://huggingface.co/EmpathicRobotics/vla-1.7b-pab-spline-adaptive}
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
        )

    print(f"\nDone: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
