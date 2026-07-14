#!/usr/bin/env python3
"""
Upload the two new VLA tokenizers to HuggingFace EmpathicRobotics.

Two repos:
  tokenizer-vla-adaptive-v2  — GPT-NeoX-20b base + SNAC (156,505 vocab)
  tokenizer-vla-qwen3        — Qwen3 base + all VLA tokens (257,897 vocab)

Usage:
    python tools/upload/upload_tokenizers_v2.py --mode adaptive_v2
    python tools/upload/upload_tokenizers_v2.py --mode qwen3
    python tools/upload/upload_tokenizers_v2.py --mode all
"""

import argparse
import os
import shutil
import tempfile

from huggingface_hub import HfApi

# ── Paths ─────────────────────────────────────────────────────────────────────

ADAPTIVE_V2_DIR = "/p/data1/mmlaion/shared/vla/tokenizer_vla_adaptive_v2"
QWEN3_DIR       = "/p/data1/mmlaion/shared/vla/tokenizer_vla_qwen3"

REPO_ADAPTIVE_V2 = "EmpathicRobotics/tokenizer-vla-adaptive-v2"
REPO_QWEN3       = "EmpathicRobotics/tokenizer-vla-qwen3"

# ── Model cards ───────────────────────────────────────────────────────────────

README_ADAPTIVE_V2 = """\
---
language: en
tags:
  - vla
  - tokenizer
  - robotics
  - multimodal
  - pose-estimation
  - audio
license: apache-2.0
---

# VLA Tokenizer — Adaptive v2 (GPT-NeoX-20b + SNAC + caption/speech)

Extended GPT-NeoX-20b tokenizer for the **FineVideo-VLA** multimodal dataset.
Adds 3D human pose tokens, video tokens, SNAC audio tokens, and caption/speech
wrapper tokens on top of the
[EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b) base.

**Vocab size: 156,509** (50,277 base + 93,938 VLA + 12,290 SNAC + 4 caption/speech)

> **v1 → v2 change:** Added 12,290 SNAC audio tokens (`<snac>`, `</snac>`,
> and 12,288 `<snac_N>` tokens) for the SNAC listen format used in
> [MixtureVitae-Omni](https://huggingface.co/datasets/mixture-vitae/MixtureVitae-Omni)
> and FineVideo-VLA audio tokenization. All existing v1 token IDs are unchanged.
>
> **Later addition (same v2 repo, in place):** Added 4 wrapper tokens —
> `<caption>`, `</caption>`, `<speech>`, `</speech>` — for inline visual
> caption / spoken-dialogue interleaving at modality-transition points in the
> token sequence. All prior token IDs (including SNAC) are unchanged.

---

## Token categories

| Category | Format | Count | Notes |
|----------|--------|-------|-------|
| Seed2 visual | `<seed2_N>` (N: 0–8191) | 8,192 | Semantic keyframe tokens, 1 FPS |
| Cosmos spatial | `<cosmos_N>` (N: 0–63999) | 64,000 | Spatial video tokens, every 8 frames |
| AVC-LM H.264 | `<avclm_N>` (N: 0–8191) | 8,192 | H.264 BPE tokens, every 8 frames |
| Agent legacy | `<agent_N>` (N: 0–255) | 256 | Legacy opaque agent tokens |
| FPS prefix | `<fps_N>` (N: 1–60) | 60 | Frame rate marker per chunk |
| Joint position | `<{joint}_x/y/z_N>` (N: 0–255) | 13,056 | Quantized xyz, maps [-2m, +2m] |
| Joint time | `<{joint}_t_N>` (N: 0–7) | 136 | Frame index within 8-frame window |
| Modality wrappers | `<seed2>`, `</agent>`, etc. | 46 | Open/close tags + joint wrappers |
| **SNAC Level 0** | `<snac_128266>` – `<snac_132361>` | 4,096 | 12.5 Hz coarse audio |
| **SNAC Level 1 even** | `<snac_132362>` – `<snac_136457>` | 4,096 | 25 Hz fine audio (even frames) |
| **SNAC Level 1 odd** | `<snac_144650>` – `<snac_148745>` | 4,096 | 25 Hz fine audio (odd frames) |
| **SNAC wrappers** | `<snac>`, `</snac>` | 2 | Block delimiters |
| **Caption/speech wrappers** | `<caption>`, `</caption>`, `<speech>`, `</speech>` | 4 | Inline caption/dialogue interleaving |

**Total new tokens: 106,232** (93,938 VLA + 12,290 SNAC + 4 caption/speech)

---

## 17 Named Joints (H36M skeleton)

`pelvis` · `r_hip` · `r_knee` · `r_ankle` · `l_hip` · `l_knee` · `l_ankle` ·
`spine` · `thorax` · `nose` · `head_top` · `l_shoulder` · `l_elbow` · `l_wrist` ·
`r_shoulder` · `r_elbow` · `r_wrist`

---

## Token format in context

Each 8-frame chunk in the interleaved sequence looks like:

```
<cosmos_63127> <cosmos_42647> ... </cosmos>
<avc_lm> <avclm_263> <avclm_107> ... </avc_lm>
<agent>
  <fps_30>
  <pelvis> <pelvis_t_0> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128>
           <pelvis_t_7> <pelvis_x_129> <pelvis_y_128> <pelvis_z_128> </pelvis>
  <r_hip>  <r_hip_t_0>  <r_hip_x_115>  ...  </r_hip>
  ... 17 joints total ...
</agent>
<snac> <snac_131580> <snac_134777> <snac_147244>
       <snac_131267> <snac_135192> <snac_148152>
       <snac_128995> <snac_133704> <snac_145875> </snac>
```

SNAC listen format: 3 tokens per base frame (L0 + L1_even + L1_odd),
37.5 tokens/sec, ~9–10 tokens per 8-frame chunk at 30 FPS.

---

## Usage

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("EmpathicRobotics/tokenizer-vla-adaptive-v2")
print(len(tok))  # 156509

# All VLA and SNAC tokens are single atomic tokens
print(tok.encode("<seed2_1137>",   add_special_tokens=False))  # [59908]
print(tok.encode("<pelvis_x_128>", add_special_tokens=False))  # [131151]
print(tok.encode("<fps_30>",       add_special_tokens=False))  # [130992]
print(tok.encode("<snac_128266>",  add_special_tokens=False))  # single ID
print(tok.encode("<snac_132362>",  add_special_tokens=False))  # single ID
print(tok.encode("<snac_144650>",  add_special_tokens=False))  # single ID
```

---

## How it was created

```python
from transformers import AutoTokenizer

# Start from existing v1 tokenizer (144,215 vocab)
tok = AutoTokenizer.from_pretrained("EmpathicRobotics/tokenizer-vla-adaptive")

snac_tokens = ["<snac>", "</snac>"]
snac_tokens += [f"<snac_{i + 128266}>" for i in range(4096)]  # L0
snac_tokens += [f"<snac_{i + 132362}>" for i in range(4096)]  # L1 even
snac_tokens += [f"<snac_{i + 144650}>" for i in range(4096)]  # L1 odd

tok.add_tokens(snac_tokens, special_tokens=True)  # all atomic
tok.save_pretrained("tokenizer-vla-adaptive-v2")
# vocab size: 156,505
```

Script: `tools/tokenizer/build_tokenizers.py` in the
[finevideo-vla](https://github.com/TieuDaoChanNhan/finevideo-vla) repo.

---

## Related

| Resource | Link |
|----------|------|
| v1 tokenizer (no SNAC) | [tokenizer-vla-adaptive](https://huggingface.co/EmpathicRobotics/tokenizer-vla-adaptive) |
| Qwen3-based version | [tokenizer-vla-qwen3](https://huggingface.co/EmpathicRobotics/tokenizer-vla-qwen3) |
| VLA model trained with v1 | [vla-1.7b-pab-spline-adaptive](https://huggingface.co/EmpathicRobotics/vla-1.7b-pab-spline-adaptive) |
| FineVideo-VLA dataset | [FineVideo-Phase7-Flattened](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase7-Flattened) |
"""

README_QWEN3 = """\
---
language: en
tags:
  - vla
  - tokenizer
  - robotics
  - multimodal
  - pose-estimation
  - audio
  - qwen3
license: apache-2.0
---

# VLA Tokenizer — Qwen3

Extended Qwen3 tokenizer for the **FineVideo-VLA** multimodal dataset.
Adds all VLA tokens (video, 3D pose, SNAC audio, caption/speech wrappers) on
top of the Qwen3 base tokenizer.

**Vocab size: 257,901** (~151,669 Qwen3 base + 106,232 VLA tokens)

> **Why Qwen3?** The Qwen3 family has strong multilingual + reasoning abilities
> and native HuggingFace ecosystem support (vLLM, llama.cpp, transformers).
> This tokenizer enables training a Qwen3-based VLA model from scratch or
> continued pretraining with full VLA token support.

---

## Token categories added on top of Qwen3

| Category | Format | Count | Notes |
|----------|--------|-------|-------|
| Seed2 visual | `<seed2_N>` (N: 0–8191) | 8,192 | Semantic keyframe tokens, 1 FPS |
| Cosmos spatial | `<cosmos_N>` (N: 0–63999) | 64,000 | Spatial video tokens, every 8 frames |
| AVC-LM H.264 | `<avclm_N>` (N: 0–8191) | 8,192 | H.264 BPE tokens, every 8 frames |
| Agent legacy | `<agent_N>` (N: 0–255) | 256 | Legacy opaque agent tokens |
| FPS prefix | `<fps_N>` (N: 1–60) | 60 | Frame rate marker per chunk |
| Joint position | `<{joint}_x/y/z_N>` (N: 0–255) | 13,056 | Quantized xyz, maps [-2m, +2m] |
| Joint time | `<{joint}_t_N>` (N: 0–7) | 136 | Frame index within 8-frame window |
| Modality wrappers | `<seed2>`, `</agent>`, etc. | 46 | Open/close tags + joint wrappers |
| SNAC Level 0 | `<snac_128266>` – `<snac_132361>` | 4,096 | 12.5 Hz coarse audio |
| SNAC Level 1 even | `<snac_132362>` – `<snac_136457>` | 4,096 | 25 Hz fine audio (even frames) |
| SNAC Level 1 odd | `<snac_144650>` – `<snac_148745>` | 4,096 | 25 Hz fine audio (odd frames) |
| SNAC wrappers | `<snac>`, `</snac>` | 2 | Block delimiters |
| Caption/speech wrappers | `<caption>`, `</caption>`, `<speech>`, `</speech>` | 4 | Inline caption/dialogue interleaving |

**Total VLA tokens added: 106,232**

---

## 17 Named Joints (H36M skeleton)

`pelvis` · `r_hip` · `r_knee` · `r_ankle` · `l_hip` · `l_knee` · `l_ankle` ·
`spine` · `thorax` · `nose` · `head_top` · `l_shoulder` · `l_elbow` · `l_wrist` ·
`r_shoulder` · `r_elbow` · `r_wrist`

---

## Usage

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("EmpathicRobotics/tokenizer-vla-qwen3")
print(len(tok))  # 257901

# All VLA tokens are atomic
print(tok.encode("<seed2_1137>",   add_special_tokens=False))  # single ID
print(tok.encode("<pelvis_x_128>", add_special_tokens=False))  # single ID
print(tok.encode("<fps_30>",       add_special_tokens=False))  # single ID
print(tok.encode("<snac_128266>",  add_special_tokens=False))  # single ID
```

---

## How it was created

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")  # or any Qwen3 variant

# Build all 106,228 VLA tokens
vla_tokens = []

# Modality wrappers
vla_tokens += ["<seed2>", "</seed2>", "<cosmos>", "</cosmos>",
                "<avc_lm>", "</avc_lm>", "<agent>", "</agent>",
                "<start_cosmo>", "</start_cosmo>", "<start_avclm>", "</start_avclm>"]
# Joint wrappers
for joint in ["pelvis", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle",
              "spine", "thorax", "nose", "head_top", "l_shoulder", "l_elbow", "l_wrist",
              "r_shoulder", "r_elbow", "r_wrist"]:
    vla_tokens += [f"<{joint}>", f"</{joint}>"]
# Video tokens
vla_tokens += [f"<agent_{i}>" for i in range(256)]
vla_tokens += [f"<avclm_{i}>" for i in range(8192)]
vla_tokens += [f"<seed2_{i}>" for i in range(8192)]
vla_tokens += [f"<cosmos_{i}>" for i in range(64000)]
# Pose tokens
vla_tokens += [f"<fps_{i}>" for i in range(1, 61)]
for joint in [...]:  # 17 joints
    vla_tokens += [f"<{joint}_x_{n}>" for n in range(256)]
    vla_tokens += [f"<{joint}_y_{n}>" for n in range(256)]
    vla_tokens += [f"<{joint}_z_{n}>" for n in range(256)]
    vla_tokens += [f"<{joint}_t_{n}>" for n in range(8)]
# SNAC tokens
vla_tokens += ["<snac>", "</snac>"]
vla_tokens += [f"<snac_{i + 128266}>" for i in range(4096)]
vla_tokens += [f"<snac_{i + 132362}>" for i in range(4096)]
vla_tokens += [f"<snac_{i + 144650}>" for i in range(4096)]

tok.add_tokens(vla_tokens, special_tokens=True)  # all atomic
tok.save_pretrained("tokenizer-vla-qwen3")
# vocab size: 257,897
```

Full script: `tools/tokenizer/build_tokenizers.py` in the
[finevideo-vla](https://github.com/TieuDaoChanNhan/finevideo-vla) repo.

---

## Interleaved token sequence format

```
USER: <activity description> [Speech: ...]  ASSISTANT:
<seed2_6750> <seed2_680> ...                    # semantic keyframes 1 FPS
<cosmos_63127> <cosmos_42647> ... </cosmos>     # spatial video every 8 frames
<avc_lm> <avclm_263> <avclm_107> ... </avc_lm>
<agent>
  <fps_30>
  <pelvis> <pelvis_t_0> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128> ... </pelvis>
  ... 17 joints ...
</agent>
<snac> <snac_131580> <snac_134777> <snac_147244> ... </snac>
```

---

## Related

| Resource | Link |
|----------|------|
| GPT-NeoX v2 tokenizer (+ SNAC) | [tokenizer-vla-adaptive-v2](https://huggingface.co/EmpathicRobotics/tokenizer-vla-adaptive-v2) |
| Original GPT-NeoX v1 tokenizer | [tokenizer-vla-adaptive](https://huggingface.co/EmpathicRobotics/tokenizer-vla-adaptive) |
| VLA model (trained with v1) | [vla-1.7b-pab-spline-adaptive](https://huggingface.co/EmpathicRobotics/vla-1.7b-pab-spline-adaptive) |
| FineVideo-VLA dataset | [FineVideo-Phase7-Flattened](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase7-Flattened) |
"""


# ── Upload functions ──────────────────────────────────────────────────────────

def upload(tokenizer_dir: str, repo_id: str, readme: str):
    api = HfApi()
    print(f"\n=== Uploading {repo_id} ===")
    print(f"  Source: {tokenizer_dir}")

    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    print(f"  Repo ready: https://huggingface.co/{repo_id}")

    with tempfile.TemporaryDirectory() as tmp:
        for fname in os.listdir(tokenizer_dir):
            shutil.copy2(os.path.join(tokenizer_dir, fname), tmp)
        with open(os.path.join(tmp, "README.md"), "w") as f:
            f.write(readme)

        files = os.listdir(tmp)
        print(f"  Files to upload: {sorted(files)}")

        api.upload_folder(
            folder_path=tmp,
            repo_id=repo_id,
            repo_type="model",
            create_pr=False,
            commit_message="Upload VLA tokenizer with model card",
        )

    print(f"  Done: https://huggingface.co/{repo_id}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=["adaptive_v2", "qwen3", "all"],
        default="all",
        help="Which tokenizer(s) to upload.",
    )
    args = p.parse_args()

    if args.mode in ("adaptive_v2", "all"):
        upload(ADAPTIVE_V2_DIR, REPO_ADAPTIVE_V2, README_ADAPTIVE_V2)

    if args.mode in ("qwen3", "all"):
        upload(QWEN3_DIR, REPO_QWEN3, README_QWEN3)

    print("\nAll done.")


if __name__ == "__main__":
    main()
