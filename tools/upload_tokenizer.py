import os
import tempfile
import shutil
from huggingface_hub import HfApi

REPO_ID = "EmpathicRobotics/tokenizer-vla-adaptive"
TOKENIZER_DIR = "/p/data1/mmlaion/shared/vla/tokenizer_vla_adaptive"

README = """\
# Tokenizer VLA Adaptive

Extended GPT-NeoX-20b tokenizer for the FineVideo-VLA dataset.

## What is this?

This tokenizer extends the [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b) tokenizer with **93,938 new tokens** for multimodal Vision-Language-Action (VLA) pretraining.

| Category | Token format | Count |
|---|---|---|
| Seed2 visual tokens | `<seed2_N>` (N=0-8191) | 8,192 |
| Cosmos spatial tokens | `<cosmos_N>` (N=0-63999) | 64,000 |
| AVC-LM H.264 BPE tokens | `<avclm_N>` (N=0-8191) | 8,192 |
| Agent legacy tokens | `<agent_N>` (N=0-255) | 256 |
| FPS prefix | `<fps_N>` (N=1-60) | 60 |
| Joint position tokens | `<{joint}_x_N>`, `_y_N`, `_z_N` (N=0-255) | 13,056 |
| Joint time tokens | `<{joint}_t_N>` (N=0-7) | 136 |
| Wrapper tags | `<seed2>`, `</seed2>`, `<agent>`, `</agent>`, etc. | 46 |

**Total vocab size: 144,215** (50,277 base + 93,938 new)

## 17 Named Joints

`pelvis`, `r_hip`, `r_knee`, `r_ankle`, `l_hip`, `l_knee`, `l_ankle`, `spine`, `thorax`, `nose`, `head_top`, `l_shoulder`, `l_elbow`, `l_wrist`, `r_shoulder`, `r_elbow`, `r_wrist`

## Usage

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("EmpathicRobotics/tokenizer-vla-adaptive")

# All VLA tokens are atomic — never split by BPE
tok.encode("<seed2_1137>")    # -> [59908]
tok.encode("<pelvis_x_128>")  # -> [131151]
```

## How it was created

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tok.add_tokens(new_vla_tokens, special_tokens=True)
tok.save_pretrained("tokenizer-vla-adaptive")
```

All tokens are registered via `add_tokens(special_tokens=True)` so the BPE merge rules treat each one as a single atomic unit.
"""

api = HfApi()
api.create_repo(REPO_ID, repo_type="model", exist_ok=True)

with tempfile.TemporaryDirectory() as tmp:
    for f in os.listdir(TOKENIZER_DIR):
        shutil.copy2(os.path.join(TOKENIZER_DIR, f), tmp)
    with open(os.path.join(tmp, "README.md"), "w") as f:
        f.write(README)
    api.upload_folder(
        folder_path=tmp,
        repo_id=REPO_ID,
        repo_type="model",
        create_pr=False,
    )

print(f"Done: https://huggingface.co/{REPO_ID}")
