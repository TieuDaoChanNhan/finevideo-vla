---
license: apache-2.0
task_categories:
  - video-classification
  - robotics
  - text-generation
tags:
  - 3d-pose
  - human-pose-estimation
  - motion
  - finevideo
  - vla
  - multimodal
  - tokenization
  - adaptive-pchip
  - megatron-lm
  - pretraining
language:
  - en
size_categories:
  - 10K<n<100K
---

# FineVideo-Phase7-Flattened — Megatron-LM Multimodal Pretraining Dataset

## Overview

This is the **final, training-ready** flattened dataset from the FineVideo-VLA pipeline. Each record is a single `{"text": "..."}` JSON line containing interleaved multimodal tokens — ready for Megatron-LM tokenization and LLM pretraining.

Three token modalities are interleaved per record (v2: AVC-LM removed pending ablation):

- **Seed2** — 1 FPS semantic keyframe tokens (vocab: 8192)
- **Cosmos** — every 8 frames spatial video tokens (vocab: 64000), kept at 50%
- **Agent** — adaptive PCHIP 3D human pose tokens with named joints (17 joints, variable control points)

Source: ~40,000 YouTube videos from [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo).

**Only activities containing 3D pose (`<agent>`) tokens are included.** This ensures every record has action data for Vision-Language-Action pretraining.

## Modality Dropout (Token Balancing) — v2

In the raw data, image tokens massively outnumber action tokens. **Modality dropout** is applied during flattening to balance the modalities.

**v2 dropout** (this dataset, Jun 2026):

| Modality | Drop rate | Reason |
|----------|-----------|--------|
| AVC-LM | **100%** | Removed until ablation studies confirm benefit |
| Cosmos | **50%** | Keep ~50% of chunks for seed2→cosmos→agent transition learning |
| Seed2 | 0% | Keep all — primary visual signal |
| Agent | 0% | Keep all |

This ensures most records contain the full `seed2 → cosmos → agent` transition chain, teaching the model to sequence modalities autonomously.

## Data Augmentation

Each record has text augmentation applied:

| Augmentation | Rate | Description |
|-------------|------|-------------|
| Synonym replacement | 15% | Content words (>5 chars) randomly replaced with WordNet synonyms |
| Stopword dropout | 5% | Common stopwords randomly removed |
| Sentence permutation | 10% | Speech transcript sentences randomly reordered |
| Speech/token interleaving | — | Speech chunks inserted at random positions among tokens |
| Layout block shuffling | — | Title/Context/Keywords/Tokens blocks randomly reordered |

## Statistics

| Metric | Value |
|--------|-------|
| Total shards | 160 |
| Train shards | 152 |
| Test shards | 8 |
| Split ratio | 95/5 (seed 42) |
| Compression | gzip level 5 |

*Note: record counts and sizes depend on the flatten run. Check the repo files tab for current shard sizes.*

## Data Format

Each line is a JSON object with a single `text` field:

```json
{
  "text": "### Title: Launching\n### Context: A video showcasing diverse vocation paths...\n### Keywords: educational, informative\n<seed2_6750> <seed2_680> ... <cosmos_18232> <cosmos_41007> ... <fps_30> <pelvis> <pelvis_t_0> <pelvis_x_128> ... </pelvis> <r_hip> ... </r_hip> ..."
}
```

### Structure within `text`

Each record contains four layout blocks (randomly shuffled):

```
### Title: <scene title, augmented>
### Context: <global context + activity prompt, augmented>
### Keywords: <scene thematic + mood, augmented>
<interleaved speech chunks and flattened tokens>
```

### Token details

**Seed2/Cosmos/AVC-LM**: Flattened from raw numbers into individual vocabulary tokens (with modality dropout applied).

**Agent (3D pose)**: Self-describing named tokens, always kept:

```
<fps_30> <pelvis> <pelvis_t_0> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128>
<pelvis_t_7> <pelvis_x_130> <pelvis_y_128> <pelvis_z_130> </pelvis>
<r_hip> <r_hip_t_0> <r_hip_x_115> <r_hip_y_130> <r_hip_z_126>
<r_hip_t_7> <r_hip_x_116> <r_hip_y_125> <r_hip_z_124> </r_hip>
...17 joints...
```

- **t tokens**: frame index 0–7 within the 8-frame window (control point time)
- **xyz tokens**: quantized uint8 [0, 255], mapping [-2.0m, +2.0m]
- **Dequantize**: `position_metres = token_value / 255.0 * 4.0 - 2.0`
- **CP tiers**: 2 CPs (static joints) / 4 CPs (moderate motion) / 8 CPs (fast motion)
- **Reconstruct 8 frames**: parse t/x/y/z per joint, apply PCHIP interpolation

### Joint names (H36M 17-joint skeleton)

| Joint | Joint | Joint |
|-------|-------|-------|
| pelvis | r_hip | r_knee |
| r_ankle | l_hip | l_knee |
| l_ankle | spine | thorax |
| nose | head_top | l_shoulder |
| l_elbow | l_wrist | r_shoulder |
| r_elbow | r_wrist | |

## Vocabulary & Tokenizer

This dataset uses an extended GPT-NeoX-20b vocabulary with 93,938 additional VLA tokens (total: 144,215).

The HuggingFace tokenizer is available at [EmpathicRobotics/tokenizer-vla-adaptive](https://huggingface.co/EmpathicRobotics/tokenizer-vla-adaptive). All VLA tokens are registered as atomic tokens via `add_tokens(special_tokens=True)` — the BPE tokenizer will never split them into sub-pieces.

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("EmpathicRobotics/tokenizer-vla-adaptive")
tok.encode("<seed2_1137>")    # -> [59908]  (single token, not split)
tok.encode("<pelvis_x_128>")  # -> [131151] (single token, not split)
```

| Token range | Count |
|-------------|-------|
| Base GPT-NeoX-20b | 50,277 |
| `<seed2_N>` | 8,192 |
| `<cosmos_N>` | 64,000 |
| `<avclm_N>` | 8,192 |
| `<fps_N>` | 60 |
| Joint tokens (xyz, t, wrappers) | 13,226 |
| Modality wrappers | 8 |
| Legacy `<agent_N>` | 256 |

## Related Resources

| Resource | Description |
|----------|-------------|
| [EmpathicRobotics/tokenizer-vla-adaptive](https://huggingface.co/EmpathicRobotics/tokenizer-vla-adaptive) | HuggingFace tokenizer for this dataset (144,215 vocab) |
| [EmpathicRobotics/FineVideo-Phase5-AgentTokens](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase5-AgentTokens) | Pre-flattening hierarchical dataset with full metadata (timestamps, scenes, activities, no dropout) |
| [EmpathicRobotics/FineVideo-Phase4-YOLOPose](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase4-YOLOPose) | Raw 3D pose data (float arrays, not tokenised) |

## Pipeline

| Phase | Description | Status |
|-------|-------------|--------|
| Step A | Seed2 + Cosmos + AVC-LM tokenisation (40 nodes x 4 GPU) | Done |
| Phase 1 | HRNet 2D pose detection | Done |
| Phase 2 | MotionBERT 2D→3D lifting | Done |
| Phase 2.5 | Resample to 30fps | Done |
| Phase 3 | Kinematics: bone normalisation, root centering, smoothing | Done |
| Phase 4 | YOLO person-detection cleaning | Done |
| Phase 5 | Adaptive PCHIP per-joint tokenisation | Done |
| Phase 6 | Merge agent tokens into multimodal dataset | Done |
| **Phase 7** | **Flatten with modality dropout + augmentation (this dataset)** | **Done** |
| Phase 8 | Megatron-LM tokenization (.bin/.idx) | Done |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("EmpathicRobotics/FineVideo-Phase7-Flattened", streaming=True)

for sample in ds["train"]:
    text = sample["text"]
    print(f"Length: {len(text.split())} tokens")
    break
```

## Citation

Part of the FineVideo-VLA project. If you use this data, please cite:

```bibtex
@misc{Farré2024FineVideo,
  title={FineVideo},
  author={Farré, Miquel and Marafioti, Andi and Tunstall, Lewis and Von Werra, Leandro and Wolf, Thomas},
  year={2024},
  howpublished={\url{https://huggingface.co/datasets/HuggingFaceFV/finevideo}},
}
```

## License

Apache 2.0
