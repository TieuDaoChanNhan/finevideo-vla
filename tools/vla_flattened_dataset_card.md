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
  - 100K<n<1M
---

# FineVideo-VLA-flattened — Megatron-LM Multimodal Pretraining Dataset

## Overview

This is the **final, training-ready** flattened dataset from the FineVideo-VLA pipeline. Each record is a single `{"text": "..."}` JSON line containing interleaved multimodal tokens — ready for Megatron-LM tokenization and LLM pretraining.

Four token modalities are interleaved per record:

- **Seed2** — 1 FPS semantic keyframe tokens (vocab: 8192)
- **Cosmos** — every 8 frames spatial video tokens (vocab: 64000)
- **AVC-LM** — every 8 frames H.264 BPE tokens (vocab: 8192)
- **Agent** — adaptive PCHIP 3D human pose tokens with named joints (17 joints, variable control points)

Source: ~40,000 YouTube videos from [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo).

## Statistics

| Metric | Value |
|--------|-------|
| Total shards | 160 |
| Total records | ~372,385 activities |
| Total size | ~2.1 TB (uncompressed) |
| Train shards | 152 |
| Test shards | 8 |
| Split ratio | 95/5 (seed 42) |
| Avg file size | 13.2 GB |
| Malformed records | 0 |

### Modality coverage

| Modality | Coverage | Description |
|----------|----------|-------------|
| seed2 | 100% | Semantic keyframes |
| cosmos | 100% | Spatial video tokens |
| avc_lm | 100% | H.264 BPE tokens |
| agent (3D pose) | ~16–20% | Only for videos with visible humans |
| speech transcript | ~97% | `[Speech: ...]` in prompt |

## Data Format

Each line is a JSON object with a single `text` field:

```json
{
  "text": "USER: A person is cooking in a kitchen. [Speech: First, we add the oil...] ASSISTANT: <seed2> <seed2_3758> <seed2_2157> ... </seed2> <cosmos> <cosmos_58567> <cosmos_56071> ... </cosmos> <avc_lm> <avclm_263> <avclm_107> ... </avc_lm> <agent> <fps_30> <pelvis> <pelvis_t_0> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128> <pelvis_t_7> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128> </pelvis> <r_hip> ... </r_hip> ... </agent> <seed2> ... </seed2> <cosmos> ... </cosmos> ..."
}
```

### Structure within `text`

```
USER: <activity_description> [Speech: <transcript>] ASSISTANT: <tokens...>
```

Tokens are grouped by 8-frame chunk, repeating in order:

```
<seed2> <seed2_N> ... </seed2>          (every 30 frames)
<cosmos> <cosmos_N> ... </cosmos>        (every 8 frames)
<avc_lm> <avclm_N> ... </avc_lm>        (every 8 frames)
<agent> <fps_30> <joint>...</joint> ... </agent>  (every 8 frames, when pose data exists)
```

### Token details

**Seed2/Cosmos/AVC-LM**: Flattened from raw numbers into individual vocabulary tokens.
- `<seed2> 3758 2157 </seed2>` → `<seed2> <seed2_3758> <seed2_2157> </seed2>`
- `<avc_lm> 263 107 </avc_lm>` → `<avc_lm> <avclm_263> <avclm_107> </avc_lm>`

**Agent (3D pose)**: Self-describing named tokens, passed through unchanged:

```
<agent>
  <fps_30>
  <pelvis> <pelvis_t_0> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128>
           <pelvis_t_7> <pelvis_x_130> <pelvis_y_128> <pelvis_z_130> </pelvis>
  <r_hip>  <r_hip_t_0>  <r_hip_x_115> <r_hip_y_130> <r_hip_z_126>
           <r_hip_t_7>  <r_hip_x_116> <r_hip_y_125> <r_hip_z_124> </r_hip>
  ...17 joints...
</agent>
```

- **t tokens**: frame index 0–7 within the 8-frame window (control point time)
- **xyz tokens**: quantized uint8 [0, 255], mapping [-2.0m, +2.0m]
- **Dequantize**: `position_metres = token_value / 255.0 * 4.0 - 2.0`
- **CP tiers**: 2 CPs (static joints) / 4 CPs (moderate motion) / 8 CPs (fast motion)
- **Token count per chunk**: 171–579, typical ~250–300
- **Reconstruct 8 frames**: parse t/x/y/z per joint → PCHIP interpolation

### Joint names (H36M 17-joint skeleton)

| Joint | Joint | Joint |
|-------|-------|-------|
| pelvis | r_hip | r_knee |
| r_ankle | l_hip | l_knee |
| l_ankle | spine | thorax |
| nose | head_top | l_shoulder |
| l_elbow | l_wrist | r_shoulder |
| r_elbow | r_wrist | |

## Time Alignment

All modalities share the same **30fps frame grid**:

| Token type | Rate | Timestamp formula |
|------------|------|-------------------|
| Seed2 | every 30 frames | `activity_start + k × 1.0s` |
| Cosmos | every 8 frames | `activity_start + k × 8/30s` |
| AVC-LM | every 8 frames | `activity_start + k × 8/30s` |
| Agent | every 8 frames | `activity_start + k × 8/30s` |

Agent tokens may be absent for some chunks (no person detected by YOLO in that window).

## Vocabulary

This dataset uses an extended GPT-NeoX-20b vocabulary. The expanded vocab file is available at [the pipeline repository](https://github.com/TieuDaoChanNhan/3D-Human-Pose-VLA).

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

## Related Datasets

| Dataset | Description |
|---------|-------------|
| [EmpathicRobotics/FineVideo-VLA-Agent](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-VLA-Agent) | Pre-flattening hierarchical dataset with full metadata (timestamps, scenes, activities) |
| [EmpathicRobotics/FineVideo-Phase4-Pose](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase4-Pose) | Raw 3D pose data (float arrays, not tokenised) |

## Pipeline

| Phase | Description | Status |
|-------|-------------|--------|
| Step A | Seed2 + Cosmos + AVC-LM tokenisation (40 nodes × 4 GPU) | Done |
| Phase 1 | HRNet 2D pose detection | Done |
| Phase 2 | MotionBERT 2D→3D lifting | Done |
| Phase 2.5 | Resample to 30fps | Done |
| Phase 3 | Kinematics: bone normalisation, root centering, smoothing | Done |
| Phase 4 | YOLO person-detection cleaning | Done |
| Phase 5 | Adaptive PCHIP per-joint tokenisation | Done |
| Phase 6 | Merge agent tokens into multimodal dataset | Done |
| **Phase 7** | **Flatten to Megatron-LM format (this dataset)** | **Done** |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("EmpathicRobotics/FineVideo-VLA-flattened", streaming=True)

for sample in ds["train"]:
    text = sample["text"]
    # text starts with "USER: ..." and contains all interleaved tokens
    has_pose = "<agent>" in text
    print(f"Length: {len(text.split())} tokens, has pose: {has_pose}")
    break
```

## Citation

Part of the FineVideo-VLA project. If you use this data, please cite:

```bibtex
@misc{finevideo2024,
  title={FineVideo},
  author={HuggingFace},
  year={2024},
  url={https://huggingface.co/datasets/HuggingFaceFV/finevideo}
}
```

## License

Apache 2.0
