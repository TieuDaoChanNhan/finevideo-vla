---
license: apache-2.0
task_categories:
  - video-classification
  - text-generation
tags:
  - finevideo
  - vla
  - multimodal
  - tokenization
  - seed2
  - cosmos
  - avc-lm
language:
  - en
size_categories:
  - 100K<n<1M
---

# FineVideo-Prototype-Tokenized — Base Video Token Dataset

## Overview

This dataset contains the **base video tokenization** output from the prototype pipeline, extracted from ~40K YouTube videos in the [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo) dataset.

Each video is tokenised into three modalities:

- **Seed2** — 1 FPS semantic keyframe tokens (vocab: 8,192)
- **Cosmos** — every 8 frames spatial video tokens (vocab: 64,000)
- **AVC-LM** — every 8 frames H.264 BPE tokens (vocab: 8,192)

This dataset does **not** contain 3D human pose (agent) tokens. Those are added in later phases of the pipeline. Use [FineVideo-Phase5-AgentTokens](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase5-AgentTokens) for the merged multimodal dataset, or [FineVideo-Phase7-Flattened](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase7-Flattened) for the final training-ready version.

## Statistics

| Metric | Value |
|--------|-------|
| Source videos | ~40,000 from [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo) |
| Total shards | 160 (`training_ready_rank_*.jsonl`) |
| Total size | ~660 GB |
| Compute | 40 SLURM nodes x 4 GPUs = 160 GPUs |
| Frame rate | 30 fps |

## Pipeline Context

This is the output of the **prototype pipeline** (Step A), which runs independently from the 3D pose pipeline. The two branches are merged in Phase 6.

| Phase | Description | Status |
|-------|-------------|--------|
| **Prototype** | **Seed2 + Cosmos + AVC-LM tokenisation (this dataset)** | **Done** |
| Phase 1 | HRNet 2D pose detection | Done |
| Phase 2 | MotionBERT 2D-to-3D lifting | Done |
| Phase 2.5 | Resample to 30fps | Done |
| Phase 3 | Kinematics: bone normalisation, root centering, smoothing | Done |
| Phase 4 | YOLO person-detection cleaning | Done |
| Phase 5 | Adaptive PCHIP per-joint tokenisation | Done |
| Phase 6 | Merge agent tokens into this dataset | Done |
| Phase 7 | Flatten to Megatron-LM format | Done |
| Phase 8 | Megatron-LM tokenization (.bin/.idx) | Done |

## Data Format

Each record is a JSON line representing one video with hierarchical structure:

```json
{
  "video_id": "abc123XYZ",
  "scenes": [
    {
      "activities": [
        {
          "text_prompt": "A person is cooking in a kitchen",
          "speech_transcript": "First, we add the oil to the pan...",
          "video_tokens": "<seed2> 3758 2157 ... </seed2> <cosmos> 18232 45001 ... </cosmos> <avc_lm> 263 107 ... </avc_lm> ..."
        }
      ]
    }
  ]
}
```

### Token modalities

| Modality | Rate | Vocab size | Description |
|----------|------|------------|-------------|
| Seed2 | 1 fps | 8,192 | Semantic keyframe tokens |
| Cosmos | every 8 frames | 64,000 | Spatial video tokens |
| AVC-LM | every 8 frames | 8,192 | H.264 BPE motion tokens |

All three modalities share a 30fps frame grid. Token values are raw integers within `<tag>...</tag>` wrapper pairs — they are flattened into `<tag_N>` format during Phase 7.

### Metadata per activity

- `text_prompt` — activity description from FineVideo annotations
- `speech_transcript` — speech-to-text transcript (when available)
- Scene-level fields: title, thematic keywords, mood

## Related Resources

| Resource | Description |
|----------|-------------|
| [EmpathicRobotics/FineVideo-Phase5-AgentTokens](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase5-AgentTokens) | This dataset + 3D pose agent tokens merged in (hierarchical, full metadata) |
| [EmpathicRobotics/FineVideo-Phase7-Flattened](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase7-Flattened) | Final flat Megatron-LM JSONL (ready for pretraining) |
| [EmpathicRobotics/tokenizer-vla-adaptive](https://huggingface.co/EmpathicRobotics/tokenizer-vla-adaptive) | HuggingFace tokenizer (144,215 vocab, all VLA tokens atomic) |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("EmpathicRobotics/FineVideo-Prototype-Tokenized", streaming=True)

for sample in ds["train"]:
    video_id = sample["video_id"]
    for scene in sample["scenes"]:
        for activity in scene["activities"]:
            tokens = activity["video_tokens"]
            has_seed2 = "<seed2>" in tokens
            has_cosmos = "<cosmos>" in tokens
            has_avc = "<avc_lm>" in tokens
            print(f"Video {video_id}: seed2={has_seed2}, cosmos={has_cosmos}, avc_lm={has_avc}")
            break
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
