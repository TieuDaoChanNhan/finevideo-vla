---
license: apache-2.0
task_categories:
  - video-classification
  - robotics
tags:
  - 3d-pose
  - human-pose-estimation
  - motion
  - finevideo
  - vla
  - multimodal
  - tokenization
  - adaptive-pchip
language:
  - en
size_categories:
  - 100K<n<1M
---

# FineVideo-Phase5-AgentTokens — Multimodal Video+Pose Dataset (Hierarchical)

## Overview

This dataset is the **full-structure merged multimodal dataset** from the FineVideo-VLA pipeline. Each record represents a YouTube video with all metadata preserved: scenes, activities, speech transcripts, timestamps, and interleaved token sequences covering four modalities:

- **Seed2** — 1 FPS semantic keyframe tokens (vocab: 8192)
- **Cosmos** — every 8 frames spatial tokens (vocab: 64000)
- **AVC-LM** — every 8 frames H.264 BPE tokens (vocab: 8192)
- **Agent** — adaptive PCHIP 3D human pose tokens with named joints (17 joints, variable CPs)

Use this dataset when you need the full hierarchical structure, timestamps, or metadata. For flat Megatron-LM training, use [FineVideo-Phase7-Flattened](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase7-Flattened) instead.

## Statistics

| Metric | Value |
|--------|-------|
| Source videos | ~40,000 from [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo) |
| Total activities | ~399,000 |
| Videos with 3D pose | 18,847 |
| Agent blocks injected | ~2.15M |
| Total shards | 160 |
| Total size | ~657 GB (uncompressed) |
| Avg shard size | ~4.1 GB |
| Train shards | 152 |
| Test shards | 8 |
| Split ratio | 95/5 (seed 42) |
| Pose frame rate | 30 fps |
| Joints per frame | 17 (H36M skeleton) |

## Data Format

Each line is a JSON record representing one video:

```json
{
  "video_id": "abc123XYZ",
  "scenes": [
    {
      "activities": [
        {
          "text_prompt": "A person is cooking in a kitchen",
          "speech_transcript": "First, we add the oil to the pan...",
          "video_tokens": "<seed2> ... </seed2> <cosmos> ... </cosmos> <avc_lm> ... </avc_lm> <agent> <fps_30> <pelvis> <pelvis_t_0> <pelvis_x_128> ... </pelvis> ... </agent> ...",
          "chunk_timing": [...],
          "timing_meta": {...},
          "agent_token_order": "image_first",
          "agent_fps": 30
        }
      ]
    }
  ]
}
```

### Token order per 8-frame chunk

```
<seed2> ... </seed2>              (every 30 frames — not every chunk)
<cosmos> ... </cosmos>            (every 8 frames)
<avc_lm> ... </avc_lm>           (every 8 frames)
<agent> <fps_30> ... </agent>    (every 8 frames, when pose data exists)
```

### Agent token format (Adaptive PCHIP)

Each joint gets 2, 4, or 8 control points based on trajectory curvature:

```
<fps_30>
<pelvis> <pelvis_t_0> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128>
         <pelvis_t_7> <pelvis_x_130> <pelvis_y_128> <pelvis_z_130> </pelvis>
<r_hip>  <r_hip_t_0>  <r_hip_x_140> <r_hip_y_130> <r_hip_z_126>
         <r_hip_t_3>  <r_hip_x_139> <r_hip_y_128> <r_hip_z_126>
         <r_hip_t_7>  <r_hip_x_141> <r_hip_y_128> <r_hip_z_124> </r_hip>
...17 joints total...
```

- **t tokens**: frame index 0–7 within the 8-frame window
- **xyz tokens**: quantized uint8 [0, 255], mapping [-2.0m, +2.0m]
- **Dequantize**: `position_metres = token_value / 255.0 * 4.0 - 2.0`
- **CP tiers**: low curvature = 2 CPs, medium = 4 CPs, high = 8 CPs
- **Token count per chunk**: 171 (all 2-CP) to 579 (all 8-CP), typical ~250–300
- **Reconstruct all 8 frames**: parse CPs per joint, apply PCHIP interpolation

### Joint names (H36M 17-joint skeleton)

| Index | Joint | Index | Joint | Index | Joint |
|-------|-------|-------|-------|-------|-------|
| 0 | pelvis | 6 | l_ankle | 12 | l_elbow |
| 1 | r_hip | 7 | spine | 13 | l_wrist |
| 2 | r_knee | 8 | thorax | 14 | r_shoulder |
| 3 | r_ankle | 9 | nose | 15 | r_elbow |
| 4 | l_hip | 10 | head_top | 16 | r_wrist |
| 5 | l_knee | 11 | l_shoulder | | |

### chunk_timing

Each activity includes a `chunk_timing` array mapping every 8-frame chunk to its temporal position:

```json
{
  "chunk_idx": 0,
  "abs_frame": 30,
  "start_sec": 1.0,
  "end_sec": 1.267,
  "has_seed2": true,
  "has_cosmos": true,
  "has_avc_lm": true,
  "has_agent": true
}
```

Use this to associate any token group with an absolute timestamp in the video.

### timing_meta

```json
{
  "video_fps": 30,
  "chunk_frames": 8,
  "seed2_rate": "1fps_keyframe",
  "cosmos_rate": "every_8_frames",
  "avc_lm_rate": "every_8_frames",
  "agent_rate": "every_8_frames_adaptive_pchip"
}
```

## Pipeline

| Phase | Description | Status |
|-------|-------------|--------|
| Step A | Seed2 + Cosmos + AVC-LM tokenisation (40 nodes x 4 GPU) | Done |
| Phase 1 | HRNet 2D pose detection | Done |
| Phase 2 | MotionBERT 2D to 3D lifting | Done |
| Phase 2.5 | Resample to 30fps | Done |
| Phase 3 | Kinematics: bone normalisation, root centering, smoothing | Done |
| Phase 4 | YOLO person-detection cleaning | Done |
| Phase 5 | Adaptive PCHIP per-joint tokenisation | Done |
| **Phase 6** | **Merge agent tokens into multimodal dataset (this dataset)** | **Done** |
| Phase 7 | Flatten to Megatron-LM format | Done |
| Phase 8 | Megatron-LM tokenization (.bin/.idx) | Done |

## Related Resources

| Resource | Description |
|----------|-------------|
| [EmpathicRobotics/tokenizer-vla-adaptive](https://huggingface.co/EmpathicRobotics/tokenizer-vla-adaptive) | HuggingFace tokenizer (144,215 vocab, all VLA tokens atomic) |
| [EmpathicRobotics/FineVideo-Phase7-Flattened](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase7-Flattened) | Flat Megatron-LM JSONL (ready for pretraining, no structure/metadata) |
| [EmpathicRobotics/FineVideo-Phase4-YOLOPose](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase4-YOLOPose) | Raw 3D pose data (float arrays, not tokenised) |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("EmpathicRobotics/FineVideo-Phase5-AgentTokens", streaming=True)

for sample in ds["train"]:
    video_id = sample["video_id"]
    for scene in sample["scenes"]:
        for activity in scene["activities"]:
            tokens = activity["video_tokens"]
            timing = activity.get("chunk_timing", [])
            speech = activity.get("speech_transcript", "")

            if "<agent>" in tokens:
                print(f"Video {video_id} has 3D pose agent tokens")

            # Get timestamp for each chunk
            for chunk in timing:
                print(f"  Chunk {chunk['chunk_idx']}: {chunk['start_sec']:.3f}s – {chunk['end_sec']:.3f}s, agent={chunk['has_agent']}")
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
