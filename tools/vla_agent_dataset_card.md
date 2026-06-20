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

# FineVideo-VLA-Agent — Multimodal Video+Pose Pretraining Dataset

## Overview

This dataset is the **final merged multimodal dataset** from the FineVideo-VLA pipeline. Each record represents a YouTube video with interleaved token sequences covering four modalities:

- **Seed2** — 1 FPS semantic keyframe tokens (vocab: 8192)
- **Cosmos** — every 8 frames spatial tokens (vocab: 64000)
- **AVC-LM** — every 8 frames H.264 BPE tokens (vocab: 8192)
- **Agent** — adaptive PCHIP 3D human pose tokens with named joints (17 joints, variable CPs)

## Statistics

- **~40,000 videos** from [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo)
- **~399K activities** across all videos
- **2.15M agent blocks** injected (5.5% of AVC blocks have matching pose data)
- **18,847 videos** with 3D pose agent tokens
- **160 JSONL files**, ~4.1 GB each, **~657 GB** total
- **30 fps** pose data, 17 joints (H36M skeleton)

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
          "speech_transcript": "...",
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
<cosmos> ... </cosmos> <avc_lm> ... </avc_lm> <agent> <fps_30> <joint> ... </joint> ... </agent>
```

### Agent token format (Adaptive PCHIP)

Each joint gets 2, 4, or 8 control points based on curvature:

```
<fps_30>
<pelvis> <pelvis_t_0> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128>
         <pelvis_t_7> <pelvis_x_130> <pelvis_y_128> <pelvis_z_130> </pelvis>
<r_hip>  <r_hip_t_0>  <r_hip_x_140> <r_hip_y_130> <r_hip_z_126>
         <r_hip_t_7>  <r_hip_x_141> <r_hip_y_128> <r_hip_z_124> </r_hip>
...17 joints total...
```

- **t tokens**: frame index 0-7 within the 8-frame window
- **xyz tokens**: quantized uint8 [0,255], mapping [-2.0m, +2.0m]
- **CP tiers**: low curvature = 2 CPs, medium = 4 CPs, high = 8 CPs
- **Token count per chunk**: 171 (all 2-CP) to 579 (all 8-CP), typical ~250-300

### Joint names (H36M 17-joint skeleton)

| Joint | Joint | Joint |
|-------|-------|-------|
| pelvis | r_hip | r_knee |
| r_ankle | l_hip | l_knee |
| l_ankle | spine | thorax |
| nose | head_top | l_shoulder |
| l_elbow | l_wrist | r_shoulder |
| r_elbow | r_wrist | |

### chunk_timing

Each activity includes a `chunk_timing` array that maps each 8-frame chunk to its temporal position:

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

| Phase | Description |
|-------|-------------|
| Step A | Seed2 + Cosmos + AVC-LM tokenisation (40 nodes x 4 GPU) |
| Phase 1 | HRNet 2D pose detection |
| Phase 2 | MotionBERT 2D to 3D lifting |
| Phase 2.5 | Resample to 30fps |
| Phase 3 | Kinematics: bone normalisation, root centering, smoothing |
| Phase 4 | YOLO person-detection cleaning |
| Phase 5 | Adaptive PCHIP per-joint tokenisation |
| **Phase 6** | **Merge agent tokens into multimodal dataset (this dataset)** |

## Reconstruction

To reconstruct 8 frames from control points, parse the t/x/y/z tokens per joint and apply PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolation.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("EmpathicRobotics/FineVideo-VLA-Agent", streaming=True)

for sample in ds["train"]:
    video_id = sample["video_id"]
    for scene in sample["scenes"]:
        for activity in scene["activities"]:
            tokens = activity["video_tokens"]
            if "<agent>" in tokens:
                print(f"Video {video_id} has 3D pose agent tokens")
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
