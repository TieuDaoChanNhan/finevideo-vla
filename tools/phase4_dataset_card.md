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
language:
  - en
size_categories:
  - 10M<n<100M
---

# FineVideo Phase 4 — YOLO-Cleaned 3D Human Pose (30fps)

## Overview

This dataset contains **YOLO-cleaned, bone-normalised 3D human pose** data extracted from ~40K YouTube videos in the [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo) dataset. It is the output of **Phase 4** in the FineVideo-VLA pipeline and serves as input to Phase 5 (adaptive PCHIP tokenisation for LLM pretraining).

Use this dataset if you need **raw 3D joint positions** (floats in metres, not tokenised). For tokenised versions, see the related datasets below.

## Statistics

| Metric | Value |
|--------|-------|
| Source videos | ~40,000 from FineVideo |
| Videos after cleaning | 40,195 |
| Total size | ~107 GB (uncompressed JSONL) |
| Frame rate | 30 fps (resampled from native video fps) |
| Joints per frame | 17 (H36M skeleton) |
| Frames per window | 8 (~0.267 seconds) |

## Pipeline Context

This dataset is part of a multi-phase pipeline that produces the **FineVideo-VLA** multimodal pretraining dataset:

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | HRNet 2D pose detection (GPU) | Done |
| Phase 2 | MotionBERT 2D→3D lifting (GPU) | Done |
| Phase 2.5 | Resample all videos to 30fps | Done |
| Phase 3 | Kinematics: bone normalisation, root centering, smoothing, hallucination filtering | Done |
| **Phase 4** | **YOLO person-detection cleaning (this dataset)** | **Done** |
| Phase 5 | Adaptive PCHIP per-joint tokenisation | Done |
| Phase 6 | Merge agent tokens into multimodal dataset | Done |
| Phase 7 | Flatten to Megatron-LM format | Done |
| Phase 8 | Megatron-LM tokenization (.bin/.idx) | Done |

## Data Format

Each record is a JSON line:

```json
{
  "video_id": "abc123XYZ",
  "window_id": 320,
  "states": [[[x, y, z], ...17 joints...], ...8 frames...]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `video_id` | string | YouTube video ID |
| `window_id` | int | Absolute frame index of the first frame in this window |
| `states` | float[8][17][3] | 3D joint positions in metres |

### Timestamp

Absolute timestamp from video start: `window_id / 30.0` seconds.

Each window covers 8 frames = `8/30 = 0.267` seconds.

### Joint coordinates

- **Root-centred**: pelvis (joint 0) is always at origin `[0, 0, 0]`
- **Bone-normalised**: skeleton retargeted to canonical bone lengths
- **Smoothed**: temporal smoothing + anti-teleportation filter applied in Phase 3
- **Coordinate range**: typically +/-0.5m, max +/-2.0m

### Joint order (H36M 17-joint skeleton)

| Index | Joint | Index | Joint |
|-------|-------|-------|-------|
| 0 | pelvis (root) | 9 | nose |
| 1 | right hip | 10 | head top |
| 2 | right knee | 11 | left shoulder |
| 3 | right ankle | 12 | left elbow |
| 4 | left hip | 13 | left wrist |
| 5 | left knee | 14 | right shoulder |
| 6 | left ankle | 15 | right elbow |
| 7 | spine | 16 | right wrist |
| 8 | thorax | | |

### Skeleton connectivity

```
        head_top (10)
            |
          nose (9)
            |
        thorax (8)
       /    |    \
  l_sh(11) spine(7) r_sh(14)
   |        |        |
 l_el(12) pelvis(0) r_el(15)
   |      /    \      |
 l_wr(13) l_hip r_hip r_wr(16)
         (4)    (1)
          |      |
        l_kn   r_kn
         (5)    (2)
          |      |
        l_an   r_an
         (6)    (3)
```

## Window structure

- Each window = **8 consecutive frames** at **30fps** (~0.267 seconds)
- `window_id` = absolute frame index (always a multiple of 8 after stride filtering)
- Absolute timestamp: `window_id / 30.0` seconds from video start

## YOLO cleaning (Phase 4)

Windows are dropped if **>= 4 of 8 frames** have no person detected by YOLOv8 (confidence >= 0.75). This removes windows where the subject is off-screen, occluded, or in a scene transition.

Some windows may still contain `null`/`NaN` values for individual joints where the pose estimator failed — downstream consumers should check for this.

## Related Resources

| Resource | Description |
|----------|-------------|
| [EmpathicRobotics/FineVideo-Phase5-AgentTokens](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase5-AgentTokens) | Merged multimodal dataset with tokenised pose + video tokens (hierarchical, full metadata) |
| [EmpathicRobotics/FineVideo-Phase7-Flattened](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase7-Flattened) | Flat Megatron-LM JSONL (ready for pretraining) |
| [EmpathicRobotics/tokenizer-vla-adaptive](https://huggingface.co/EmpathicRobotics/tokenizer-vla-adaptive) | HuggingFace tokenizer (144,215 vocab) |

## Usage

```python
from datasets import load_dataset
import numpy as np

ds = load_dataset("EmpathicRobotics/FineVideo-Phase4-YOLOPose", streaming=True)

for sample in ds["train"]:
    video_id = sample["video_id"]
    window_id = sample["window_id"]
    states = np.array(sample["states"])  # (8, 17, 3)
    timestamp = window_id / 30.0         # seconds from video start

    print(f"Video: {video_id}")
    print(f"Window: {window_id} ({timestamp:.3f}s)")
    print(f"Pelvis (frame 0): {states[0, 0]}")  # always [0, 0, 0]
    print(f"Right wrist (frame 0): {states[0, 16]}")
    break
```

## Citation

Part of the FineVideo-VLA project. If you use this data, please cite the FineVideo dataset:

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
