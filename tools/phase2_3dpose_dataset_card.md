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
  - motionbert
language:
  - en
size_categories:
  - 10K<n<100K
---

# FineVideo-Phase2-3DPose — 3D Human Pose from MotionBERT

## Overview

This dataset contains **3D human pose** data lifted from 2D detections using [MotionBERT](https://github.com/Walter0807/MotionBERT), extracted from ~40K YouTube videos in the [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo) dataset.

This is the output of **Phase 2** (+ Phase 2.5 resampling) in the FineVideo-VLA pipeline. It contains raw 3D joint positions as NumPy arrays at 30fps, before any filtering, normalisation, or tokenisation.

## Statistics

| Metric | Value |
|--------|-------|
| Source videos | ~40,000 from FineVideo |
| Videos processed | 40,804 |
| Total size | ~259 GB (raw 3D) / ~67 GB (30fps resampled) |
| Frame rate | 30 fps (resampled from native video fps) |
| Joints per frame | 17 (H36M skeleton) |

## Pipeline Context

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | HRNet 2D pose detection (GPU) | Done |
| **Phase 2** | **MotionBERT 2D-to-3D lifting (this dataset)** | **Done** |
| Phase 2.5 | Resample all videos to 30fps | Done |
| Phase 3 | Kinematics: bone normalisation, root centering, smoothing | Done |
| Phase 4 | YOLO person-detection cleaning | Done |
| Phase 5 | Adaptive PCHIP per-joint tokenisation | Done |
| Phase 6 | Merge agent tokens into multimodal dataset | Done |
| Phase 7 | Flatten to Megatron-LM format | Done |
| Phase 8 | Megatron-LM tokenization (.bin/.idx) | Done |

## Data Format

Each record contains 3D joint positions for one video as a NumPy array:

- **Shape:** `(num_frames, 17, 3)` — frames at 30fps, 17 joints, xyz coordinates
- **Units:** metres (MotionBERT output space)
- **Coordinate system:** camera-relative (not root-centred — root centering happens in Phase 3)

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

## Processing Details

1. **Phase 1 (HRNet):** Ran HRNet with Faster R-CNN person detection to get 2D joint coordinates per frame
2. **Phase 2 (MotionBERT):** Lifted 2D poses to 3D using MotionBERT pretrained on Human3.6M, processed at native video fps
3. **Phase 2.5 (Resample):** Resampled from native video fps to uniform 30fps via linear interpolation, so poses align to the same time grid as video tokens (Seed2/Cosmos/AVC-LM)

## Downstream Processing

For cleaned and normalised poses, see [FineVideo-Phase4-YOLOPose](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase4-YOLOPose) which applies:
- Temporal smoothing (Butterworth filter)
- Bone length normalisation to canonical skeleton
- Root centering (pelvis at origin)
- Anti-teleportation filter
- YOLO person-presence cleaning

## Related Resources

| Resource | Description |
|----------|-------------|
| [EmpathicRobotics/FineVideo-Phase4-YOLOPose](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase4-YOLOPose) | Cleaned + normalised 3D poses (after Phase 3+4) |
| [EmpathicRobotics/FineVideo-Phase5-AgentTokens](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase5-AgentTokens) | Merged multimodal dataset with tokenised pose + video tokens |
| [EmpathicRobotics/FineVideo-Phase7-Flattened](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase7-Flattened) | Flat Megatron-LM JSONL (ready for pretraining) |
| [EmpathicRobotics/tokenizer-vla-adaptive](https://huggingface.co/EmpathicRobotics/tokenizer-vla-adaptive) | HuggingFace tokenizer (144,215 vocab) |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("EmpathicRobotics/FineVideo-Phase2-3DPose", streaming=True)

for sample in ds["train"]:
    video_id = sample["video_id"]
    poses_3d = sample["poses"]  # (num_frames, 17, 3)
    print(f"Video: {video_id}, Frames: {len(poses_3d)}")
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
