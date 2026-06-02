"""
Upload rebuilt parquet shards to EmpathicRobotics/finevideo-3d-pose,
replacing the current train split entirely, and push an updated README.

Features:
  - Resume safe: checks which shards are already on HF and skips them.
  - Uploads README.md as the dataset card.

Usage:
    python tools/upload_parquet_hf.py \
        --parquet-dir /e/scratch/reformo/nguyen38/parquet_3d_shards_30fps \
        --repo        EmpathicRobotics/finevideo-3d-pose \
        --token       YOUR_HF_TOKEN   # or set HF_TOKEN env var
"""

import argparse
import os
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, login

README = """\
---
license: cc-by-4.0
task_categories:
  - robotics
  - video-classification
language:
  - en
tags:
  - 3d-pose
  - human-pose-estimation
  - video
  - vla
  - finevideo
pretty_name: FineVideo 3D Pose (30 fps)
size_categories:
  - 10K<n<100K
---

# FineVideo 3D Pose Dataset

3D human pose sequences extracted from the
[HuggingFace FineVideo dataset](https://huggingface.co/datasets/HuggingFaceFV/finevideo)
(~40 K YouTube videos), lifted to 3D with
[MotionBERT](https://github.com/Walter0807/MotionBERT) and **resampled to a
uniform 30 fps** to match the other modalities in the FineVideo-VLA pretraining
pipeline (Cosmos, AVC-LM, Seed2 tokenisers all operate at 30 fps).

## What changed in this version

| Field | Previous | This version |
|---|---|---|
| `joints_3d` | native fps (mixed: 24 / 25 / 29.97 / 30 …) | **resampled to 30 fps** via linear interpolation |
| `joints_xyzt` | not present | **added** — `(T, 17, 4)` with `[x, y, z, t_seconds]` per joint |
| `fps` | not present | **added** — original video fps for provenance |
| `num_frames` | native frame count | updated to 30 fps frame count |

The native fps ranged from 6 to 30 across the dataset (most common: 30 fps
→ 28 280 videos, 25 fps → 7 578 videos, 24 fps → 6 995 videos).

## Schema

| Column | Type | Description |
|---|---|---|
| `video_id` | `string` | YouTube video ID (matches FineVideo) |
| `num_frames` | `int32` | Number of frames **at 30 fps** |
| `joints_3d` | `bytes` | Raw `float32` array, shape `(num_frames, 17, 3)`, metres, H36M joint order |
| `joints_xyzt` | `bytes` | Raw `float32` array, shape `(num_frames, 17, 4)` — `[x, y, z, t_seconds]` per joint |
| `fps` | `float64` | **Original** video fps before resampling (for provenance) |

## Joint order (H36M, 17 joints)

```
 0 Pelvis       1 R_Hip        2 R_Knee      3 R_Ankle
 4 L_Hip        5 L_Knee       6 L_Ankle     7 Spine
 8 Thorax       9 Nose        10 Head       11 L_Shoulder
12 L_Elbow     13 L_Wrist     14 R_Shoulder 15 R_Elbow
16 R_Wrist
```

Coordinates are **root-centred** (pelvis = origin) and in metres.

## How to load

```python
from datasets import load_dataset
import numpy as np

ds = load_dataset("EmpathicRobotics/finevideo-3d-pose", split="train")

row = ds[0]

# (T, 17, 3) — x, y, z in metres at 30 fps
joints = np.frombuffer(row["joints_3d"], dtype=np.float32).reshape(row["num_frames"], 17, 3)

# (T, 17, 4) — x, y, z, t_seconds at 30 fps  ← includes timestamp per joint point
joints_xyzt = np.frombuffer(row["joints_xyzt"], dtype=np.float32).reshape(row["num_frames"], 17, 4)

print(f"video_id={row['video_id']}  original_fps={row['fps']}  shape={joints_xyzt.shape}")
# e.g. video_id=--5iwqOe8G8  original_fps=24.0  shape=(18996, 17, 4)
```

## How to load a single video by ID

```python
ds_filtered = ds.filter(lambda x: x["video_id"] == "YOUR_VIDEO_ID")
row = ds_filtered[0]
joints_xyzt = np.frombuffer(row["joints_xyzt"], dtype=np.float32).reshape(row["num_frames"], 17, 4)
xyz = joints_xyzt[:, :, :3]   # (T, 17, 3) positions
t   = joints_xyzt[:, 0, 3]    # (T,)       time in seconds (same for all joints)
```

## Resample back to native fps (if needed)

```python
from scipy.interpolate import interp1d

def resample(joints_30fps, native_fps, target_fps=30.0):
    N = len(joints_30fps)
    M = round(N * native_fps / target_fps)
    t_src = np.linspace(0, 1, N)
    t_dst = np.linspace(0, 1, M)
    flat = joints_30fps.reshape(N, -1).astype(np.float64)
    return interp1d(t_src, flat, axis=0)(t_dst).astype(np.float32).reshape(M, 17, 3)

joints_native = resample(joints, native_fps=row["fps"])
```

## Pipeline context (FineVideo-VLA)

This dataset is one component of the **FineVideo-VLA** pretraining corpus
(~25B tokens). Each video activity produces an interleaved token sequence:

```
USER: <activity_description> [Speech: ...]  ASSISTANT:
  <seed2> ... </seed2>      # 1 FPS semantic keyframe  (vocab 8192)
  <cosmos> ... </cosmos>    # every 8 frames spatial   (vocab 64000)
  <avc_lm> ... </avc_lm>   # every 8 frames H.264 BPE (vocab 8192)
  <agent> ... </agent>      # every 8 frames 3D pose   (vocab 256)
```

The `<agent>` tokens are derived from this dataset. Resampling to 30 fps
ensures the 8-frame pose windows align exactly with the 8-frame video chunks
used by the other tokenisers.

## Extraction pipeline

1. **Phase 1** — HRNet (MMPose) 2D keypoint detection, every frame
2. **Phase 2** — MotionBERT 3D lifting → `(N_native, 17, 3)` at native fps
3. **Phase 2.5** — Linear resampling to 30 fps → this dataset
4. **Phase 3** — Kinematics (velocity, acceleration) windowed at 30 fps
5. **Phase 4** — YOLO person-presence filter
6. **Phase 5** — PCHIP interpolation tokeniser → 256 uint8 tokens / 8-frame chunk
"""


def get_uploaded_shards(api: HfApi, repo: str) -> set:
    try:
        files = api.list_repo_files(repo_id=repo, repo_type="dataset")
        return {Path(f).name for f in files if f.startswith("data/") and f.endswith(".parquet")}
    except Exception:
        return set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--repo",        default="EmpathicRobotics/finevideo-3d-pose")
    parser.add_argument("--token",       default=None,
                        help="HF write token (fallback: HF_TOKEN env var)")
    parser.add_argument("--skip-readme", action="store_true",
                        help="Skip README upload (useful if re-running only parquet)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite all shards even if already on HF (use when data changed)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Pass --token or set HF_TOKEN env var")

    login(token=token, add_to_git_credential=False)
    api = HfApi()

    shards = sorted(Path(args.parquet_dir).glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No train-*.parquet found in {args.parquet_dir}")

    # Resume: find what's already on HF (skip if --force)
    if args.force:
        pending = shards
        print(f"--force: uploading all {len(pending)} shards (overwriting existing)", flush=True)
    else:
        print("Checking already-uploaded shards on HF ...", flush=True)
        already_uploaded = get_uploaded_shards(api, args.repo)
        pending = [s for s in shards if s.name not in already_uploaded]
        print(f"  {len(shards)} total shards, {len(already_uploaded)} already on HF, "
              f"{len(pending)} to upload", flush=True)

    for i, shard in enumerate(pending, 1):
        api.upload_file(
            path_or_fileobj=str(shard),
            path_in_repo=f"data/{shard.name}",
            repo_id=args.repo,
            repo_type="dataset",
            commit_message=f"30fps resampling + fps column: {shard.name}",
        )
        print(f"  [{i}/{len(pending)}] {shard.name}", flush=True)

    # Upload README
    if not args.skip_readme:
        print("\nUploading README.md ...", flush=True)
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as tmp:
            tmp.write(README)
            tmp_path = tmp.name
        try:
            api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo="README.md",
                repo_id=args.repo,
                repo_type="dataset",
                commit_message="Update dataset card: 30fps resampling, fps column, usage examples",
            )
        finally:
            os.unlink(tmp_path)
        print("  README.md uploaded.", flush=True)

    print(f"\nDone. https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
