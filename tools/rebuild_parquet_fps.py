"""
Rebuild parquet shards with three changes:
  1. Replace joints_3d bytes with the 30fps PCHIP-resampled version from 3d_npy_30fps/
  2. Add fps column (native fps from fps_lookup.json)
  3. Add joints_xyzt column: (T, 17, 4) float32 bytes — [x, y, z, t_seconds] per joint

Input shards:  --input-dir   (e.g. /e/scratch/reformo/nguyen38/parquet_3d_shards)
Output shards: --output-dir  (e.g. /e/scratch/reformo/nguyen38/parquet_3d_shards_30fps)
Resampled npy: --npy-dir     (e.g. outputs/3d_npy_30fps)
FPS lookup:    --fps-json    (e.g. outputs/fps_lookup.json)

Videos with no 30fps .npy or no fps entry are kept with original joints_3d
and fps=0.0 so the shard is never silently dropped.

Run locally (fast enough, pure CPU):
    python tools/rebuild_parquet_fps.py \
        --input-dir  /e/scratch/reformo/nguyen38/parquet_3d_shards \
        --output-dir /e/scratch/reformo/nguyen38/parquet_3d_shards_30fps \
        --npy-dir    outputs/3d_npy_30fps \
        --fps-json   outputs/fps_lookup.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_FPS = 30.0


def make_xyzt(arr: np.ndarray) -> bytes:
    """
    arr: (T, 17, 3) float32 at 30fps
    returns bytes of (T, 17, 4) float32 where dim 4 = [x, y, z, t_seconds]
    t_seconds is the wall-clock time of each frame: frame_idx / 30.0
    """
    T = arr.shape[0]
    t_col = (np.arange(T, dtype=np.float32) / TARGET_FPS)   # (T,)
    t_broadcast = np.broadcast_to(t_col[:, None, None], (T, 17, 1))
    xyzt = np.concatenate([arr, t_broadcast], axis=2)        # (T, 17, 4)
    return xyzt.astype(np.float32).tobytes()


def rebuild_shard(
    in_path: str,
    out_path: str,
    npy_dir: str,
    fps_lookup: dict,
) -> dict:
    df = pd.read_parquet(in_path)

    new_joints = []
    new_xyzt   = []
    new_fps    = []
    new_nframes = []
    replaced = missing_npy = missing_fps = 0

    for _, row in df.iterrows():
        vid = row["video_id"]
        npy_path = os.path.join(npy_dir, f"{vid}.npy")
        native_fps = fps_lookup.get(vid)

        if os.path.exists(npy_path):
            arr = np.load(npy_path).astype(np.float32)
            new_joints.append(arr.tobytes())
            new_xyzt.append(make_xyzt(arr))
            new_nframes.append(len(arr))
            replaced += 1
        else:
            # Keep original joints_3d; build xyzt from existing bytes
            arr_orig = np.frombuffer(row["joints_3d"], dtype=np.float32).reshape(row["num_frames"], 17, 3)
            new_joints.append(row["joints_3d"])
            new_xyzt.append(make_xyzt(arr_orig))
            new_nframes.append(row["num_frames"])
            missing_npy += 1

        if native_fps is not None:
            new_fps.append(float(native_fps))
        else:
            new_fps.append(0.0)
            missing_fps += 1

    df["joints_3d"]  = new_joints
    df["joints_xyzt"] = new_xyzt
    df["num_frames"] = new_nframes
    df["fps"]        = new_fps

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df.to_parquet(out_path, index=False)

    return {
        "shard": os.path.basename(in_path),
        "rows": len(df),
        "replaced": replaced,
        "missing_npy": missing_npy,
        "missing_fps": missing_fps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--npy-dir",    required=True)
    parser.add_argument("--fps-json",   required=True)
    args = parser.parse_args()

    with open(args.fps_json) as f:
        fps_lookup = json.load(f)

    shards = sorted(Path(args.input_dir).glob("train-*.parquet"))
    print(f"Found {len(shards)} parquet shards")

    total_replaced = total_missing_npy = total_missing_fps = 0

    for shard in shards:
        out_path = os.path.join(args.output_dir, shard.name)
        if os.path.exists(out_path):
            print(f"  SKIP (exists): {shard.name}")
            continue
        stats = rebuild_shard(str(shard), out_path, args.npy_dir, fps_lookup)
        total_replaced    += stats["replaced"]
        total_missing_npy += stats["missing_npy"]
        total_missing_fps += stats["missing_fps"]
        print(
            f"  {stats['shard']}: {stats['rows']} rows, "
            f"{stats['replaced']} replaced, "
            f"{stats['missing_npy']} missing npy, "
            f"{stats['missing_fps']} missing fps",
            flush=True,
        )

    print(f"\nAll shards done.")
    print(f"  Total replaced:    {total_replaced}")
    print(f"  Missing npy:       {total_missing_npy}")
    print(f"  Missing fps:       {total_missing_fps}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
