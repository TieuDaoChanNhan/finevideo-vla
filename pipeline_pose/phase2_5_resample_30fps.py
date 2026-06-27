"""
Phase 2.5 — Resample native-fps 3D pose arrays to 30 fps.

Reads   outputs/3d_npy/{video_id}.npy         shape (N_native, 17, 3)
Writes  outputs/3d_npy_30fps/{video_id}.npy   shape (N_30, 17, 3)

Uses linear interpolation along the time axis only.
Videos whose native fps is already 30 (or very close) are copied directly.
Videos missing from fps_lookup.json are skipped with a warning.

Run locally with multiprocessing (recommended):
    python pipeline/phase2_5_resample_30fps.py \
        --input-dir  outputs/3d_npy \
        --output-dir outputs/3d_npy_30fps \
        --fps-json   outputs/fps_lookup.json \
        --workers    32

Run as SLURM array (see slurm/submit_phase2_5.sh):
    same command without --workers
"""

import argparse
import json
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.interpolate import interp1d

TARGET_FPS = 30.0
FPS_TOLERANCE = 0.05   # treat as already 30fps if within ±5%


def resample_pose(arr: np.ndarray, src_fps: float, dst_fps: float = TARGET_FPS) -> np.ndarray:
    """
    Resample (N, 17, 3) from src_fps to dst_fps using linear interpolation.

    Linear is intentionally chosen here over PCHIP/cubic because the raw
    MotionBERT output still contains [0,0,0] frames (no person detected).
    Cubic splines overshoot wildly at the valid-frame boundaries adjacent to
    those zero runs, corrupting otherwise good poses by tens of metres.
    Linear interpolation passes cleanly through zeros without overshoot.

    PCHIP is the right tool downstream in Phase 5, where Phase 4 has already
    removed all zero frames before spline fitting.
    """
    N = arr.shape[0]
    if N < 2:
        return arr

    duration = N / src_fps
    M = max(2, round(duration * dst_fps))

    t_src = np.linspace(0.0, 1.0, N)
    t_dst = np.linspace(0.0, 1.0, M)

    flat = arr.reshape(N, -1).astype(np.float64)
    f = interp1d(t_src, flat, axis=0, kind="linear", assume_sorted=True)
    return f(t_dst).astype(np.float32).reshape(M, 17, 3)


def process_one(args_tuple):
    npy_path, out_path, native_fps = args_tuple
    arr = np.load(npy_path)
    if arr.ndim != 3 or arr.shape[1:] != (17, 3):
        return "bad_shape"

    if abs(native_fps / TARGET_FPS - 1.0) < FPS_TOLERANCE:
        np.save(out_path, arr)
    else:
        np.save(out_path, resample_pose(arr, native_fps))
    return "done"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps-json",   required=True)
    parser.add_argument("--workers",    type=int, default=0,
                        help="Parallel workers. 0 = use SLURM_ARRAY partitioning (sequential per worker).")
    args = parser.parse_args()

    with open(args.fps_json) as f:
        fps_lookup = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    all_npy = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))

    if args.workers > 0:
        # Local multiprocessing — process all files across N workers
        my_files = all_npy
        print(f"[local] {len(my_files)} files, {args.workers} workers", flush=True)
    else:
        # SLURM array partitioning
        task_id   = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
        my_files  = [f for i, f in enumerate(all_npy) if i % num_tasks == task_id]
        print(f"[SLURM worker {task_id}/{num_tasks}] {len(my_files)} files", flush=True)

    # Build work list — skip already-done and missing-fps upfront
    work = []
    already = no_fps = 0
    for npy_path in my_files:
        video_id = os.path.basename(npy_path).replace(".npy", "")
        out_path = os.path.join(args.output_dir, f"{video_id}.npy")
        if os.path.exists(out_path):
            already += 1
            continue
        native_fps = fps_lookup.get(video_id)
        if native_fps is None:
            no_fps += 1
            continue
        work.append((npy_path, out_path, native_fps))

    print(f"  {len(work)} to process, {already} already done, {no_fps} missing fps", flush=True)

    done = bad_shape = 0

    if args.workers > 0:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_one, item): item for item in work}
            for i, fut in enumerate(as_completed(futures), 1):
                result = fut.result()
                if result == "done":
                    done += 1
                else:
                    bad_shape += 1
                if i % 1000 == 0:
                    print(f"  {i}/{len(work)} done ...", flush=True)
    else:
        for item in work:
            result = process_one(item)
            if result == "done":
                done += 1
            else:
                bad_shape += 1
            if (done + bad_shape) % 1000 == 0:
                print(f"  {done + bad_shape}/{len(work)} done ...", flush=True)

    print(f"\nDone: {done} resampled/copied, {bad_shape} bad shape, "
          f"{already} already existed, {no_fps} missing fps", flush=True)


if __name__ == "__main__":
    main()
