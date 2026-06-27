"""
Phase 5b — Per-joint XYZ tokenizer.

Replaces the opaque <agent_N> 256-token encoding with self-describing tokens:

    <fps_30> <joint_0_x_127> <joint_0_y_200> <joint_0_z_143> <joint_1_x_130> ...

Token stream layout per window (409 tokens total):
    1 fps token   : <fps_30>
    8 frames × 17 joints × 3 dims = 408 joint tokens

Joint order (H36M):
    0  pelvis (root, always [0,0,0] after centering)
    1  r_hip      2  r_knee     3  r_ankle
    4  l_hip      5  l_knee     6  l_ankle
    7  spine      8  thorax     9  nose    10  head_top
    11 l_shoulder 12 l_elbow   13 l_wrist
    14 r_shoulder 15 r_elbow   16 r_wrist

Quantization (same range as existing anchor encoding):
    uint8 N = clip(round((v + 2.0) / 4.0 * 255), 0, 255)
    range: [-2.0 m, +2.0 m]  precision: 4.0/255 ≈ 15.7 mm

Numpy output per video:  {video_id}_xyzt.npy
    shape  : (N_windows, 8, 17, 4)
    dtype  : float32
    dim[3] : [x_m, y_m, z_m, t_sec]
    t_sec  : (window_id + frame_offset) / 30.0

JSONL output per video:  {video_id}_tokens.jsonl
    {"video_id": str, "window_id": int, "fps": 30, "token_str": "<fps_30> <joint_0_x_127> ..."}

Input:   outputs/yolo_cleaned_30fps/{video_id}_cleaned.jsonl
Output:  outputs/agent_tokens_xyzt/{video_id}_tokens.jsonl
         outputs/agent_xyzt_npy/{video_id}_xyzt.npy

Run (SLURM, see slurm/submit_phase5b.sh):
    SLURM_ARRAY_TASK_ID=0 SLURM_ARRAY_TASK_COUNT=64 python pipeline/phase5b_xyzt_tokenizer.py \\
        --input-dir  outputs/yolo_cleaned_30fps \\
        --output-dir outputs/agent_tokens_xyzt \\
        --npy-dir    outputs/agent_xyzt_npy
"""

import argparse
import glob
import json
import os

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

TARGET_FPS    = 30
WINDOW_FRAMES = 8
N_JOINTS      = 17
COORD_RANGE   = 2.0      # ±2 m; all bone-normalised joints fit inside this
STRIDE        = 8        # only windows where window_id % STRIDE == 0

JOINT_NAMES = [
    "pelvis", "r_hip", "r_knee", "r_ankle",
    "l_hip",  "l_knee", "l_ankle",
    "spine",  "thorax", "nose",  "head_top",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_shoulder", "r_elbow", "r_wrist",
]

# ── Quantization ───────────────────────────────────────────────────────────────

def quantize(v: float) -> int:
    """Map a metric coordinate in [-2, +2] m to uint8 [0, 255]."""
    return int(np.clip(round((v + COORD_RANGE) / (2.0 * COORD_RANGE) * 255), 0, 255))


def dequantize(n: int) -> float:
    """Inverse of quantize — for reference / testing."""
    return n / 255.0 * (2.0 * COORD_RANGE) - COORD_RANGE


# ── Token builder ──────────────────────────────────────────────────────────────

def build_token_str(states: np.ndarray, fps: int = TARGET_FPS) -> str:
    """
    states : (8, 17, 3) float32, root-centred metric coordinates
    returns: '<fps_30> <joint_0_x_127> <joint_0_y_200> ... <joint_16_z_88>'
    """
    parts = [f"<fps_{fps}>"]
    for frame in range(WINDOW_FRAMES):
        for joint in range(N_JOINTS):
            x, y, z = states[frame, joint]
            parts.append(f"<joint_{joint}_x_{quantize(x)}>")
            parts.append(f"<joint_{joint}_y_{quantize(y)}>")
            parts.append(f"<joint_{joint}_z_{quantize(z)}>")
    return " ".join(parts)


def build_xyzt(states: np.ndarray, window_id: int, fps: int = TARGET_FPS) -> np.ndarray:
    """
    states    : (8, 17, 3) float32
    window_id : first frame index of the window in the 30fps array
    returns   : (8, 17, 4) float32  — [x, y, z, t_seconds]
    """
    xyzt = np.zeros((WINDOW_FRAMES, N_JOINTS, 4), dtype=np.float32)
    xyzt[:, :, :3] = states
    for f in range(WINDOW_FRAMES):
        xyzt[f, :, 3] = (window_id + f) / float(fps)
    return xyzt


# ── Per-file processing ────────────────────────────────────────────────────────

def process_file(
    input_path: str,
    output_jsonl: str,
    output_npy: str,
    video_id: str,
    stride: int = STRIDE,
) -> int:
    """
    Read one *_cleaned.jsonl, tokenize stride-filtered windows, write JSONL + npy.
    Returns number of windows written (0 means nothing written, no output files created).
    """
    records_jsonl = []
    xyzt_chunks   = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data      = json.loads(line)
            window_id = int(data["window_id"])

            if window_id % stride != 0:
                continue

            states = np.array(data["states"], dtype=np.float32)

            if states.shape != (WINDOW_FRAMES, N_JOINTS, 3):
                continue
            if np.isnan(states).any():
                continue

            token_str = build_token_str(states)
            xyzt      = build_xyzt(states, window_id)

            records_jsonl.append({
                "video_id":  video_id,
                "window_id": window_id,
                "fps":       TARGET_FPS,
                "token_str": token_str,
            })
            xyzt_chunks.append(xyzt)

    if not records_jsonl:
        return 0

    # Atomic write for JSONL
    tmp_jsonl = output_jsonl + ".tmp"
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(tmp_jsonl, "w", encoding="utf-8") as f:
        for rec in records_jsonl:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp_jsonl, output_jsonl)

    # Atomic write for npy  — shape (N_windows, 8, 17, 4)
    tmp_npy = output_npy + ".tmp.npy"
    os.makedirs(os.path.dirname(output_npy), exist_ok=True)
    np.save(tmp_npy, np.stack(xyzt_chunks, axis=0))
    os.replace(tmp_npy, output_npy)

    return len(records_jsonl)


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5b — per-joint XYZ tokenizer (self-describing token format)."
    )
    parser.add_argument("--input-dir",  required=True,
                        help="Directory containing *_cleaned.jsonl from Phase 4.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write *_tokens.jsonl files.")
    parser.add_argument("--npy-dir",    required=True,
                        help="Directory to write *_xyzt.npy files.")
    parser.add_argument("--stride",     type=int, default=STRIDE,
                        help=f"Keep only windows where window_id %% stride == 0. Default: {STRIDE}")
    parser.add_argument("--file-list",  default=None,
                        help="Optional text file listing specific *_cleaned.jsonl paths (one per line). "
                             "When provided, --input-dir is not scanned.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.npy_dir,    exist_ok=True)

    if args.file_list:
        with open(args.file_list) as f:
            all_files = [l.strip() for l in f if l.strip()]
    else:
        all_files = sorted(glob.glob(os.path.join(args.input_dir, "*_cleaned.jsonl")))

    task_id   = int(os.environ.get("SLURM_ARRAY_TASK_ID",   0))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    my_files  = [f for i, f in enumerate(all_files) if i % num_tasks == task_id]
    total     = len(my_files)

    print(f"\n[Worker {task_id}/{num_tasks}] {total} files to process.")
    print("=" * 60)

    processed = skipped = empty = 0

    for idx, input_path in enumerate(my_files, start=1):
        base     = os.path.basename(input_path)
        video_id = base[: -len("_cleaned.jsonl")]

        out_jsonl = os.path.join(args.output_dir, f"{video_id}_tokens.jsonl")
        out_npy   = os.path.join(args.npy_dir,    f"{video_id}_xyzt.npy")

        if os.path.exists(out_jsonl) and os.path.exists(out_npy):
            skipped += 1
            print(f"⏩ [{idx}/{total}] {video_id} — already done", end="\r")
            continue

        try:
            n = process_file(input_path, out_jsonl, out_npy, video_id, stride=args.stride)
            if n > 0:
                processed += 1
                pct = (processed + skipped + empty) / total * 100
                print(f"✅ [{idx}/{total}] {pct:.1f}% | {video_id} — {n} windows")
            else:
                empty += 1
        except Exception as e:
            print(f"❌ [{idx}/{total}] {video_id} — {e}")
            # clean up partial outputs
            for p in (out_jsonl + ".tmp", out_npy + ".tmp.npy"):
                if os.path.exists(p):
                    os.remove(p)

    print("\n" + "=" * 60)
    print(f"[Worker {task_id}] done — processed: {processed}, skipped: {skipped}, empty: {empty}")


if __name__ == "__main__":
    main()
