"""
Phase 5 — Adaptive PCHIP per-joint tokenizer.

For each 8-frame window from Phase 4, each of the 17 joints gets an
independent PCHIP compression with adaptive control-point count based
on per-joint curvature:

  Tier 2 (2 CPs: start + end)        — max curvature < tau_low
  Tier 4 (4 CPs: start + end + 2)    — tau_low <= curvature < tau_high
  Tier 8 (8 CPs: all frames)         — curvature >= tau_high

Token stream per window:
    <fps_30>
    <pelvis> <pelvis_t_0> <pelvis_x_N> <pelvis_y_N> <pelvis_z_N>
             <pelvis_t_7> <pelvis_x_N> <pelvis_y_N> <pelvis_z_N> </pelvis>
    <r_hip>  <r_hip_t_0> <r_hip_x_N> ...  </r_hip>
    ...

Quantization: [-2.0 m, +2.0 m] -> [0, 255]  (precision ~15.7 mm)
Time tokens:  frame index 0-7 within the 8-frame window

Input:   outputs/yolo_cleaned_30fps/{video_id}_cleaned.jsonl
Output:  outputs/agent_tokens_adaptive/{video_id}_tokens.jsonl
         Each line: {"video_id", "window_id", "fps", "token_str", "cp_counts"}
"""

import argparse
import glob
import json
import os

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_FPS = 30
WINDOW_FRAMES = 8
N_JOINTS = 17
COORD_RANGE = 2.0
STRIDE = 8

TAU_LOW = 0.005
TAU_HIGH = 0.05

JOINT_NAMES = [
    "pelvis", "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "spine", "thorax", "nose", "head_top",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_shoulder", "r_elbow", "r_wrist",
]

# ── Quantization ──────────────────────────────────────────────────────────────

def quantize(v: float) -> int:
    return int(np.clip(round((v + COORD_RANGE) / (2.0 * COORD_RANGE) * 255), 0, 255))


def dequantize(n: int) -> float:
    return n / 255.0 * (2.0 * COORD_RANGE) - COORD_RANGE


# ── Per-joint adaptive CP selection ───────────────────────────────────────────

def joint_curvature(trajectory: np.ndarray) -> float:
    """Max curvature (acceleration norm) for a single joint trajectory (8, 3)."""
    if trajectory.shape[0] < 3:
        return 0.0
    vel = np.diff(trajectory, axis=0)
    acc = np.diff(vel, axis=0)
    return float(np.max(np.linalg.norm(acc, axis=1)))


def select_cp_indices(trajectory: np.ndarray, tau_low: float, tau_high: float) -> np.ndarray:
    """Choose which frame indices become control points for one joint."""
    curv = joint_curvature(trajectory)

    if curv >= tau_high:
        return np.arange(WINDOW_FRAMES)

    if curv < tau_low:
        return np.array([0, WINDOW_FRAMES - 1])

    # Tier 4: start + end + 2 highest-curvature interior frames
    vel = np.diff(trajectory, axis=0)
    acc = np.diff(vel, axis=0)
    acc_norms = np.linalg.norm(acc, axis=1)  # (6,)

    # acc[i] corresponds to frame i+1 (second derivative offset)
    interior_curv = np.zeros(WINDOW_FRAMES)
    for i in range(len(acc_norms)):
        interior_curv[i + 1] = acc_norms[i]

    # Exclude endpoints (already included), pick top 2 interior frames
    interior_curv[0] = -1.0
    interior_curv[-1] = -1.0
    top2 = np.argsort(interior_curv)[-2:]

    indices = np.unique(np.sort(np.concatenate(([0], top2, [WINDOW_FRAMES - 1]))))
    return indices.astype(int)


# ── Token builder ─────────────────────────────────────────────────────────────

def build_token_str(
    states: np.ndarray,
    fps: int = TARGET_FPS,
    tau_low: float = TAU_LOW,
    tau_high: float = TAU_HIGH,
) -> tuple:
    """
    states : (8, 17, 3) float32, root-centred metric coordinates
    Returns (token_str, cp_counts_dict)
    """
    parts = [f"<fps_{fps}>"]
    cp_counts = {}

    for j in range(N_JOINTS):
        name = JOINT_NAMES[j]
        trajectory = states[:, j, :]  # (8, 3)
        cp_idx = select_cp_indices(trajectory, tau_low, tau_high)
        cp_counts[name] = len(cp_idx)

        parts.append(f"<{name}>")
        for fi in cp_idx:
            x, y, z = trajectory[fi]
            parts.append(f"<{name}_t_{fi}>")
            parts.append(f"<{name}_x_{quantize(x)}>")
            parts.append(f"<{name}_y_{quantize(y)}>")
            parts.append(f"<{name}_z_{quantize(z)}>")
        parts.append(f"</{name}>")

    return " ".join(parts), cp_counts


# ── Per-file processing ──────────────────────────────────────────────────────

def process_file(
    input_path: str,
    output_jsonl: str,
    video_id: str,
    stride: int = STRIDE,
    tau_low: float = TAU_LOW,
    tau_high: float = TAU_HIGH,
) -> int:
    records = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            window_id = int(data["window_id"])

            if window_id % stride != 0:
                continue

            states = np.array(data["states"], dtype=np.float32)

            if states.shape != (WINDOW_FRAMES, N_JOINTS, 3):
                continue
            if np.isnan(states).any():
                continue

            token_str, cp_counts = build_token_str(states, TARGET_FPS, tau_low, tau_high)

            records.append({
                "video_id": video_id,
                "window_id": window_id,
                "fps": TARGET_FPS,
                "token_str": token_str,
                "cp_counts": cp_counts,
            })

    if not records:
        return 0

    tmp = output_jsonl + ".tmp"
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, output_jsonl)

    return len(records)


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 5 — Adaptive PCHIP per-joint tokenizer."
    )
    p.add_argument("--input-dir", required=True,
                    help="Directory with *_cleaned.jsonl from Phase 4.")
    p.add_argument("--output-dir", required=True,
                    help="Directory to write *_tokens.jsonl files.")
    p.add_argument("--stride", type=int, default=STRIDE,
                    help=f"Keep windows where window_id %% stride == 0. Default: {STRIDE}")
    p.add_argument("--tau-low", type=float, default=TAU_LOW,
                    help=f"Curvature threshold for 2-CP tier. Default: {TAU_LOW}")
    p.add_argument("--tau-high", type=float, default=TAU_HIGH,
                    help=f"Curvature threshold for 8-CP tier. Default: {TAU_HIGH}")
    p.add_argument("--file-list", default=None,
                    help="Optional text file listing specific *_cleaned.jsonl paths.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.file_list:
        with open(args.file_list) as f:
            all_files = [l.strip() for l in f if l.strip()]
    else:
        all_files = sorted(glob.glob(os.path.join(args.input_dir, "*_cleaned.jsonl")))

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    my_files = [f for i, f in enumerate(all_files) if i % num_tasks == task_id]
    total = len(my_files)

    print(f"\n[Worker {task_id}/{num_tasks}] {total} files to process.")
    print("=" * 60)

    processed = skipped = empty = 0
    tier_counts = {2: 0, 4: 0, 8: 0}

    for idx, input_path in enumerate(my_files, start=1):
        base = os.path.basename(input_path)
        video_id = base[: -len("_cleaned.jsonl")]

        out_jsonl = os.path.join(args.output_dir, f"{video_id}_tokens.jsonl")

        if os.path.exists(out_jsonl):
            skipped += 1
            print(f"[{idx}/{total}] {video_id} — already done", end="\r")
            continue

        try:
            n = process_file(
                input_path, out_jsonl, video_id,
                stride=args.stride,
                tau_low=args.tau_low,
                tau_high=args.tau_high,
            )
            if n > 0:
                processed += 1
                pct = (processed + skipped + empty) / total * 100
                print(f"[{idx}/{total}] {pct:.1f}% | {video_id} — {n} windows")
            else:
                empty += 1
        except Exception as e:
            print(f"[{idx}/{total}] ERROR {video_id} — {e}")
            for p in (out_jsonl + ".tmp",):
                if os.path.exists(p):
                    os.remove(p)

    print("\n" + "=" * 60)
    print(f"[Worker {task_id}] done — processed: {processed}, skipped: {skipped}, empty: {empty}")


if __name__ == "__main__":
    main()
