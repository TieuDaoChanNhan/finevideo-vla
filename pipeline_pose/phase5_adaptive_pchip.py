"""
Phase 5 — Adaptive PCHIP per-joint tokenizer.

For each window from Phase 4 (WINDOW_FRAMES frames -- 8 originally, 24 as of
2026-07-22, see REPORT.md #38), each of the 17 joints gets an independent
PCHIP compression with adaptive control-point count based on per-joint
curvature over the WHOLE window (all WINDOW_FRAMES frames considered, not a
fixed sub-grid):

  Tier 2       (2 CPs: start + end)                  — max curvature < tau_low
  Tier 4       (4 CPs: start + end + top-2 interior)  — tau_low <= curvature < tau_high
  Tier MAX_CPS (MAX_CPS CPs: start+end+top-(MAX_CPS-2) interior, chosen by
                curvature out of ALL WINDOW_FRAMES candidates) — curvature >= tau_high

MAX_CPS is fixed at 8 regardless of WINDOW_FRAMES -- widening the window
gives the top tier more candidate positions to pick its best 8 from, not more
tokens/joint. See MAX_CPS's own docstring below for why this matters.

Token stream per window (WINDOW_FRAMES=24 example; t values now range over
however many of the window's real frame indices got chosen as CPs, e.g.
t_0 and t_23 for tier 2, or t_0/t_5/t_14/t_23 for tier 4 if frames 5 and 14
had the highest curvature):
    <fps_30>
    <pelvis> <pelvis_t_0> <pelvis_x_N> <pelvis_y_N> <pelvis_z_N>
             <pelvis_t_23> <pelvis_x_N> <pelvis_y_N> <pelvis_z_N> </pelvis>
    <r_hip>  <r_hip_t_0> <r_hip_x_N> ...  </r_hip>
    ...

Quantization: [-2.0 m, +2.0 m] -> [0, 255]  (precision ~15.7 mm)
Time tokens:  real frame index within the window, 0 to WINDOW_FRAMES-1 --
              requires <{joint}_t_N> tokens up to WINDOW_FRAMES-1 in the
              tokenizer vocab (only 0-7 existed before 2026-07-22).

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
# 2026-07-22 (REPORT.md #38): cap on control points per joint, independent of
# WINDOW_FRAMES. Before this change WINDOW_FRAMES==MAX_CPS==8 always (the top
# tier was literally "use every frame"), so widening the window to 24 frames
# would have silently tripled worst-case tokens/joint (24 CPs instead of 8)
# with no code change needed to trigger it. Keeping MAX_CPS fixed at 8 means
# the top tier now means "pick the best 8 of WINDOW_FRAMES candidates by
# curvature" instead of "use all of them" -- same worst-case token cost as
# before, but the 8 chosen points can be anywhere in the (now wider) window
# instead of forced onto a fixed 8-slot grid. This is the whole point of
# "Option 2" (dense pose, no subsampling before curve-fitting) agreed with
# the user: Phase 3 keeps every real frame, Phase 5 decides freely which
# frames matter most.
MAX_CPS = 8

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
    """Choose which frame indices become control points for one joint.

    Tier sizes are fixed at 2 / 4 / MAX_CPS regardless of how many frames are
    in the window (see MAX_CPS's docstring) -- the top tier picks the
    MAX_CPS-2 highest-curvature *interior* frames out of every candidate in
    the window, not a fixed grid position. With WINDOW_FRAMES==MAX_CPS==8
    (the original config) this is exactly equivalent to the old
    `np.arange(WINDOW_FRAMES)` behavior, since "top 6 of 6 interior
    candidates" is all of them.
    """
    curv = joint_curvature(trajectory)
    n_frames = trajectory.shape[0]

    if curv < tau_low:
        return np.array([0, n_frames - 1])

    n_interior = 2 if curv < tau_high else (MAX_CPS - 2)
    n_interior = min(n_interior, max(n_frames - 2, 0))

    vel = np.diff(trajectory, axis=0)
    acc = np.diff(vel, axis=0)
    acc_norms = np.linalg.norm(acc, axis=1)  # (n_frames-2,)

    # acc[i] corresponds to frame i+1 (second derivative offset)
    interior_curv = np.zeros(n_frames)
    for i in range(len(acc_norms)):
        interior_curv[i + 1] = acc_norms[i]

    # Exclude endpoints (already included), pick top-n_interior interior frames
    interior_curv[0] = -1.0
    interior_curv[-1] = -1.0
    top_n = np.argsort(interior_curv)[-n_interior:] if n_interior > 0 else np.array([], dtype=int)

    indices = np.unique(np.sort(np.concatenate(([0], top_n, [n_frames - 1]))))
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
    p.add_argument("--window-frames", type=int, default=WINDOW_FRAMES,
                    help=f"Must match Phase 3/4's --window-size. Default: {WINDOW_FRAMES}. "
                         f"2026-07-22: use 24 to match the wider cosmos chunk window -- "
                         f"see REPORT.md #38. MAX_CPS (top-tier control-point count) is "
                         f"NOT tied to this and stays 8 either way.")
    return p.parse_args()


def main() -> None:
    global WINDOW_FRAMES
    args = parse_args()
    WINDOW_FRAMES = args.window_frames
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
