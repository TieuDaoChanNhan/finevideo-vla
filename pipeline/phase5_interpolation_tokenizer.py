import argparse
import glob
import json
import math
import os

import numpy as np
from scipy.interpolate import PchipInterpolator


class AdaptiveInterpolationTokenizer:
    """
    Adaptive Interpolation Tokenizer (Robotics-Safe Version)
    - Control points lie EXACTLY on the trajectory.
    - Adaptive sampling based on curvature.
    - Uses PCHIP to prevent dangerous trajectory overshoots.
    """

    def __init__(self, frames_per_chunk=8):
        self.frames_per_chunk = frames_per_chunk

    def compute_adaptive_indices(self, chunk_positions):
        velocity = np.diff(chunk_positions, axis=0)
        acceleration = np.diff(velocity, axis=0)
        curvature = np.mean(np.linalg.norm(acceleration, axis=2), axis=1)

        top_2_idx = np.argsort(curvature)[-2:] + 1
        top_2_idx = np.clip(top_2_idx, 1, 6)
        top_2_idx = np.unique(top_2_idx)

        for c in [2, 3, 4, 5]:
            if len(top_2_idx) >= 2:
                break
            if c not in top_2_idx:
                top_2_idx = np.append(top_2_idx, c)

        adaptive_idx = np.sort(np.concatenate(([0], top_2_idx, [7])))
        return adaptive_idx.astype(int)

    def compute_arc_length_param(self, chunk_positions):
        joint_diffs = np.linalg.norm(
            chunk_positions[1:] - chunk_positions[:-1], axis=2
        )
        mean_diffs = np.mean(joint_diffs, axis=1)
        
        # FIX: Force arc length to always increment slightly (1e-5) to prevent time from freezing
        mean_diffs = np.maximum(mean_diffs, 1e-5)
        
        s = np.concatenate([[0], np.cumsum(mean_diffs)])
        if s[-1] > 0:
            s = s / s[-1]
        return s

    def quantize(self, x):
        x = np.clip(x, -1.0, 1.0)
        return ((x + 1.0) * 127.5).astype(np.uint8)

    def dequantize(self, x):
        return (x.astype(np.float32) / 127.5) - 1.0

    def encode_chunk(self, chunk_positions, time_delta=0.26):
        anchor = chunk_positions[0].copy()
        rel = chunk_positions - anchor
        scale = np.max(np.abs(rel)) + 1e-6
        norm = rel / scale

        t_eval = self.compute_arc_length_param(norm)
        sample_idx = self.compute_adaptive_indices(norm)

        cp = norm[sample_idx]
        t_cp = t_eval[sample_idx]
        tokens = self.quantize(cp.flatten())

        return {
            "time_delta": float(time_delta),
            "anchor": anchor.tolist(),
            "scale": float(scale),
            "sample_idx": sample_idx.tolist(),
            "t_cp": t_cp.tolist(),
            "tokens": tokens.tolist(),
        }

    def decode_chunk(self, package):
        tokens = np.array(package["tokens"])
        scale = package["scale"]
        anchor = np.array(package["anchor"])

        cp_norm = self.dequantize(tokens).reshape(4, 17, 3)
        cp = cp_norm * scale + anchor
        return cp

    def reconstruct(self, cp, t_cp):
        t_cp = np.array(t_cp, dtype=np.float64)
        
        for i in range(1, len(t_cp)):
            if t_cp[i] <= t_cp[i-1]:
                t_cp[i] = t_cp[i-1] + 1e-5
        if t_cp[-1] > 0:
            t_cp = t_cp / t_cp[-1]

        t = np.linspace(0, 1, 50)
        recon = np.zeros((50, 17, 3))

        for j in range(17):
            for d in range(3):
                spline = PchipInterpolator(t_cp, cp[:, j, d])
                recon[:, j, d] = spline(t)

        return recon


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 5 - Interpolation tokenizer, compatible with current pipeline and SLURM."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing *_cleaned.jsonl files from Phase 4.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save *_tokens.jsonl outputs.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Keep only windows where window_id %% stride == 0.",
    )
    parser.add_argument(
        "--frames-per-chunk",
        type=int,
        default=8,
        help="Number of frames per chunk used by the tokenizer.",
    )
    parser.add_argument(
        "--file-list",
        default=None,
        help="Optional path to a text file listing specific *_cleaned.jsonl paths to process "
             "(one per line). When provided, --input-dir is only used as a fallback for the "
             "output path and is not scanned.",
    )
    return parser.parse_args()


def extract_video_id(input_path):
    base_name = os.path.basename(input_path)
    if base_name.endswith("_cleaned.jsonl"):
        return base_name[: -len("_cleaned.jsonl")]
    return os.path.splitext(base_name)[0]


def build_output_path(output_dir, input_path):
    video_id = extract_video_id(input_path)
    return os.path.join(output_dir, f"{video_id}_tokens.jsonl")


def process_state_file(input_path, output_path, tokenizer, video_id, stride=16):
    """
    Read *_cleaned.jsonl, filter duplicates by stride, and compress to tokens.
    """
    valid_chunks = 0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_path = f"{output_path}.tmp"

    try:
        with open(input_path, "r", encoding="utf-8") as f_in, open(temp_path, "w", encoding="utf-8") as f_out:
            for line in f_in:
                if not line.strip():
                    continue

                data = json.loads(line)

                if data["window_id"] % stride != 0:
                    continue

                states = np.array(data["states"], dtype=float)

                if np.isnan(states).any():
                    continue

                package = tokenizer.encode_chunk(states)

                record = {
                    "video_id": video_id,
                    "window_id": data["window_id"],
                    "package": package,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                valid_chunks += 1

        if valid_chunks > 0:
            os.replace(temp_path, output_path)
        else:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    return valid_chunks


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AdaptiveInterpolationTokenizer(frames_per_chunk=args.frames_per_chunk)

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))

    if args.file_list:
        with open(args.file_list, "r") as f:
            state_files = [line.strip() for line in f if line.strip()]
        state_files = sorted(state_files)
    else:
        state_files = sorted(glob.glob(os.path.join(args.input_dir, "*_cleaned.jsonl")))

    total_files = len(state_files)

    if total_files == 0:
        print(
            f"❌ [Worker {task_id}] No files to process "
            f"({'from --file-list ' + args.file_list if args.file_list else 'in ' + args.input_dir})"
        )
        raise SystemExit(0)

    chunk_size = math.ceil(total_files / num_tasks)
    start_idx = (task_id - 1) * chunk_size
    end_idx = min(start_idx + chunk_size, total_files)
    my_files = state_files[start_idx:end_idx]

    print(
        f"🚀 [Worker {task_id}/{num_tasks}] Assigned {len(my_files)}/{total_files} files to process."
    )
    print("=" * 60)

    processed = 0
    skipped = 0
    total_tokens_generated = 0

    for idx, input_path in enumerate(my_files, start=1):
        video_id = extract_video_id(input_path)
        output_path = build_output_path(args.output_dir, input_path)

        if os.path.exists(output_path):
            skipped += 1
            print(
                f"⏩ [Worker {task_id}] Checked: {idx}/{len(my_files)} | Resumed: {skipped}",
                end="\r",
            )
            continue

        try:
            tokens_count = process_state_file(
                input_path=input_path,
                output_path=output_path,
                tokenizer=tokenizer,
                video_id=video_id,
                stride=args.stride,
            )
            processed += 1
            total_tokens_generated += tokens_count

            progress = (processed + skipped) / len(my_files) * 100
            print(
                f"✅ [Worker {task_id}] {progress:.1f}% | Processed: {processed} | Tokens: {total_tokens_generated}",
                end="\r",
            )

        except Exception as e:
            print(f"\n❌ Error processing {input_path}: {e}")

    print(
        f"\n🎉 [Worker {task_id}] COMPLETED! Successfully compressed "
        f"{total_tokens_generated} tokens from {processed} files "
        f"(Skipped: {skipped})."
    )