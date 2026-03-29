import argparse
import glob
import json
import math
import os
from collections import defaultdict


MIN_TOKENS_PER_VIDEO = 3
MIN_YIELD_RATE = 0.4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 6 - Macro filter dataset, compatible with current pipeline and SLURM."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing token JSONL files from the previous phase.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save clean dataset shards.",
    )
    parser.add_argument(
        "--log-dir",
        required=True,
        help="Directory to save flagged/discard logs.",
    )
    parser.add_argument(
        "--input-pattern",
        default="*_tokens.jsonl",
        help="Glob pattern used to discover token files inside --input-dir.",
    )
    return parser.parse_args()


def process_file(input_path, f_out, f_log):
    """Đọc 1 file token, gom nhóm theo video, lọc và ghi kết quả."""
    video_data = defaultdict(list)

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            video_id = data.get("video_id", "unknown")
            video_data[video_id].append(data)

    clean_count = 0
    discarded_count = 0

    for video_id, chunks in video_data.items():
        chunks.sort(key=lambda x: x["window_id"])

        num_tokens = len(chunks)
        if num_tokens == 0:
            continue

        min_id = chunks[0]["window_id"]
        max_id = chunks[-1]["window_id"]
        expected_tokens = ((max_id - min_id) // 16) + 1
        yield_rate = num_tokens / expected_tokens if expected_tokens > 0 else 0

        if num_tokens >= MIN_TOKENS_PER_VIDEO:
            for chunk in chunks:
                f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            clean_count += 1
        else:
            f_log.write(
                f"File: {os.path.basename(input_path)} | Video: {video_id} | "
                f"Tokens: {num_tokens}/{expected_tokens} (Yield: {yield_rate * 100:.1f}%)\n"
            )
            discarded_count += 1

    return clean_count, discarded_count


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))

    token_files = sorted(glob.glob(os.path.join(args.input_dir, args.input_pattern)))
    total_files = len(token_files)

    if total_files == 0:
        print(f"❌ [Worker {task_id}] Không tìm thấy file nào khớp pattern {args.input_pattern} trong {args.input_dir}")
        raise SystemExit(0)

    chunk_size = math.ceil(total_files / num_tasks)
    start_idx = (task_id - 1) * chunk_size
    end_idx = min(start_idx + chunk_size, total_files)
    my_files = token_files[start_idx:end_idx]

    print(f"🚀 [Worker {task_id}/{num_tasks}] Phân công xử lý {len(my_files)}/{total_files} files.")

    out_clean_path = os.path.join(args.output_dir, f"clean_dataset_part_{task_id:04d}.jsonl")
    out_log_path = os.path.join(args.log_dir, f"flagged_part_{task_id:04d}.txt")

    total_clean = 0
    total_discarded = 0
    total_seen = 0
    total_assigned = len(my_files)

    with open(out_clean_path, "w", encoding="utf-8") as f_out, open(out_log_path, "w", encoding="utf-8") as f_log:
        if total_assigned == 0:
            print(f"⚠️ [Worker {task_id}] Không có file nào được phân cho worker này.")
        else:
            for idx, file_path in enumerate(my_files, start=1):
                clean, discarded = process_file(file_path, f_out, f_log)
                total_clean += clean
                total_discarded += discarded
                total_seen += 1

                progress = (total_seen / total_assigned) * 100
                print(
                    f"   ⏳ [Worker {task_id}] {total_seen}/{total_assigned} files | "
                    f"{progress:.1f}% | Sạch: {total_clean} | Bỏ: {total_discarded}",
                    end="\n" if total_assigned == 1 else "\r",
                )

    if total_assigned > 1:
        print()

    print(f"✅ [Worker {task_id}] HOÀN THÀNH! Video đạt chuẩn: {total_clean} | Bị loại: {total_discarded}")
    print(f"   📁 Dữ liệu sạch: {out_clean_path}")
    print(f"   📋 Log lỗi    : {out_log_path}")
