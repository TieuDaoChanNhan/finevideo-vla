import json
import glob
import re
import os
import argparse
import logging
import multiprocessing as mp
from datetime import datetime

PASSTHROUGH_TAGS = {"agent"}
TAG_PATTERN = re.compile(r"<([a-zA-Z0-9_]+)>\s*(.*?)\s*</\1>")


def reformat_video_tokens(raw_tokens_str):
    if not raw_tokens_str:
        return ""

    def replacer(match):
        tag_name = match.group(1)
        if tag_name in PASSTHROUGH_TAGS:
            return match.group(0)
        numbers_str = match.group(2)
        prefix = "avclm" if tag_name == "avc_lm" else tag_name
        numbers = numbers_str.strip().split()
        formatted_nums = " ".join([f"<{prefix}_{n}>" for n in numbers])
        return f"<{tag_name}> {formatted_nums} </{tag_name}>"

    return TAG_PATTERN.sub(replacer, raw_tokens_str)


def flatten_one_file(args):
    input_file, output_file = args
    base_name = os.path.basename(input_file)

    if os.path.exists(output_file):
        size = os.path.getsize(output_file)
        if size > 0:
            return f"SKIP {base_name} (exists, {size / 1e6:.0f} MB)"

    tmp_file = output_file + ".tmp"
    file_written = 0

    with open(tmp_file, "w", encoding="utf-8") as f_out:
        with open(input_file, "r", encoding="utf-8") as f_in:
            for line in f_in:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                for scene in data.get("scenes", []):
                    for activity in scene.get("activities", []):
                        prompt = activity.get("text_prompt", "").strip()
                        speech = activity.get("speech_transcript", "").strip()
                        raw_vid_tokens = activity.get("video_tokens", "")

                        if not raw_vid_tokens:
                            continue

                        clean_vid_tokens = reformat_video_tokens(raw_vid_tokens)
                        user_text = prompt
                        if speech:
                            user_text += f" [Speech: {speech}]"

                        flat_record = {
                            "text": f"USER: {user_text} ASSISTANT: {clean_vid_tokens}"
                        }
                        f_out.write(json.dumps(flat_record, ensure_ascii=False) + "\n")
                        file_written += 1

    os.rename(tmp_file, output_file)
    return f"DONE {base_name} -> {file_written} records"


def main():
    parser = argparse.ArgumentParser(description="Flatten final_dataset_adaptive to Megatron JSONL")
    parser.add_argument(
        "--input-dir",
        default="/e/data1/datasets/playground/mmlaion/shared/nguyen38/FineVideo-VLA/final_dataset_adaptive",
    )
    parser.add_argument(
        "--output-dir",
        default="/e/data1/datasets/playground/mmlaion/shared/nguyen38/FineVideo-VLA/megatron_dataset_adaptive",
    )
    parser.add_argument("--workers", type=int, default=min(16, mp.cpu_count()))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, "final_vla_adaptive_rank_*.jsonl")))
    if not files:
        print("No input files found!")
        return

    tasks = []
    for f in files:
        base_name = os.path.basename(f)
        output_file = os.path.join(args.output_dir, f"flat_{base_name}")
        tasks.append((f, output_file))

    skip_count = sum(1 for _, o in tasks if os.path.exists(o) and os.path.getsize(o) > 0)
    print(f"Found {len(files)} files, {skip_count} already done, {len(files) - skip_count} to process")
    print(f"Using {args.workers} workers")

    with mp.Pool(args.workers) as pool:
        for result in pool.imap_unordered(flatten_one_file, tasks):
            print(result)

    print("All done.")


if __name__ == "__main__":
    main()
