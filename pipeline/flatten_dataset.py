import json
import glob
import re
import os
import logging
from datetime import datetime

# Logging configuration
log_filename = f"flatten_file_by_file_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[logging.FileHandler(log_filename, encoding='utf-8'), logging.StreamHandler()]
)

def reformat_video_tokens(raw_tokens_str):
    if not raw_tokens_str: return ""
    def replacer(match):
        tag_name = match.group(1)
        numbers_str = match.group(2)
        prefix = "avclm" if tag_name == "avc_lm" else tag_name
        numbers = numbers_str.strip().split()
        formatted_nums = " ".join([f"<{prefix}_{n}>" for n in numbers])
        return f"<{tag_name}> {formatted_nums} </{tag_name}>"
    pattern = r"<([a-zA-Z0-9_]+)>\s*(.*?)\s*</\1>"
    return re.sub(pattern, replacer, raw_tokens_str)

def flatten_file_by_file():
    input_glob = "/e/project1/reformo/nguyen38/prototype/FineVideo-VLA/final_dataset/final_vla_rank_*.jsonl"
    output_dir = "/e/project1/reformo/nguyen38/prototype/FineVideo-VLA/megatron_dataset"
    os.makedirs(output_dir, exist_ok=True)

    files = glob.glob(input_glob)
    if not files:
        logging.error("❌ No data files found!")
        return

    logging.info(f"🔍 Found {len(files)} files. Processing file-by-file...")

    for file_idx, file in enumerate(files, start=1):
        base_name = os.path.basename(file)
        # Create a separate output file per input rank
        output_file = os.path.join(output_dir, f"flat_{base_name}")

        logging.info(f"▶️ Processing [{file_idx}/{len(files)}]: {base_name} ➔ flat_{base_name}")
        file_written = 0

        with open(output_file, 'w', encoding='utf-8') as f_out:
            with open(file, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    if not line.strip(): continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    for scene in data.get("scenes", []):
                        for activity in scene.get("activities", []):
                            prompt = activity.get("text_prompt", "").strip()
                            speech = activity.get("speech_transcript", "").strip()
                            raw_vid_tokens = activity.get("video_tokens", "")

                            if not raw_vid_tokens: continue

                            clean_vid_tokens = reformat_video_tokens(raw_vid_tokens)
                            user_text = prompt
                            if speech: user_text += f" [Speech: {speech}]"

                            flat_record = {
                                "text": f"USER: {user_text} ASSISTANT: {clean_vid_tokens}"
                            }
                            f_out.write(json.dumps(flat_record, ensure_ascii=False) + "\n")
                            file_written += 1

        logging.info(f"✅ Done with {base_name}. Extracted {file_written} records.")

        # Add a break condition here if you want to stop early (e.g., once enough data is generated).

if __name__ == "__main__":
    flatten_file_by_file()
