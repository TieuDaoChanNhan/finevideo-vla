import os
import glob
import json
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def count_tokens_in_file(filepath):
    total_tokens = 0
    video_count = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    video_count += 1
                    for scene in data.get("scenes", []):
                        for act in scene.get("activities", []):
                            tokens_str = act.get("video_tokens", "")
                            if tokens_str:
                                total_tokens += len(tokens_str.split())
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"⚠️ Error reading {filepath}: {e}")

    return video_count, total_tokens

if __name__ == "__main__":
    file_list = glob.glob("training_ready_rank_*.jsonl")

    if not file_list:
        print("❌ No training_ready_rank_*.jsonl files found.")
        exit()

    print(f"🚀 Found {len(file_list)} files. Starting parallel count...")

    total_dataset_tokens = 0
    total_dataset_videos = 0

    with ProcessPoolExecutor(max_workers=32) as executor:
        results = list(tqdm(executor.map(count_tokens_in_file, file_list), total=len(file_list), desc="Scanning shards"))

    for v_count, t_count in results:
        total_dataset_videos += v_count
        total_dataset_tokens += t_count

    print("\n" + "="*50)
    print("📊 EXACT TOKEN COUNT")
    print("="*50)
    print(f"🎬 Total videos processed : {total_dataset_videos:,}")
    print(f"🪙 Total tokens produced  : {total_dataset_tokens:,}")
    print("="*50)
