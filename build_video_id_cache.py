import os
import json
import glob

JSONL_DIR = "../prototype/FineVideo-VLA"
OUTPUT_PATH = "cached_video_ids.json"

def build_cache():
    target_ids = set()

    for file_path in glob.glob(os.path.join(JSONL_DIR, "training_ready_rank_*.jsonl")):
        print(f"Processing: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if "video_id" in data:
                        target_ids.add(str(data["video_id"]))
                except json.JSONDecodeError:
                    continue

    target_ids = sorted(list(target_ids))

    with open(OUTPUT_PATH, "w") as f:
        json.dump(target_ids, f)

    print(f"✅ Saved {len(target_ids)} video_ids to {OUTPUT_PATH}")

if __name__ == "__main__":
    build_cache()