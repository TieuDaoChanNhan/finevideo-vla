import os
import json
from datasets import load_from_disk
from tqdm import tqdm

# ================= CONFIGURATION =================
JSONL_DIR = "../prototype/FineVideo-VLA"
DATASET_PATH = "/e/scratch/reformo/nguyen38/finevideo_disk"
STAGING_DIR = "videos_staging"
NUM_SHARDS_TO_FETCH = 6 # Fetch from rank_0 to rank_5

def load_target_video_ids(jsonl_dir, num_shards):
    """
    Read multiple JSONL files and load all video_ids into an O(1) hash set.
    """
    target_ids = set()
    print(f"🔍 Scanning JSONL files from shard 0 to {num_shards - 1}...")
    
    for rank in range(num_shards):
        file_path = os.path.join(jsonl_dir, f"training_ready_rank_{rank}.jsonl")
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: File not found -> {file_path}")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    if "video_id" in data:
                        target_ids.add(str(data["video_id"]))
                except json.JSONDecodeError:
                    pass # Skip corrupted lines
                    
    print(f"✅ Successfully loaded {len(target_ids)} unique video IDs to extract.")
    return target_ids

def fetch_and_stage_videos():
    # 1. Prepare staging directory and target ID list
    os.makedirs(STAGING_DIR, exist_ok=True)
    target_ids = load_target_video_ids(JSONL_DIR, NUM_SHARDS_TO_FETCH)
    
    if not target_ids:
        print("❌ No videos to fetch. Exiting script.")
        return

    # 2. Load dataset
    print(f"📂 Loading dataset from: {DATASET_PATH}...")
    try:
        dataset = load_from_disk(DATASET_PATH)
    except Exception as e:
        print(f"❌ Dataset loading error: {e}")
        return

    total_records = len(dataset)
    found_count = 0
    
    # 3. Scan entire dataset and extract matching videos
    print("🚀 Starting video extraction...")
    pbar = tqdm(total=total_records, desc="Scanning Dataset")
    
    for item in dataset:
        pbar.update(1)
        
        # --- REPLICATE ID EXTRACTION LOGIC FROM YOUR REFERENCE CODE ---
        raw_metadata = item.get('json', {})
        video_id = raw_metadata.get("original_video_filename", "unknown").replace(".mp4", "")
        if video_id == "unknown":
            video_id = raw_metadata.get("youtube_title", "video").replace(" ", "_").lower()
            
        # O(1) membership check
        if video_id in target_ids:
            output_path = os.path.join(STAGING_DIR, f"{video_id}.mp4")
            
            # --- FAULT TOLERANCE (skip if already downloaded) ---
            if os.path.exists(output_path):
                found_count += 1
                if found_count >= len(target_ids): break
                continue
            
            # --- SAVE VIDEO USING STANDARD SCHEMA ('mp4') ---
            video_bytes = item.get('mp4')
            if video_bytes:
                with open(output_path, "wb") as f:
                    f.write(video_bytes)
                found_count += 1
            else:
                print(f"\n⚠️ Error: 'mp4' bytes not found for video {video_id}")
                
            # Early stop when all target videos are collected
            if found_count >= len(target_ids):
                print(f"\n🎯 Success! Extracted all {found_count} target videos.")
                break
                
    pbar.close()
    print(f"🏁 Done! Saved {found_count} videos to '{STAGING_DIR}/'.")

if __name__ == "__main__":
    fetch_and_stage_videos()