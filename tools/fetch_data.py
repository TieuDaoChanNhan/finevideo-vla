import os
from datasets import load_from_disk
from tqdm import tqdm

# ================= CONFIGURATION =================
DATASET_PATH = "/e/scratch/reformo/nguyen38/finevideo_disk"
OUTPUT_DIR = "videos_staging"

def fetch_all_videos():
    print(f"📂 Loading dataset from: {DATASET_PATH}...")
    try:
        dataset = load_from_disk(DATASET_PATH)
    except Exception as e:
        print(f"❌ Dataset loading error: {e}")
        return

    total_records = len(dataset)
    print(f"🚀 Scanning and extracting {total_records} videos...")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    success_count = 0
    skip_count = 0
    error_count = 0

    # Iterate over the full dataset
    for item in tqdm(dataset, desc="Extracting Videos", unit="video"):
        # --- Extract video ID from metadata ---
        raw_metadata = item.get('json', {})
        video_id = raw_metadata.get("original_video_filename", "unknown").replace(".mp4", "")

        # Handle missing filename
        if video_id == "unknown":
            video_id = f"unknown_video_{success_count + skip_count + error_count}"

        output_path = os.path.join(OUTPUT_DIR, f"{video_id}.mp4")

        # Skip if already downloaded
        if os.path.exists(output_path):
            skip_count += 1
            continue

        # --- Extract and save MP4 bytes ---
        video_bytes = item.get('mp4')
        if video_bytes:
            try:
                with open(output_path, "wb") as f:
                    f.write(video_bytes)
                success_count += 1
            except Exception as e:
                # Catch disk-full or I/O errors
                error_count += 1
        else:
            error_count += 1

    print("\n" + "="*50)
    print("🎉 EXTRACTION COMPLETE!")
    print(f"✅ Successfully extracted : {success_count} videos")
    print(f"⏩ Skipped (already exist): {skip_count} videos")
    print(f"❌ Errors / no MP4 bytes  : {error_count} videos")
    print("="*50)

if __name__ == "__main__":
    fetch_all_videos()
