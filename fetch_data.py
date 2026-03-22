import os
import json
from datasets import load_from_disk
from tqdm import tqdm

# ================= CONFIGURATION =================
JSONL_DIR = "../prototype/FineVideo-VLA"
DATASET_PATH = "/e/scratch/reformo/nguyen38/finevideo_disk"
STAGING_DIR = "videos_staging"
NUM_SHARDS_TO_FETCH = 6 # Lấy từ rank_0 đến rank_5

def load_target_video_ids(jsonl_dir, num_shards):
    """
    Đọc 6 file JSONL và nạp tất cả video_id vào một Hash Set O(1).
    """
    target_ids = set()
    print(f"🔍 Đang quét các file JSONL từ shard 0 đến {num_shards - 1}...")
    
    for rank in range(num_shards):
        file_path = os.path.join(jsonl_dir, f"training_ready_rank_{rank}.jsonl")
        if not os.path.exists(file_path):
            print(f"⚠️ Cảnh báo: Không tìm thấy file -> {file_path}")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    if "video_id" in data:
                        target_ids.add(str(data["video_id"]))
                except json.JSONDecodeError:
                    pass # Bỏ qua line bị hỏng
                    
    print(f"✅ Đã nạp thành công {len(target_ids)} unique video IDs cần trích xuất.")
    return target_ids

def fetch_and_stage_videos():
    # 1. Chuẩn bị thư mục và danh sách ID mục tiêu
    os.makedirs(STAGING_DIR, exist_ok=True)
    target_ids = load_target_video_ids(JSONL_DIR, NUM_SHARDS_TO_FETCH)
    
    if not target_ids:
        print("❌ Không có video nào cần tải. Dừng script.")
        return

    # 2. Load dataset
    print(f"📂 Đang load dataset từ: {DATASET_PATH}...")
    try:
        dataset = load_from_disk(DATASET_PATH)
    except Exception as e:
        print(f"❌ Lỗi load dataset: {e}")
        return

    total_records = len(dataset)
    found_count = 0
    
    # 3. Quét toàn bộ dataset và bắt đúng các video cần thiết
    print("🚀 Bắt đầu trích xuất video...")
    pbar = tqdm(total=total_records, desc="Quét Dataset")
    
    for item in dataset:
        pbar.update(1)
        
        # --- BẮT CHƯỚC LOGIC EXTRACT ID TỪ CODE THAM KHẢO CỦA BẠN ---
        raw_metadata = item.get('json', {})
        video_id = raw_metadata.get("original_video_filename", "unknown").replace(".mp4", "")
        if video_id == "unknown":
            video_id = raw_metadata.get("youtube_title", "video").replace(" ", "_").lower()
            
        # Kiểm tra siêu tốc O(1)
        if video_id in target_ids:
            output_path = os.path.join(STAGING_DIR, f"{video_id}.mp4")
            
            # --- FAULT TOLERANCE (Bỏ qua nếu đã tải) ---
            if os.path.exists(output_path):
                found_count += 1
                if found_count >= len(target_ids): break
                continue
            
            # --- LƯU VIDEO DỰA THEO SCHEMA CHUẨN ('mp4') ---
            video_bytes = item.get('mp4')
            if video_bytes:
                with open(output_path, "wb") as f:
                    f.write(video_bytes)
                found_count += 1
            else:
                print(f"\n⚠️ Lỗi: Không tìm thấy byte 'mp4' cho video {video_id}")
                
            # Dừng sớm nếu đã tìm đủ số lượng video của cả 6 shard
            if found_count >= len(target_ids):
                print(f"\n🎯 Tuyệt vời! Đã trích xuất đủ {found_count} videos mục tiêu.")
                break
                
    pbar.close()
    print(f"🏁 Hoàn tất! Đã lưu {found_count} videos tại thư mục '{STAGING_DIR}/'.")

if __name__ == "__main__":
    fetch_and_stage_videos()