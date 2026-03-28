import os
from datasets import load_from_disk
from tqdm import tqdm

# ================= CONFIGURATION =================
DATASET_PATH = "/e/scratch/reformo/nguyen38/finevideo_disk"
TARGET_ID = "--5iwqOe8G8"
OUTPUT_DIR = "videos"  # Lưu thẳng ra thư mục hiện tại để xem trên VS Code

def fetch_single_video():
    print(f"📂 Loading dataset from: {DATASET_PATH}...")
    try:
        dataset = load_from_disk(DATASET_PATH)
    except Exception as e:
        print(f"❌ Dataset loading error: {e}")
        return

    total_records = len(dataset)
    
    print(f"🚀 Scanning dataset for video ID: '{TARGET_ID}'...")
    pbar = tqdm(total=total_records, desc="Scanning Dataset")
    
    for item in dataset:
        pbar.update(1)
        
        # --- Lấy ID video từ metadata ---
        raw_metadata = item.get('json', {})
        video_id = raw_metadata.get("original_video_filename", "unknown").replace(".mp4", "")
        
        # --- Tìm kiếm và trích xuất ---
        # Dùng toán tử `in` để bao trọn các trường hợp ID có tiền tố/hậu tố
        if TARGET_ID in video_id:
            output_path = os.path.join(OUTPUT_DIR, f"{TARGET_ID}.mp4")
            
            # Rút trích dữ liệu byte mp4
            video_bytes = item.get('mp4')
            if video_bytes:
                with open(output_path, "wb") as f:
                    f.write(video_bytes)
                print(f"\n🎯 THÀNH CÔNG! Đã lưu video tại: {output_path}")
            else:
                print(f"\n⚠️ Lỗi: Không tìm thấy dữ liệu 'mp4' (byte video) cho {video_id}")
            
            # Tìm thấy mục tiêu thì thoát vòng lặp ngay lập tức
            pbar.close()
            return
            
    pbar.close()
    print(f"\n❌ Quét xong toàn bộ dữ liệu nhưng không tìm thấy ID '{TARGET_ID}'.")

if __name__ == "__main__":
    fetch_single_video()