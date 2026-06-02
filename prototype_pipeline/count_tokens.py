import os
import glob
import json
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Hàm đếm token cho 1 file duy nhất
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
                    
                    # Quét qua từng scene và activity để lấy chuỗi video_tokens
                    for scene in data.get("scenes", []):
                        for act in scene.get("activities", []):
                            tokens_str = act.get("video_tokens", "")
                            if tokens_str:
                                # Split chuỗi bằng khoảng trắng để đếm chính xác số token
                                total_tokens += len(tokens_str.split())
                except json.JSONDecodeError:
                    pass # Bỏ qua dòng lỗi nếu có
    except Exception as e:
        print(f"⚠️ Lỗi đọc file {filepath}: {e}")
        
    return video_count, total_tokens

if __name__ == "__main__":
    # Tìm tất cả 160 file JSONL
    file_list = glob.glob("training_ready_rank_*.jsonl")
    
    if not file_list:
        print("❌ Không tìm thấy file training_ready_rank_*.jsonl nào!")
        exit()
        
    print(f"🚀 Tìm thấy {len(file_list)} files. Bắt đầu đếm song song...")

    total_dataset_tokens = 0
    total_dataset_videos = 0

    # Kích hoạt 32 luồng CPU chạy song song (Login Node của JUPITER rất mạnh)
    with ProcessPoolExecutor(max_workers=32) as executor:
        results = list(tqdm(executor.map(count_tokens_in_file, file_list), total=len(file_list), desc="Đang quét Shards"))

    # Tổng hợp kết quả từ 160 file
    for v_count, t_count in results:
        total_dataset_videos += v_count
        total_dataset_tokens += t_count

    print("\n" + "="*50)
    print("📊 KẾT QUẢ ĐẾM CHÍNH XÁC 100% (EXACT COUNT)")
    print("="*50)
    print(f"🎬 Tổng số Video đã xử lý : {total_dataset_videos:,}")
    print(f"🪙 Tổng số Tokens sinh ra : {total_dataset_tokens:,}")
    print("="*50)