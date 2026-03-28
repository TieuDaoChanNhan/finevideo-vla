import json
import os
import glob
import numpy as np
from collections import defaultdict

# ================= CONFIGURATION =================
# Giả định Phase 4 sinh ra nhiều file token, gom chung vào một thư mục
INPUT_DIR = "outputs/agent_tokens"  
OUTPUT_DIR = "outputs/clean_pose_dataset"
LOG_DIR = "outputs/logs"

# Điều kiện giữ lại video
MIN_TOKENS_PER_VIDEO = 3       # Ít nhất phải có 3 tokens (tương đương ~1 giây)
MIN_YIELD_RATE = 0.4           # Tỷ lệ sống sót của frames phải > 40%

def process_file(input_path, f_out, f_log):
    """Đọc 1 file token, gom nhóm theo video, lọc và ghi kết quả."""
    video_data = defaultdict(list)
    
    with open(input_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            video_id = data.get("video_id", "unknown")
            video_data[video_id].append(data)
            
    clean_count = 0
    discarded_count = 0
    
    for video_id, chunks in video_data.items():
        # Sắp xếp token theo thời gian tuyệt đối
        chunks.sort(key=lambda x: x["window_id"])
        
        num_tokens = len(chunks)
        if num_tokens == 0:
            continue
            
        # Toán học tính Yield Rate
        min_id = chunks[0]["window_id"]
        max_id = chunks[-1]["window_id"]
        # +1 vì token đầu tiên cũng tính là 1 đoạn dữ liệu
        expected_tokens = ((max_id - min_id) // 16) + 1 
        
        yield_rate = num_tokens / expected_tokens if expected_tokens > 0 else 0
        
        # KIỂM DUYỆT (MACRO-FILTER)
        if num_tokens >= MIN_TOKENS_PER_VIDEO and yield_rate >= MIN_YIELD_RATE:
            # Video NGON -> Ghi vào file Clean của riêng Worker này
            for chunk in chunks:
                f_out.write(json.dumps(chunk) + "\n")
            clean_count += 1
        else:
            # Video NÁT -> Ghi vào file Log của riêng Worker này
            f_log.write(f"File: {os.path.basename(input_path)} | Video: {video_id} | Tokens: {num_tokens}/{expected_tokens} (Yield: {yield_rate*100:.1f}%)\n")
            discarded_count += 1
            
    return clean_count, discarded_count

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 1. NHẬN DIỆN SLURM (Phân bổ tài nguyên CPU)
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', '1'))
    num_tasks = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', '1'))

    # 2. LẤY TOÀN BỘ FILE ĐẦU VÀO
    # Cậu có thể sửa "*.jsonl" thành pattern khớp với cách Phase 4 lưu file
    token_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jsonl")))
    total_files = len(token_files)

    if total_files == 0:
        print(f"❌ [Worker {task_id}] Không tìm thấy file JSONL nào trong {INPUT_DIR}")
        exit(0)

    # 3. THUẬT TOÁN CHIA ĐỂ TRỊ (Data Slicing)
    chunk_size = int(np.ceil(total_files / num_tasks))
    start_idx = (task_id - 1) * chunk_size
    end_idx = min(start_idx + chunk_size, total_files)
    my_files = token_files[start_idx:end_idx]

    print(f"🚀 [Worker {task_id}/{num_tasks}] Phân công xử lý {len(my_files)}/{total_files} files.")

    # 4. GHI CỤC BỘ (Tránh Race Condition)
    out_clean_path = os.path.join(OUTPUT_DIR, f"clean_dataset_part_{task_id:04d}.jsonl")
    out_log_path = os.path.join(LOG_DIR, f"flagged_part_{task_id:04d}.txt")

    total_clean = 0
    total_discarded = 0

    with open(out_clean_path, 'w') as f_out, open(out_log_path, 'w') as f_log:
        for idx, file_path in enumerate(my_files, 1):
            clean, discarded = process_file(file_path, f_out, f_log)
            total_clean += clean
            total_discarded += discarded
            
            # In tiến độ không trôi log
            if idx % 10 == 0 or idx == len(my_files):
                progress = (idx / len(my_files)) * 100
                print(f"   ⏳ [Worker {task_id}] Tiến độ: {progress:.1f}% | Sạch: {total_clean} | Bỏ: {total_discarded}", end='\r')

    print(f"\n✅ [Worker {task_id}] HOÀN THÀNH! Video đạt chuẩn: {total_clean} | Bị loại: {total_discarded}")
    print(f"   📁 Dữ liệu sạch: {out_clean_path}")
    print(f"   📋 Log lỗi    : {out_log_path}")