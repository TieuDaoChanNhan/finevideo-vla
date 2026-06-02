import os
import time
import datetime
import glob
import json

# Import your pipeline library
from pipeline_benchmark import VLADatasetBuilder, RANK, WORLD_SIZE

if __name__ == "__main__":
    # 1. CLEANUP: Xóa kết quả cũ để tránh bị logic Skip làm sai lệch benchmark
    if RANK == 0:
        for f in glob.glob("benchmark_*.jsonl"):
            os.remove(f)
        print(f"🚀 Starting Benchmark across {WORLD_SIZE} GPUs...")

    builder = VLADatasetBuilder(
        base_video_folder="./videos", 
        base_jsonl_folder="./metadata", 
        overlap_threshold=0.2
    )
    
    start_time = time.time()
    
    # Chạy pipeline 
    # Mẹo: Sửa pipeline.py để nhận tham số limit, hoặc chủ động scancel sau 100 video
    builder.process_pipeline(output_base_name="benchmark_run")
    
    end_time = time.time()
    total_seconds = end_time - start_time
    
    # 2. ĐẾM THỰC TẾ: Đếm xem các GPU đã thực sự ghi được bao nhiêu dòng vào JSONL
    if RANK == 0:
        # Đợi một chút để các Rank khác kịp ghi file xong
        time.sleep(5) 
        actual_processed = 0
        for log_file in glob.glob("benchmark_run_rank_*.jsonl"):
            with open(log_file, 'r', encoding='utf-8') as f:
                actual_processed += sum(1 for line in f)
        
        if actual_processed > 0:
            avg_time_per_video = total_seconds / (actual_processed / WORLD_SIZE)
            # 43,751 là tổng dataset FineVideo
            estimated_total_sec = avg_time_per_video * (43751 / WORLD_SIZE)
            
            print("\n" + "="*50)
            print(f"📊 REAL-WORLD BENCHMARK ({actual_processed} Videos Tokenized)")
            print("="*50)
            print(f"⏱️ Total Wall-time         : {datetime.timedelta(seconds=total_seconds)}")
            print(f"⚡ Throughput (All GPUs)   : {actual_processed / total_seconds:.2f} videos/sec")
            print(f"🔮 Est. for 43,751 videos  : {datetime.timedelta(seconds=estimated_total_sec)}")
            print("="*50)