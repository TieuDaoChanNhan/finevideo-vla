import os
import sys
import torch

# 1. TRỎ CACHE VỀ ĐÚNG Ổ SCRATCH (Giống hệt cấu hình lúc sau bạn chạy Offline)
SCRATCH_CACHE_DIR = "/p/scratch/laionize/nguyen38/hf_cache"
os.environ["HF_HOME"] = SCRATCH_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = SCRATCH_CACHE_DIR

# 2. ĐẢM BẢO ĐANG BẬT MẠNG (Xóa cờ offline nếu đang kẹt trong môi trường)
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.pop("HF_DATASETS_OFFLINE", None)

print("🌐 ĐANG KẾT NỐI INTERNET: Kéo toàn bộ 'ruột gan' của Seed2 về Scratch...")

# 3. Import và khởi tạo Seed2 để ép nó tải data
sys.path.append("./seed2")
try:
    from seed2_tokenizer import Seed2Tokenizer
    
    # Lệnh này sẽ tự động tải MỌI THỨ còn thiếu (weights, configs, sub-models) 
    # và nhét thẳng vào thư mục SCRATCH_CACHE_DIR
    tokenizer = Seed2Tokenizer.from_pretrained("./seed2", torch_dtype=torch.float16)
    
    print("✅ THÀNH CÔNG! Toàn bộ Cache của Seed2 đã nằm gọn trong ổ Scratch.")
    print("💡 Giờ bạn có thể mang sang Booster Node ngắt mạng và chạy benchmark_pipeline.py!")
    
except Exception as e:
    print(f"❌ Lỗi: {e}")