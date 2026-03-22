import os
import glob
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class KinematicDataset(Dataset):
    def __init__(self, data_dir="outputs/states/", max_clip_val=15.0):
        """
        Khởi tạo Dataset.
        Đọc toàn bộ file JSONL, ép kiểu sang Float32 (chuẩn của PyTorch) và gom vào RAM.
        """
        self.windows = []
        self.max_clip_val = max_clip_val
        
        jsonl_files = glob.glob(os.path.join(data_dir, '*_states.jsonl'))
        print(f"📦 Đang nạp {len(jsonl_files)} files vào bộ nhớ...")
        
        for file in jsonl_files:
            try:
                with open(file, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        # Ép mảng về dạng Float32 để GPU xử lý nhanh nhất
                        state = np.array(data["states"], dtype=np.float32)
                        
                        # Chỉ lấy những chunk đúng kích thước (8, 153)
                        if state.shape == (8, 153):
                            self.windows.append(state)
            except Exception as e:
                print(f"⚠️ Bỏ qua file lỗi {file}: {e}")
                
        self.total_samples = len(self.windows)
        print(f"✅ Đã nạp thành công {self.total_samples:,} chunks động lực học!")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # 1. Bốc mảng Numpy và chuyển thành PyTorch Tensor
        x = torch.tensor(self.windows[idx])
        
        return x

# ================= CHẠY THỬ (SANITY CHECK CHO DATALOADER) =================
if __name__ == "__main__":
    # Khởi tạo Dataset
    dataset = KinematicDataset(data_dir="outputs/states/")
    
    # Khởi tạo DataLoader: Bốc mỗi lần 256 chunks, xáo trộn ngẫu nhiên (shuffle), 
    # dùng 4 tiến trình con (num_workers) để nạp data song song.
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
    
    # Kéo thử 1 Batch đầu tiên ra xem hình thù thế nào
    for batch_idx, batch in enumerate(dataloader):
        print("\n🚀 BATCH ĐẦU TIÊN ĐÃ VÀO GPU:")
        print(f"   -> Kiểu dữ liệu : {batch.dtype}")
        print(f"   -> Kích thước   : {batch.shape}")
        
        # Kiểm tra cực trị để chắc chắn clamp hoạt động
        print(f"   -> Max value    : {batch.max().item():.2f}")
        print(f"   -> Min value    : {batch.min().item():.2f}")
        break # Chỉ in 1 batch rồi dừng