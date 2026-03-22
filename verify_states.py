import os
import glob
import json
import numpy as np

# ================= CẤU HÌNH =================
STATE_DIR = "outputs/states/"

def verify_kinematic_data():
    jsonl_files = glob.glob(os.path.join(STATE_DIR, '*_states.jsonl'))
    total_files = len(jsonl_files)
    
    if total_files == 0:
        print(f"❌ Không tìm thấy file JSONL nào trong {STATE_DIR}")
        return

    print(f"🔍 Đang load và kiểm định {total_files} file states...\n")
    
    all_states = []
    total_windows = 0
    corrupted_files = 0
    
    for idx, file_path in enumerate(jsonl_files, start=1):
        # In tiến độ chớp nháy đè lên dòng cũ
        filename = os.path.basename(file_path)
        print(f"⏳ Đang quét [{idx}/{total_files}] ({idx/total_files*100:.1f}%) : {filename[:20]}...", end='\r')
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    states = np.array(data["states"]) # Shape mong đợi: (8, 156)
                    
                    # 1. KIỂM TRA SHAPE
                    if states.shape != (8, 153):
                        print(f"\n⚠️ Lỗi Shape tại file {file_path}: {states.shape}")
                        continue
                        
                    all_states.append(states)
                    total_windows += 1
        except Exception as e:
            print(f"\n❌ File hỏng {file_path}: {e}")
            corrupted_files += 1

    print(" " * 80, end='\r') # Xóa sạch dòng tiến độ sau khi chạy xong
    print(f"✅ Quét xong {total_files} files! Đang tính toán ma trận tổng...\n")

    if not all_states:
        print("❌ Không có dữ liệu hợp lệ để phân tích.")
        return

    # Gom toàn bộ dữ liệu thành một siêu ma trận: (N_windows, 8, 156)
    mega_tensor = np.stack(all_states)
    
    # Ép phẳng chiều không gian và thời gian để tính thống kê chung: (N_windows * 8, 156)
    flat_data = mega_tensor.reshape(-1, 153)
    
    # 2. KIỂM TRA NaN & INF (Kẻ thù của Deep Learning)
    has_nan = np.isnan(flat_data).any()
    has_inf = np.isinf(flat_data).any()
    
    # Phân rã lại thành 4 nhánh vật lý để kiểm tra độc lập
    pos = flat_data[:, 0:51]
    vel = flat_data[:, 51:102]
    acc = flat_data[:, 102:153]

    print("=" * 60)
    print("🛂 BÁO CÁO KIỂM ĐỊNH KINEMATICS (SANITY CHECK) 🛂")
    print("=" * 60)
    print(f"📦 Tổng số File quét       : {total_files} files (Hỏng: {corrupted_files})")
    print(f"🧱 Tổng số Window 8-frames : {total_windows:,} chunks")
    print(f"🔢 Kích thước Mega Tensor  : {mega_tensor.shape}")
    
    print("\n🚨 KIỂM TRA TÍNH TOÀN VẸN (INTEGRITY):")
    print(f"   -> Chứa NaN (Not a Number)? : {'❌ CÓ' if has_nan else '✅ KHÔNG'}")
    print(f"   -> Chứa Infinity (Vô cực)?  : {'❌ CÓ' if has_inf else '✅ KHÔNG'}")

    print("\n📊 KIỂM TRA PHÂN PHỐI Z-SCORE (MEAN ≈ 0, STD ≈ 1):")
    print(f"   -> Toàn hệ thống | Mean: {flat_data.mean():.4f} | Std: {flat_data.std():.4f}")
    
    print("\n🏃 KIỂM TRA CẬN BIÊN VẬT LÝ (MIN / MAX):")
    print("   (Lưu ý: Vì dữ liệu đã được Z-score hóa, các giá trị này thường nằm trong khoảng [-5, 5] đến [-10, 10])")
    print(f"   -> Position (Tọa độ)  : Min = {pos.min():>7.2f} | Max = {pos.max():>7.2f}")
    print(f"   -> Velocity (Vận tốc) : Min = {vel.min():>7.2f} | Max = {vel.max():>7.2f}")
    print(f"   -> Accel    (Gia tốc) : Min = {acc.min():>7.2f} | Max = {acc.max():>7.2f}")
    print("=" * 60)

    # Đưa ra phán quyết cuối cùng
    if has_nan or has_inf:
        print("❌ KẾT LUẬN: DỮ LIỆU BỊ Ô NHIỄM. KHÔNG THỂ TRAIN VQ-VAE!")
    elif np.max(np.abs(flat_data)) > 50.0:
        print("⚠️ KẾT LUẬN: Dữ liệu có Outlier (giá trị quá lớn). Cần kiểm tra lại khâu Z-score.")
    else:
        print("✅ KẾT LUẬN: DỮ LIỆU SẠCH SẼ HOÀN HẢO. SẴN SÀNG CHO PHASE 2!")

if __name__ == "__main__":
    verify_kinematic_data()