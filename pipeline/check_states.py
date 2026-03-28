import json
import numpy as np

def verify_states(jsonl_path="../outputs/state.jsonl"):
    print(f"🔍 Đang kiểm tra file: {jsonl_path}...")
    
    try:
        with open(jsonl_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("❌ Không tìm thấy file JSONL.")
        return

    if not lines:
        print("❌ File rỗng!")
        return

    print(f"📂 Tổng số Windows (Clip 8-frames) trích xuất được: {len(lines)}")

    # Gom toàn bộ data để tính toán
    all_windows = []
    for line in lines:
        data = json.loads(line)
        all_windows.append(data["states"])

    # Chuyển thành Numpy. Các chữ `null` trong JSON sẽ tự động biến thành `np.nan`
    arr = np.array(all_windows, dtype=float) 
    
    # --- BÁO CÁO KẾT QUẢ ---
    print("\n" + "="*50)
    print("📊 BÁO CÁO CHẤT LƯỢNG DATA SAU PHASE 3")
    print("="*50)
    
    print(f"✅ Định dạng mảng (N, 8, 17, 3) : {arr.shape}")
    
    # Kiểm tra Null
    total_frames = arr.shape[0] * arr.shape[1]
    # Ta chỉ cần check khớp 0 (pelvis) có nan không là biết cả frame đó có nan không
    null_frames = np.sum(np.isnan(arr[:, :, 0, 0])) 
    print(f"👻 Số khung hình bị xóa (Null)   : {null_frames} / {total_frames} ({(null_frames/total_frames)*100:.2f}%)")
    
    # Kiểm tra phép chuẩn hóa (Scale)
    if null_frames < total_frames:
        min_val = np.nanmin(arr)
        max_val = np.nanmax(arr)
        print(f"📏 Phạm vi không gian (Min/Max) : {min_val:.4f} -> {max_val:.4f}")
        
        if -2.0 <= min_val and max_val <= 2.0:
            print("   ✨ CHUẨN: Tọa độ đã được ép về hệ quy chiếu nhỏ (xấp xỉ mét). Không còn pixel khổng lồ!")
        else:
            print("   ⚠️ CẢNH BÁO: Tọa độ vẫn còn quá lớn, hàm normalize có vấn đề!")
            
    print("="*50)

if __name__ == "__main__":
    verify_states()