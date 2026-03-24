import json
import sys
import numpy as np

def verify_phase1_json(file_path):
    print(f"🔍 Đang kiểm tra file: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Lỗi đọc file (Không phải chuẩn JSON): {e}")
        return

    # 1. Kiểm tra cấu trúc mảng
    if not isinstance(data, list):
        print("❌ Lỗi: File không phải là một danh sách (list) các frames.")
        return
    
    total_frames = len(data)
    print(f"✅ Tổng số khung hình (frames): {total_frames}")
    if total_frames == 0:
        print("⚠️ Cảnh báo: File rỗng (0 frames).")
        return

    # 2. Kiểm tra cấu trúc của một frame bất kỳ (ví dụ frame đầu tiên)
    sample_frame = data[0]
    expected_keys = {"frame_id", "keypoints"}
    
    print("\n--- Kiểm tra Format Frame 0 ---")
    if not expected_keys.issubset(sample_frame.keys()):
        print(f"❌ Lỗi: Thiếu key bắt buộc. Các keys hiện có: {list(sample_frame.keys())}")
        return
    else:
        print(f"✅ Cấu trúc keys chuẩn: {list(sample_frame.keys())}")

    # 3. Kiểm tra ma trận Keypoints
    kpts = np.array(sample_frame["keypoints"])
    print(f"📐 Shape của keypoints: {kpts.shape}")
    
    if kpts.shape == (17, 3):
        print("✅ Kích thước ma trận chuẩn: 17 khớp (H36M) x 3 chiều (x, y, confidence)")
    else:
        print(f"❌ Kích thước SAI! Yêu cầu (17, 3), thực tế là {kpts.shape}")
        
    # 4. In thử dữ liệu trực quan
    print("\n--- Dữ liệu mẫu (Khớp số 9 - Mũi / Nose) ---")
    nose_kpt = kpts[9]
    print(f"📍 Tọa độ X: {nose_kpt[0]:.2f}")
    print(f"📍 Tọa độ Y: {nose_kpt[1]:.2f}")
    print(f"🎯 Độ tự tin (Confidence): {nose_kpt[2]:.2f} (1.0 = tốt, 0.0 = mất dấu)")
    
    # 5. Kiểm tra an toàn (Missing values)
    zero_conf_count = np.sum(kpts[:, 2] == 0.0)
    if zero_conf_count > 0:
        print(f"⚠️ Lưu ý: Có {zero_conf_count}/17 khớp trong frame này bị mất dấu (confidence = 0.0).")
    else:
        print("✅ Tất cả 17 khớp đều được detect thành công trong frame này.")

if __name__ == "__main__":
    # Thay đường dẫn này bằng 1 file thực tế trong thư mục của bạn
    # Ví dụ chạy lệnh: python check_phase1.py outputs/2d_json/xyz_2d.json
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        print("Vui lòng cung cấp đường dẫn file. Ví dụ: python check_phase1.py outputs/2d_json/sample_2d.json")
        sys.exit(1)
        
    verify_phase1_json(target_file)