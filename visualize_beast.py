import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
import os

# Import class BeastTokenizer từ file bạn vừa chốt
from phase4_beast_tokenizer import BeastTokenizer

def load_real_chunk(video_id):
    """
    Đọc chunk đầu tiên từ file states.jsonl của video và tách lấy tọa độ 3D
    """
    target_file = f"outputs/states/{video_id}_states.jsonl"
    
    if not os.path.exists(target_file):
        raise FileNotFoundError(f"Không tìm thấy file: {target_file}")
        
    with open(target_file, 'r') as f:
        # Đọc dòng đầu tiên (chính là chunk 8 frames đầu tiên)
        first_line = f.readline()
        data = json.loads(first_line)
        
        # Mảng này có shape (8, 153) gồm cả pos, vel, acc
        state_153 = np.array(data["states"], dtype=np.float32)
        
        # Tách lấy 51 biến đầu tiên (Position) và reshape về (8 frames, 17 joints, 3 axes)
        chunk_positions = state_153[:, 0:51].reshape(8, 17, 3)
        
        return chunk_positions

def visualize_real_trajectory(video_id):
    # 1. Load dữ liệu thật từ Phase 3
    print(f"📦 Đang load dữ liệu từ video: {video_id}...")
    chunk_positions = load_real_chunk(video_id)
    
    # Lấy tọa độ của khớp số 16 (Cổ tay phải - Right Wrist) để vẽ
    # Shape lúc này sẽ là (8 frames, 3 axes)
    wrist_raw = chunk_positions[:, 16, :]
    x_raw, y_raw, z_raw = wrist_raw[:, 0], wrist_raw[:, 1], wrist_raw[:, 2]
    t_raw = np.linspace(0, 1, 8)

    # 2. Chạy qua BEAST Tokenizer (Phase 4)
    tokenizer = BeastTokenizer(frames_per_chunk=8)
    
    # Nén thành Token
    encoded = tokenizer.encode_chunk(chunk_positions)
    
    # Giải mã lại thành Control Points (Shape: 4, 17, 3)
    recovered_cp = tokenizer.decode_chunk(encoded) 
    
    # Lấy 4 Control points của riêng khớp cổ tay phải
    cp_wrist = recovered_cp[:, 16, :]
    cp_x, cp_y, cp_z = cp_wrist[:, 0], cp_wrist[:, 1], cp_wrist[:, 2]
    
    # 3. Nội suy đường cong B-spline mượt (50 điểm)
    high_res_t = np.linspace(0, 1, 50)
    spline_x = BSpline(tokenizer.knots, cp_x, k=3)
    spline_y = BSpline(tokenizer.knots, cp_y, k=3)
    spline_z = BSpline(tokenizer.knots, cp_z, k=3)
    
    smooth_x = spline_x(high_res_t)
    smooth_y = spline_y(high_res_t)
    smooth_z = spline_z(high_res_t)

    # ================= VẼ ĐỒ THỊ =================
    fig = plt.figure(figsize=(18, 6))

    # Đồ thị 1: 3D Trajectory
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.plot(x_raw, y_raw, z_raw, 'ro--', label='Raw Points (8 frames)', markersize=6, alpha=0.5)
    ax1.plot(smooth_x, smooth_y, smooth_z, 'b-', label='BEAST B-Spline Curve', linewidth=3)
    ax1.scatter(cp_x, cp_y, cp_z, c='green', s=100, marker='s', label='Control Points (Tokens)')
    
    ax1.set_title(f'Right Wrist 3D Trajectory (Video: {video_id})')
    ax1.legend()

    # Đồ thị 2: 2D Smoothing Check (Trục X)
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(t_raw, x_raw, 'ro--', label='Raw X (Jittery)')
    ax2.plot(high_res_t, smooth_x, 'b-', label='Smooth X (BEAST)', linewidth=2)
    ax2.scatter(np.linspace(0, 1, 4), cp_x, c='green', s=80, marker='s', label='Control Points X', zorder=5)
    
    ax2.set_title('X Coordinate over Time (Jitter Elimination)')
    ax2.set_xlabel('Normalized Time')
    ax2.set_ylabel('Position X')
    ax2.legend()

    # Đồ thị 3: Error (Raw vs Smooth)
    ax3 = fig.add_subplot(1, 3, 3)

    # interpolate smooth_x về đúng 8 điểm để so sánh
    spline_x_interp = spline_x(t_raw)

    error = x_raw - spline_x_interp

    ax3.plot(t_raw, error, 'm-o', label='Error (Raw - Smooth)')
    ax3.axhline(0, color='black', linestyle='--', linewidth=1)

    ax3.set_title('Reconstruction Error (X axis)')
    ax3.set_xlabel('Normalized Time')
    ax3.set_ylabel('Error')
    ax3.legend()

    plt.tight_layout()
    output_img = f'beast_visualization_{video_id}.png'
    plt.savefig(output_img, dpi=300)
    print(f"✅ Đã lưu ảnh visualize vào '{output_img}'")

if __name__ == "__main__":
    video_id = "05xCYOyY1bg"
    visualize_real_trajectory(video_id)