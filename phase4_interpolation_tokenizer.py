import numpy as np
import json
import os
import glob
import math
from scipy.interpolate import PchipInterpolator

class AdaptiveInterpolationTokenizer:
    """
    Adaptive Interpolation Tokenizer (Robotics-Safe Version)
    - Control points lie EXACTLY on the trajectory.
    - Adaptive sampling based on curvature.
    - Uses PCHIP to prevent dangerous trajectory overshoots.
    """

    def __init__(self, frames_per_chunk=8):
        self.frames_per_chunk = frames_per_chunk

    # ---------------------------
    # 1. Chọn điểm thông minh (Curvature-based với Safety Net)
    # ---------------------------
    def compute_adaptive_indices(self, chunk_positions):
        velocity = np.diff(chunk_positions, axis=0)
        acceleration = np.diff(velocity, axis=0)
        curvature = np.mean(np.linalg.norm(acceleration, axis=2), axis=1)
        
        # Lấy 2 index có độ cong lớn nhất
        top_2_idx = np.argsort(curvature)[-2:] + 1
        
        # SAFETY NET 1: Đảm bảo index nằm gọn trong khoảng [1, 6]
        top_2_idx = np.clip(top_2_idx, 1, 6)
        
        # SAFETY NET 2: Đảm bảo 2 index này không bị trùng nhau
        top_2_idx = np.unique(top_2_idx)
        
        # Nếu robot đứng im (curvature = 0 đều), unique() có thể chỉ trả về 1 phần tử
        # Ta ép thêm một frame bất kỳ (ví dụ frame 3) để luôn đủ 4 control points
        for c in [2, 3, 4, 5]:
            if len(top_2_idx) >= 2: # Đã đủ 2 điểm thì dừng ngay!
                break
            if c not in top_2_idx:
                top_2_idx = np.append(top_2_idx, c)
                
        # Gom và sắp xếp: [0, idx_1, idx_2, 7]
        adaptive_idx = np.sort(np.concatenate(([0], top_2_idx, [7])))
        return adaptive_idx.astype(int)

    # ---------------------------
    # 2. Arc-length (Giữ nguyên)
    # ---------------------------
    def compute_arc_length_param(self, chunk_positions):
        joint_diffs = np.linalg.norm(
            chunk_positions[1:] - chunk_positions[:-1], axis=2
        )
        mean_diffs = np.mean(joint_diffs, axis=1)
        s = np.concatenate([[0], np.cumsum(mean_diffs)])
        if s[-1] > 0:
            s = s / s[-1]
        return s

    # ---------------------------
    # 3. Quantization (Giữ nguyên)
    # ---------------------------
    def quantize(self, x):
        x = np.clip(x, -1.0, 1.0)
        return ((x + 1.0) * 127.5).astype(np.uint8)

    def dequantize(self, x):
        return (x.astype(np.float32) / 127.5) - 1.0

    # ---------------------------
    # 4. ENCODE
    # ---------------------------
    def encode_chunk(self, chunk_positions, time_delta=0.26):
        anchor = chunk_positions[0].copy()
        rel = chunk_positions - anchor
        scale = np.max(np.abs(rel)) + 1e-6
        norm = rel / scale

        t_eval = self.compute_arc_length_param(norm)
        sample_idx = self.compute_adaptive_indices(norm)
        
        cp = norm[sample_idx]           
        t_cp = t_eval[sample_idx]       

        tokens = self.quantize(cp.flatten())

        return {
            "time_delta": float(time_delta),
            "anchor": anchor.tolist(),
            "scale": float(scale),
            "sample_idx": sample_idx.tolist(),
            "t_cp": t_cp.tolist(),
            "tokens": tokens.tolist()
        }

    # ---------------------------
    # 5. DECODE 
    # ---------------------------
    def decode_chunk(self, package):
        tokens = np.array(package["tokens"])
        scale = package["scale"]
        anchor = np.array(package["anchor"])

        cp_norm = self.dequantize(tokens).reshape(4, 17, 3)
        cp = cp_norm * scale + anchor
        return cp

    # ---------------------------
    # 6. RECONSTRUCT (Upgrade lên PCHIP)
    # ---------------------------
    def reconstruct(self, cp, t_cp):
        """
        Dựng lại quỹ đạo bằng PchipInterpolator.
        Đảm bảo robot không bao giờ bị vung tay quá đà (overshoot).
        """
        t = np.linspace(0, 1, 50) # Quỹ đạo độ phân giải cao
        recon = np.zeros((50, 17, 3))

        for j in range(17):
            for d in range(3):
                # Thay CubicSpline bằng PchipInterpolator
                spline = PchipInterpolator(t_cp, cp[:, j, d])
                recon[:, j, d] = spline(t)

        return recon

# ==========================================
# MULTI-PROCESSING & SLURM EXECUTION BLOCK
# ==========================================

def process_state_file(input_path, output_path, tokenizer, stride=16):
    """
    Đọc file states.jsonl, lọc trùng lặp bằng stride, và nén thành token.
    """
    valid_chunks = 0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_path = f"{output_path}.tmp"
    
    try:
        with open(input_path, 'r') as f_in, open(temp_path, 'w') as f_out:
            for line in f_in:
                if not line.strip(): 
                    continue
                    
                data = json.loads(line)
                
                # BỘ LỌC CHỐNG TRÙNG LẶP (Redundancy Filter)
                # Chỉ mã hóa các chunk có window_id chia hết cho 16
                if data["window_id"] % stride != 0:
                    continue
                    
                states = np.array(data["states"], dtype=float)
                
                # Bỏ qua các chunk rỗng/hỏng (đã bị gán NaN ở Phase 3)
                if np.isnan(states).any():
                    continue
                    
                # Encode thành token
                package = tokenizer.encode_chunk(states)
                
                # Ghi ra file
                record = {
                    "video_id": data["video_id"],
                    "window_id": data["window_id"],
                    "package": package
                }
                f_out.write(json.dumps(record) + "\n")
                valid_chunks += 1
                
        # Hoàn thành an toàn thì đổi tên file (Atomic replace)
        if valid_chunks > 0:
            os.replace(temp_path, output_path)
        else:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e
        
    return valid_chunks

if __name__ == "__main__":
    # Cấu hình thư mục (Tùy chỉnh theo cấu trúc của cậu)
    INPUT_DIR = "outputs/states_jsonl"   # Chứa các file *_states.jsonl từ Phase 3
    OUTPUT_DIR = "outputs/agent_tokens"  # Nơi lưu các file *_tokens.jsonl
    
    tokenizer = AdaptiveInterpolationTokenizer(frames_per_chunk=8)

    # 1. NHẬN DIỆN SLURM (Phân bổ tài nguyên CPU)
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', '1'))
    num_tasks = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', '1'))

    # 2. LẤY TOÀN BỘ FILE ĐẦU VÀO TỪ PHASE 3
    state_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_states.jsonl")))
    # Hoặc nếu Phase 3 cậu lưu là state.jsonl chung thì có thể đổi pattern
    if not state_files:
        # Hỗ trợ tìm file state.jsonl nếu test cục bộ 1 video
        single_file = os.path.join(INPUT_DIR, "state.jsonl")
        if os.path.exists(single_file):
            state_files = [single_file]
            
    total_files = len(state_files)

    if total_files == 0:
        print(f"❌ [Worker {task_id}] Không tìm thấy file input nào trong {INPUT_DIR}")
        exit(0)

    # 3. THUẬT TOÁN CHIA ĐỂ TRỊ (Data Slicing)
    chunk_size = math.ceil(total_files / num_tasks)
    start_idx = (task_id - 1) * chunk_size
    end_idx = min(start_idx + chunk_size, total_files)
    my_files = state_files[start_idx:end_idx]

    print(f"🚀 [Worker {task_id}/{num_tasks}] Phân công xử lý {len(my_files)}/{total_files} files.")
    print("=" * 60)

    processed = 0
    skipped = 0
    total_tokens_generated = 0

    # 4. THỰC THI (Có cơ chế Resume)
    for idx, input_path in enumerate(my_files, start=1):
        # Tạo tên file output (ví dụ: videoA_states.jsonl -> videoA_tokens.jsonl)
        base_name = os.path.basename(input_path).replace("_states.jsonl", "").replace("state.jsonl", "tokens")
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_tokens.jsonl")
        
        # Bỏ qua nếu đã xử lý rồi (Resume function)
        if os.path.exists(output_path):
            skipped += 1
            print(f"⏩ [Worker {task_id}] Checked: {idx}/{len(my_files)} (Resumed: {skipped})", end='\r')
            continue
            
        try:
            tokens_count = process_state_file(input_path, output_path, tokenizer, stride=16)
            processed += 1
            total_tokens_generated += tokens_count
            
            progress = (processed + skipped) / len(my_files) * 100
            print(f"✅ [Worker {task_id}] {progress:.1f}% | Processed: {processed} | Tokens: {total_tokens_generated}", end='\r')
            
        except Exception as e:
            print(f"\n❌ Error processing {input_path}: {e}")

    print(f"\n🎉 [Worker {task_id}] HOÀN THÀNH! Đã nén thành công {total_tokens_generated} tokens từ {processed} files (Skipped {skipped}).")