import numpy as np
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