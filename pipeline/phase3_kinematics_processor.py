import numpy as np
import json
import os

class KinematicPreprocessor:
    """
    Phase 3 (Version Final + ID Switch Shield): Bộ lọc Tín hiệu, Động học & Không gian linh hoạt trục tọa độ.
    Tích hợp Temporal Smoothing, Stiff Leg Heuristic và Anti-Teleportation.
    """

    def __init__(self, fps=30.0, target_bone_lengths=None, 
                 strict_robotics_filter=False, 
                 vertical_axis=1, y_points_down=False):
        self.fps = fps
        self.dt = 1.0 / fps
        self.pelvis_idx = 0
        self.strict_robotics_filter = strict_robotics_filter
        
        self.vertical_axis = vertical_axis
        self.y_points_down = y_points_down
        
        self.skeleton_tree = [
            (0, 1), (1, 2), (2, 3),
            (0, 4), (4, 5), (5, 6),
            (0, 7), (7, 8), (8, 9),
            (8, 11), (11, 12), (12, 13),
            (8, 14), (14, 15), (15, 16)
        ]

        if target_bone_lengths is None:
            self.target_bone_lengths = {
                (0, 1): 0.2, (1, 2): 0.45, (2, 3): 0.45,
                (0, 4): 0.2, (4, 5): 0.45, (5, 6): 0.45,
                (0, 7): 0.3, (7, 8): 0.3, (8, 9): 0.2,
                (8, 11): 0.2, (11, 12): 0.35, (12, 13): 0.3,
                (8, 14): 0.2, (14, 15): 0.35, (15, 16): 0.3
            }
        else:
            self.target_bone_lengths = target_bone_lengths

    def split_root_motion(self, pose):
        root = pose[:, self.pelvis_idx, :]
        centered = pose - root[:, None, :]
        return centered

    def normalize_bone_lengths(self, pose):
        retargeted = np.zeros_like(pose)
        retargeted[:, self.pelvis_idx] = 0.0  

        for parent, child in self.skeleton_tree:
            bone = pose[:, child] - pose[:, parent]
            length = np.linalg.norm(bone, axis=1, keepdims=True) + 1e-8
            direction = bone / length
            target_len = self.target_bone_lengths[(parent, child)]
            retargeted[:, child] = retargeted[:, parent] + (direction * target_len)

        return retargeted

    def temporal_smooth(self, pose, window_size=5):
        """Dùng Moving Average Window để 'là phẳng' nhiễu tần số cao."""
        smoothed = np.copy(pose)
        pad_size = window_size // 2
        kernel = np.ones(window_size) / window_size
        
        for j in range(pose.shape[1]):
            for c in range(pose.shape[2]):
                signal = pose[:, j, c]
                if np.isnan(signal).all():
                    continue
                
                padded_signal = np.pad(signal, (pad_size, pad_size), mode='edge')
                smoothed_signal = np.convolve(padded_signal, kernel, mode='valid')
                
                valid_mask = ~np.isnan(signal)
                smoothed[valid_mask, j, c] = smoothed_signal[valid_mask]
                
        return smoothed

    def process(self, raw_pose3d):
        if raw_pose3d.ndim != 3 or raw_pose3d.shape[1:] != (17, 3):
            raise ValueError(f"Input must be (N, 17, 3), got {raw_pose3d.shape}")

        pose = raw_pose3d.copy()

        # 1. Mất Root -> Gán NaN toàn bộ khung hình
        mask_zero = np.all(pose == 0.0, axis=-1)
        pose[mask_zero] = np.nan
        frame_missing_root = mask_zero[:, self.pelvis_idx]

        # 1.5. BẮT LỖI DỊCH CHUYỂN TỨC THỜI (ID SWITCH / TELEPORTATION)
        # Bắt lỗi trên tọa độ gốc trước khi bị split_root_motion làm mất thông tin tịnh tiến
        raw_pelvis = pose[:, self.pelvis_idx, :]
        pelvis_jump = np.zeros(len(pose))
        
        # Tính khoảng cách Euclidean của xương chậu giữa 2 frame liên tiếp
        pelvis_jump[1:] = np.linalg.norm(raw_pelvis[1:] - raw_pelvis[:-1], axis=-1)
        
        # Ngưỡng: 0.5 mét / frame. Tương đương di chuyển > 54 km/h -> Lỗi đổi người
        with np.errstate(invalid='ignore'):
            id_switch_mask = pelvis_jump > 0.5
            
        pose[id_switch_mask] = np.nan
        
        # Gộp chung vào một cờ tổng để ngắt chuỗi
        frame_invalid_root = frame_missing_root | id_switch_mask

        # 2. Đưa về gốc tọa độ và NORMALIZE
        centered = self.split_root_motion(pose)
        norm_pose = np.full_like(centered, np.nan)
        valid_root_mask = ~frame_invalid_root
        if np.any(valid_root_mask):
            norm_pose[valid_root_mask] = self.normalize_bone_lengths(centered[valid_root_mask])

        # 3. LÀ PHẲNG TÍN HIỆU (Temporal Smoothing)
        norm_pose = self.temporal_smooth(norm_pose, window_size=5)

        # 4. TÍNH SCALE VÀ ĐỘNG HỌC
        pelvis = norm_pose[:, self.pelvis_idx, :]
        dist_to_pelvis = np.linalg.norm(norm_pose - pelvis[:, None, :], axis=-1)
        with np.errstate(invalid='ignore'):
            avg_bone_span = np.nanmean(dist_to_pelvis, axis=1)
        clip_median_span = np.nanmedian(avg_bone_span)
        if np.isnan(clip_median_span) or clip_median_span == 0:
            return np.full_like(norm_pose, np.nan)

        velocity = np.full_like(norm_pose, np.nan)
        velocity[1:] = (norm_pose[1:] - norm_pose[:-1]) / self.dt
        acceleration = np.full_like(velocity, np.nan)
        acceleration[2:] = (velocity[2:] - velocity[1:-1]) / self.dt
        
        v_norm = np.linalg.norm(velocity, axis=-1) / clip_median_span
        a_norm = np.linalg.norm(acceleration, axis=-1) / clip_median_span
        
        # Bắt lỗi Động học cấp độ Khớp (Ngưỡng an toàn sau khi đã Smooth)
        with np.errstate(invalid='ignore'):
            joint_kinematic_anomaly = (np.nan_to_num(v_norm, nan=0.0) > 80.0) | (np.nan_to_num(a_norm, nan=0.0) > 100.0)
        
        norm_pose[joint_kinematic_anomaly] = np.nan
        frame_kinematic_anomaly = np.any(joint_kinematic_anomaly, axis=1)

        # 5. SPATIAL HEURISTIC (Lọc Không gian -> Sửa thành STIFF LEG)
        left_invalid = np.zeros(len(pose), dtype=bool)
        right_invalid = np.zeros(len(pose), dtype=bool)

        if self.strict_robotics_filter:
            def safe_normalize(v):
                norm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8
                return v / norm

            h = norm_pose[:, :, self.vertical_axis]
            pelvis_h = h[:, self.pelvis_idx]
            margin_y = 0.1 * clip_median_span

            with np.errstate(invalid='ignore'):
                if self.y_points_down:
                    left_foot_up = h[:, 6] < (pelvis_h - margin_y)
                    right_foot_up = h[:, 3] < (pelvis_h - margin_y)
                else:
                    left_foot_up = h[:, 6] > (pelvis_h + margin_y)
                    right_foot_up = h[:, 3] > (pelvis_h + margin_y)
            
            hips_dir = safe_normalize(norm_pose[:, 4, :] - norm_pose[:, 1, :]) 
            r_thigh = safe_normalize(norm_pose[:, 2, :] - norm_pose[:, 1, :])
            r_calf = safe_normalize(norm_pose[:, 3, :] - norm_pose[:, 2, :])
            l_thigh = safe_normalize(norm_pose[:, 5, :] - norm_pose[:, 4, :])
            l_calf = safe_normalize(norm_pose[:, 6, :] - norm_pose[:, 5, :])

            r_knee_axis = np.cross(r_thigh, r_calf)
            l_knee_axis = np.cross(l_thigh, l_calf)

            r_bend_sign = np.sum(r_knee_axis * hips_dir, axis=-1)
            l_bend_sign = np.sum(l_knee_axis * hips_dir, axis=-1)
            with np.errstate(invalid='ignore'):
                right_knee_flamingo = r_bend_sign < -0.1
                left_knee_flamingo = l_bend_sign < -0.1

            r_bend_angle = np.sum(r_thigh * r_calf, axis=-1)
            l_bend_angle = np.sum(l_thigh * l_calf, axis=-1)
            with np.errstate(invalid='ignore'):
                right_knee_scorpion = r_bend_angle < 0.0
                left_knee_scorpion = l_bend_angle < 0.0

            left_invalid = np.nan_to_num(left_foot_up | left_knee_flamingo | left_knee_scorpion, nan=False).astype(bool)
            right_invalid = np.nan_to_num(right_foot_up | right_knee_flamingo | right_knee_scorpion, nan=False).astype(bool)

            spine_down = safe_normalize(norm_pose[:, 0, :] - norm_pose[:, 8, :])

            if np.any(left_invalid):
                thigh_len = self.target_bone_lengths[(4, 5)]
                calf_len = self.target_bone_lengths[(5, 6)]
                ldir = spine_down[left_invalid] 
                norm_pose[left_invalid, 5, :] = norm_pose[left_invalid, 4, :] + ldir * thigh_len
                norm_pose[left_invalid, 6, :] = norm_pose[left_invalid, 5, :] + ldir * calf_len

            if np.any(right_invalid):
                thigh_len = self.target_bone_lengths[(1, 2)]
                calf_len = self.target_bone_lengths[(2, 3)]
                rdir = spine_down[right_invalid]
                norm_pose[right_invalid, 2, :] = norm_pose[right_invalid, 1, :] + rdir * thigh_len
                norm_pose[right_invalid, 3, :] = norm_pose[right_invalid, 2, :] + rdir * calf_len

        norm_pose[frame_invalid_root] = np.nan

        # # 6. DEBUG THỐNG KÊ
        # total_frames = len(pose)
        # print(f"\n📊 DEBUG THỐNG KÊ LỖI (Tổng: {total_frames} frames):")
        # print(f"  - Mất xương chậu (Root Zero): {np.sum(frame_missing_root)}")
        # print(f"  - Lỗi ID Switch (Đổi người): {np.sum(id_switch_mask)}")
        # print(f"  - ĐÃ LÀM PHẲNG LỖI ĐỘNG HỌC (Giật/Lag Khớp): {np.sum(frame_kinematic_anomaly)} frames")
        # print(f"  - ĐÃ FIX CỨNG CHÂN (Stiff Leg): Trái ({np.sum(left_invalid)}), Phải ({np.sum(right_invalid)})")
        # if self.strict_robotics_filter:
        #     print(f"    + Chi tiết Trái: Lộn ngược ({np.sum(left_foot_up)}), Flamingo ({np.sum(left_knee_flamingo)}), Scorpion ({np.sum(left_knee_scorpion)})")
        #     print(f"    + Chi tiết Phải: Lộn ngược ({np.sum(right_foot_up)}), Flamingo ({np.sum(right_knee_flamingo)}), Scorpion ({np.sum(right_knee_scorpion)})")
        # print(f"  -> TỔNG KHUNG HÌNH BỊ CHÉM: {np.sum(frame_invalid_root)} / {total_frames}\n")

        return norm_pose

# -------------------------
# UTILITY FUNCTIONS 
# -------------------------
def interpolate_nan_gaps(pose, max_gap=5):
    N = pose.shape[0]
    pose_flat = pose.reshape(N, -1)
    interpolated = pose_flat.copy()
    
    for j in range(pose_flat.shape[1]):
        y = pose_flat[:, j]
        nans = np.isnan(y)
        if not nans.any() or nans.all():
            continue
            
        x = np.arange(N)
        y_interp = np.interp(x, x[~nans], y[~nans])
        
        is_nan = np.concatenate(([0], nans.astype(int), [0]))
        diff = np.diff(is_nan)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for start, end in zip(starts, ends):
            gap_size = end - start
            if gap_size <= max_gap:
                interpolated[start:end, j] = y_interp[start:end]
                
    return interpolated.reshape(pose.shape)

def create_windows(state, window_size=8, stride=1):
    if len(state) < window_size:
        return [], []
    
    windows = []
    valid_indices = []
    for i in range(0, len(state) - window_size + 1, stride):
        window = state[i:i + window_size]
        if not np.isnan(window).any():
            windows.append(window)
            valid_indices.append(i)
            
    if not windows:
        return [], []
        
    return np.stack(windows), valid_indices

def to_safe_json_list(arr):
    arr_obj = arr.astype(object)
    arr_obj[np.isnan(arr)] = None
    return arr_obj.tolist()

def process_file(input_path, output_path, processor, video_id):
    pose3d = np.load(input_path)
    if pose3d.ndim != 3 or pose3d.shape[1:] != (17, 3):
        return False

    pose = processor.process(pose3d)
    pose_interp = interpolate_nan_gaps(pose, max_gap=5)
    
    # Nhận 2 biến trả về
    windows, valid_indices = create_windows(pose_interp, window_size=8)
    
    if len(windows) == 0:
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        # Dùng zip để ghép window với id thời gian thực của nó
        for w, true_frame_id in zip(windows, valid_indices):
            record = {
                "video_id": video_id,
                "window_id": int(true_frame_id), # LƯU ID THẬT, KHÔNG DÙNG ENUMERATE NỮA!
                "states": to_safe_json_list(w)
            }
            f.write(json.dumps(record, allow_nan=False) + "\n")
            
    return True

if __name__ == "__main__":
    processor = KinematicPreprocessor(
        fps=30.0, 
        strict_robotics_filter=True, 
        vertical_axis=1, 
        y_points_down=True
    )

    final_jsonl = "../outputs/state.jsonl"
    temp_jsonl = f"{final_jsonl}.tmp" 
    npy_path = "../outputs/X3D.npy"
    video_id = 'martial_art'
    
    try:
        success = process_file(npy_path, temp_jsonl, processor, video_id)
        if success:
            os.replace(temp_jsonl, final_jsonl)
            print(f"✅ THÀNH CÔNG! Đã lưu JSONL: {final_jsonl}")
        else:
            if os.path.exists(temp_jsonl):
                os.remove(temp_jsonl)
            print(f"⚠️ Video bị skip.")
    except Exception as e:
        print(f"❌ Lỗi khi xử lý {video_id}: {e}")
        if os.path.exists(temp_jsonl):
            os.remove(temp_jsonl)