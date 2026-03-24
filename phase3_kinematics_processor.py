import numpy as np
import json
import os
from scipy.ndimage import gaussian_filter1d
import glob


class KinematicPreprocessor:
    """
    Phase 1: Convert raw MotionBERT 3D keypoints into a clean,
    scale-invariant, temporally-consistent kinematic representation.

    Pipeline:
        raw pose (N,17,3)
        → smoothing
        → root centering
        → bone normalization (canonical skeleton)
        → velocity & acceleration
        → facing direction (stable)
        → flatten to state vector
    """

    def __init__(self, target_bone_lengths=None, fps=30.0, smooth_sigma=1.0):
        self.fps = fps
        self.dt = 1.0 / fps
        self.smooth_sigma = smooth_sigma

        # Key joints (H36M assumption)
        self.pelvis_idx = 0
        self.neck_idx = 8
        self.l_shoulder_idx = 11
        self.r_shoulder_idx = 14

        # Skeleton tree (parent → child)
        self.skeleton_tree = [
            (0, 1), (1, 2), (2, 3),
            (0, 4), (4, 5), (5, 6),
            (0, 7), (7, 8), (8, 9),
            (8, 11), (11, 12), (12, 13),
            (8, 14), (14, 15), (15, 16)
        ]

        # Canonical skeleton (scale normalization)
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

    # -------------------------
    # 1. SMOOTHING
    # -------------------------
    def smooth(self, pose):
        """Reduce temporal noise from pose estimation."""
        return gaussian_filter1d(pose, sigma=self.smooth_sigma, axis=0)

    # -------------------------
    # 2. ROOT SEPARATION
    # -------------------------
    def split_root_motion(self, pose):
        """Separate global root motion from local pose."""
        root = pose[:, self.pelvis_idx, :]
        centered = pose - root[:, None, :]
        return centered, root

    # -------------------------
    # 3. BONE NORMALIZATION
    # -------------------------
    def normalize_bone_lengths(self, pose):
        """
        Retarget skeleton to canonical proportions.
        Removes scale differences across subjects.
        """
        retargeted = np.zeros_like(pose)
        retargeted[:, self.pelvis_idx] = 0.0  # root

        for parent, child in self.skeleton_tree:
            bone = pose[:, child] - pose[:, parent]

            length = np.linalg.norm(bone, axis=1, keepdims=True) + 1e-8
            direction = bone / length

            target_len = self.target_bone_lengths[(parent, child)]
            new_vec = direction * target_len

            retargeted[:, child] = retargeted[:, parent] + new_vec

        return retargeted

    # -------------------------
    # 5. DERIVATIVES
    # -------------------------
    def compute_derivatives(self, x):
        """Compute velocity and acceleration with smoothing."""
        v = np.gradient(x, axis=0) / self.dt
        v = gaussian_filter1d(v, sigma=1.0, axis=0)

        a = np.gradient(v, axis=0) / self.dt
        a = gaussian_filter1d(a, sigma=1.0, axis=0)

        return v, a

    # -------------------------
    # 6. MAIN PIPELINE
    # -------------------------
    def process(self, raw_pose3d, global_mean=None, global_std=None):
        """Full preprocessing pipeline (Positions Only)."""

        pose = self.smooth(raw_pose3d)
        centered, root = self.split_root_motion(pose)
        norm_pose = self.normalize_bone_lengths(centered)
        
        # Chỉ lấy vị trí, làm phẳng (flatten) thành mảng 51 chiều (17 khớp * 3)
        state = norm_pose.reshape(len(norm_pose), -1)

        # 🚀 IMPORTANT CHANGE: Use global stats if provided,
        # otherwise skip normalization (used during global stats computation)
        if global_mean is not None and global_std is not None:
            # Lưu ý: global_mean và global_std bây giờ cũng phải là mảng 51 chiều!
            state = (state - global_mean) / global_std

        return state, norm_pose


# -------------------------
# 7. TEMPORAL WINDOWING
# -------------------------
def create_windows(state, window_size=8, stride=1):
    """Convert frame-wise states → temporal windows."""
    windows = []
    for i in range(0, len(state) - window_size + 1, stride):
        windows.append(state[i:i + window_size])
    return np.stack(windows)


# -------------------------
# 8. FILE PROCESSING
# -------------------------
def process_file(input_path, output_path, processor, video_id, global_mean, global_std):
    pose3d = np.load(input_path)
    if pose3d.ndim != 3 or pose3d.shape[1:] != (17, 3):
        raise ValueError(f"Invalid shape")

    # Pass global stats here
    state, pose = processor.process(pose3d, global_mean, global_std)
    windows = create_windows(state, window_size=8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for i, w in enumerate(windows):
            record = {
                "video_id": video_id,  # Add ID for later traceability to original video
                "window_id": i,
                "states": w.tolist()
            }
            f.write(json.dumps(record) + "\n")

# -------------------------
# 9. ENTRY POINT (BATCH PROCESSING)
# -------------------------
if __name__ == "__main__":
    processor = KinematicPreprocessor()

    input_dir = "outputs/3d_npy/"
    output_dir = "outputs/states/"
    os.makedirs(output_dir, exist_ok=True)

    # 1. TASK DISTRIBUTION USING "L'X" STYLE (MODULO N)
    all_npy_files = sorted(glob.glob(os.path.join(input_dir, '*.npy')))
    
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    num_tasks = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
    
    # [FIXED HERE] Assign file list to current worker
    npy_files = [f for i, f in enumerate(all_npy_files) if i % num_tasks == task_id]
    total_files = len(npy_files)

    print(f"\n🚀 [Worker {task_id}/{num_tasks}] Processing {total_files} files.")
    print(f"🚀 [TieuDaoChanNhan] Kinematics Resume System Activated.")
    print("=" * 60)

    processed = 0
    skipped = 0
    
    stats = np.load("outputs/global_stats.npz")
    global_mean = stats['mean']
    global_std = stats['std']

    for idx, npy_path in enumerate(npy_files, start=1):
        video_id = os.path.basename(npy_path).split('.')[0]
        final_jsonl = os.path.join(output_dir, f"{video_id}_states.jsonl")
        temp_jsonl = f"{final_jsonl}.tmp"

        # 2. RESUME MECHANISM: check final output file
        if os.path.exists(final_jsonl):
            skipped += 1
            if skipped % 5 == 0 or idx == total_files:
                print(f"⏩ [Worker {task_id}] Checked: {idx}/{total_files} (Resumed: {skipped})", end='\r')
            continue

        print(f"\n⏳ [{idx}/{total_files}] Processing kinematics: {video_id}")
        
        try:
            # 3. WRITE TEMP FILE (ATOMIC WRITE)
            process_file(npy_path, temp_jsonl, processor, video_id, global_mean, global_std)
            
            # 4. FINALIZE FILE (RENAME)
            os.rename(temp_jsonl, final_jsonl)
            processed += 1
            print(f"✅ Saved -> {final_jsonl}")
            
        except Exception as e:
            print(f"❌ Error processing {video_id}: {e}")
            if os.path.exists(temp_jsonl):
                os.remove(temp_jsonl) # Remove corrupted temp file if error occurs
            continue

    print("\n" + "=" * 60)
    print(f"🎉 WORKER {task_id} FINISHED! (Processed: {processed}, Skipped: {skipped})")