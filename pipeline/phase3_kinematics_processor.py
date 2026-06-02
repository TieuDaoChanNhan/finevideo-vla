import numpy as np
import json
import os
import glob
import argparse

class KinematicPreprocessor:
    """
    Phase 3 (Version Final + ID Switch Shield): Signal, Kinematics & Spatial filter with flexible coordinate axis.
    Integrates Temporal Smoothing, Stiff Leg Heuristic, and Anti-Teleportation.
    """

    def __init__(self, fps=30.0, target_bone_lengths=None,
                 strict_robotics_filter=True,
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

    def detect_hallucinations(self, pose_3d):
        if pose_3d.ndim != 3 or pose_3d.shape[1:] != (17, 3):
            raise ValueError(f"Input must be (N, 17, 3), got {pose_3d.shape}")

        pose = np.asarray(pose_3d, dtype=np.float64)
        all_nan = np.isnan(pose).all(axis=(1, 2))

        # 1. GEOMETRIC PROPORTIONS
        right_shoulder = pose[:, 14, :]
        left_shoulder = pose[:, 11, :]
        pelvis = pose[:, 0, :]
        thorax = pose[:, 8, :]

        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder, axis=1)
        torso_length = np.linalg.norm(pelvis - thorax, axis=1) + 1e-6

        ratio = shoulder_width / torso_length
        proportion_filter = (ratio < 0.1) | (ratio > 3.0)

        # 2. ROGUE JOINTS FILTER
        parent_idx = np.array([p for p, c in self.skeleton_tree], dtype=np.int64)
        child_idx = np.array([c for p, c in self.skeleton_tree], dtype=np.int64)

        bones = pose[:, child_idx, :] - pose[:, parent_idx, :]
        bone_lengths = np.linalg.norm(bones, axis=2)
        
        # Safely suppress All-NaN slice warning
        max_bone_length = np.zeros(len(pose))
        valid_bones = ~np.isnan(bone_lengths).all(axis=1)
        if np.any(valid_bones):
            max_bone_length[valid_bones] = np.nanmax(bone_lengths[valid_bones], axis=1)

        rogue_joint_filter = max_bone_length > (2.5 * torso_length)

        # 3. CORE COLLAPSE FILTER
        other_joints = pose[:, 1:, :]
        dist_to_pelvis = np.linalg.norm(other_joints - pelvis[:, None, :], axis=2)
        
        median_dist_to_pelvis = np.zeros(len(pose))
        valid_dists = ~np.isnan(dist_to_pelvis).all(axis=1)
        if np.any(valid_dists):
            median_dist_to_pelvis[valid_dists] = np.nanmedian(dist_to_pelvis[valid_dists], axis=1)

        core_collapse_filter = median_dist_to_pelvis < (0.15 * torso_length)

        hallucination_mask = (
            all_nan |
            np.nan_to_num(proportion_filter, nan=False).astype(bool) |
            np.nan_to_num(rogue_joint_filter, nan=False).astype(bool) |
            np.nan_to_num(core_collapse_filter, nan=False).astype(bool)
        )

        return hallucination_mask.astype(bool)

    def temporal_smooth(self, pose, window_size=5):
        """Apply a Moving Average Window to smooth high-frequency noise, NaN-safe."""
        smoothed = np.copy(pose)
        pad_size = window_size // 2
        kernel = np.ones(window_size)

        for j in range(pose.shape[1]):
            for c in range(pose.shape[2]):
                signal = pose[:, j, c]
                valid_mask = ~np.isnan(signal)
                if not valid_mask.any():
                    continue

                # 1. Replace NaN with 0 so convolution is not contaminated
                signal_zeroed = np.where(valid_mask, signal, 0.0)
                padded_signal = np.pad(signal_zeroed, (pad_size, pad_size), mode='edge')
                
                # 2. Build a mask counting how many frames in the window are actually valid (1.0 or 0.0)
                padded_mask = np.pad(valid_mask.astype(float), (pad_size, pad_size), mode='edge')

                # 3. Compute sum of signal and sum of valid frames
                sum_signal = np.convolve(padded_signal, kernel, mode='valid')
                sum_mask = np.convolve(padded_mask, kernel, mode='valid')

                # 4. Divide by valid count (not always 5) to get the true mean
                with np.errstate(invalid='ignore'):
                    smoothed_signal = sum_signal / sum_mask

                # 5. Write back only to valid positions
                smoothed[valid_mask, j, c] = smoothed_signal[valid_mask]

        return smoothed

    def process(self, raw_pose3d):
        if raw_pose3d.ndim != 3 or raw_pose3d.shape[1:] != (17, 3):
            raise ValueError(f"Input must be (N, 17, 3), got {raw_pose3d.shape}")

        pose = raw_pose3d.copy()
        geometric_hallucination_mask = self.detect_hallucinations(pose)
        pose[geometric_hallucination_mask] = np.nan

        # 1. Missing root -> mark entire frame as NaN
        pelvis_zero = np.all(pose[:, self.pelvis_idx, :] == 0.0, axis=-1)
        pose[pelvis_zero] = np.nan
        frame_missing_root = pelvis_zero

        # 1.5. DETECT INSTANT DISPLACEMENT ERRORS (ID SWITCH / TELEPORTATION)
        # Detect on raw coordinates before split_root_motion discards translation info
        raw_pelvis = pose[:, self.pelvis_idx, :]
        pelvis_jump = np.zeros(len(pose))

        # Compute Euclidean distance of the pelvis between consecutive frames
        pelvis_jump[1:] = np.linalg.norm(raw_pelvis[1:] - raw_pelvis[:-1], axis=-1)

        # Threshold: 0.5 m/frame, equivalent to >54 km/h -> identity switch error
        with np.errstate(invalid='ignore'):
            id_switch_mask = pelvis_jump > 0.5

        pose[id_switch_mask] = np.nan

        # Combine into a single flag to break the sequence
        frame_invalid_root = frame_missing_root | id_switch_mask

        # 2. Center at origin and NORMALIZE
        centered = self.split_root_motion(pose)
        norm_pose = np.full_like(centered, np.nan)
        valid_root_mask = ~frame_invalid_root
        if np.any(valid_root_mask):
            norm_pose[valid_root_mask] = self.normalize_bone_lengths(centered[valid_root_mask])

        # 3. SMOOTH SIGNAL (Temporal Smoothing)
        norm_pose = self.temporal_smooth(norm_pose, window_size=5)

        # 4. COMPUTE SCALE AND KINEMATICS
        pelvis = norm_pose[:, self.pelvis_idx, :]
        dist_to_pelvis = np.linalg.norm(norm_pose - pelvis[:, None, :], axis=-1)
        with np.errstate(invalid='ignore'):
            avg_bone_span = np.nanmean(dist_to_pelvis, axis=1)
        clip_median_span = np.nanmedian(avg_bone_span)
        if np.isnan(clip_median_span) or clip_median_span == 0:
            print(f"🪓 Geometric Hallucination Filter: {np.sum(geometric_hallucination_mask)} / {len(pose)} frames removed.")
            return np.full_like(norm_pose, np.nan)

        velocity = np.full_like(norm_pose, np.nan)
        velocity[1:] = (norm_pose[1:] - norm_pose[:-1]) / self.dt
        acceleration = np.full_like(velocity, np.nan)
        acceleration[2:] = (velocity[2:] - velocity[1:-1]) / self.dt

        v_norm = np.linalg.norm(velocity, axis=-1) / clip_median_span
        a_norm = np.linalg.norm(acceleration, axis=-1) / clip_median_span

        # Detect per-joint kinematics anomalies (safe threshold after smoothing)
        with np.errstate(invalid='ignore'):
            joint_kinematic_anomaly = (np.nan_to_num(v_norm, nan=0.0) > 150.0) | \
                                      (np.nan_to_num(a_norm, nan=0.0) > 250.0)

        norm_pose[joint_kinematic_anomaly] = np.nan
        frame_kinematic_anomaly = np.any(joint_kinematic_anomaly, axis=1)

        # 5. SPATIAL HEURISTIC (Spatial filter -> correct to STIFF LEG)
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

        # # 6. DEBUG STATISTICS
        # total_frames = len(pose)
        # print(f"\n📊 DEBUG ERROR STATISTICS (Total: {total_frames} frames):")
        # print(f"  - Missing pelvis (Root Zero): {np.sum(frame_missing_root)}")
        # print(f"  - ID Switch error (person change): {np.sum(id_switch_mask)}")
        # print(f"  - SMOOTHED KINEMATIC ERRORS (Joint jitter/lag): {np.sum(frame_kinematic_anomaly)} frames")
        # print(f"  - STIFF LEG APPLIED: Left ({np.sum(left_invalid)}), Right ({np.sum(right_invalid)})")
        # if self.strict_robotics_filter:
        #     print(f"    + Left detail: Inverted ({np.sum(left_foot_up)}), Flamingo ({np.sum(left_knee_flamingo)}), Scorpion ({np.sum(left_knee_scorpion)})")
        #     print(f"    + Right detail: Inverted ({np.sum(right_foot_up)}), Flamingo ({np.sum(right_knee_flamingo)}), Scorpion ({np.sum(right_knee_scorpion)})")
        # print(f"  -> TOTAL FRAMES REMOVED: {np.sum(frame_invalid_root)} / {total_frames}\n")

        print(f"🪓 Geometric Hallucination Filter: {np.sum(geometric_hallucination_mask)} / {len(pose)} frames removed.")

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

def forward_fill_joints(window):
    out = window.copy()
    T, J, C = out.shape
    for j in range(J):
        for c in range(C):
            col = out[:, j, c]
            valid = ~np.isnan(col)
            if not valid.any():
                continue
            first = np.where(valid)[0][0]
            col[:first] = col[first]
            for t in range(first + 1, T):
                if np.isnan(col[t]):
                    col[t] = col[t - 1]
            out[:, j, c] = col
    return out

def create_windows(state, window_size=8, stride=1):
    if len(state) < window_size:
        return [], []

    windows = []
    valid_indices = []
    for i in range(0, len(state) - window_size + 1, stride):
        window = state[i:i + window_size]
        if not np.isnan(window[:, 0, :]).any():
            window_clean = forward_fill_joints(window)
            windows.append(window_clean)
            valid_indices.append(i)

    if not windows:
        return [], []
    return np.stack(windows), valid_indices

def to_safe_json_list(arr):
    arr_obj = arr.astype(object)
    arr_obj[np.isnan(arr)] = None
    return arr_obj.tolist()

def apply_2d_mask(pose3d, json_2d_path):
    if pose3d.ndim != 3 or pose3d.shape[1:] != (17, 3):
        raise ValueError(f"Input must be (N, 17, 3), got {pose3d.shape}")

    if not os.path.exists(json_2d_path):
        raise FileNotFoundError(f"2D json file not found: {json_2d_path}")

    with open(json_2d_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pose2d_list = []
    for item in data:
        pose2d_list.append(item["keypoints"])
    pose2d = np.asarray(pose2d_list, dtype=np.float64)

    num_frames = min(len(pose3d), len(pose2d))
    masked_pose3d = pose3d.astype(np.float64, copy=True)

    # 2D Gate: Translation dictionary COCO (HRNet) -> Human3.6M (MotionBERT)
    coco_to_h36m = {
        0: 9,   # Nose
        5: 11,  # L_Shoulder
        6: 14,  # R_Shoulder
        7: 12,  # L_Elbow
        8: 15,  # R_Elbow
        9: 13,  # L_Wrist
        10: 16, # R_Wrist
        11: 4,  # L_Hip
        12: 1,  # R_Hip
        13: 5,  # L_Knee
        14: 2,  # R_Knee
        15: 6,  # L_Ankle
        16: 3   # R_Ankle
    }

    zero_joint_mask = np.all(pose2d[:num_frames] == 0.0, axis=-1)

    # Only mark NaN on the corresponding joints; leave interpolated joints (e.g. Pelvis) untouched
    for coco_idx, h36m_idx in coco_to_h36m.items():
        missing_mask = zero_joint_mask[:, coco_idx]
        masked_pose3d[:num_frames, h36m_idx, :][missing_mask] = np.nan

    return masked_pose3d

def process_file(input_path, output_path, processor, video_id, json_2d_dir):
    pose3d = np.load(input_path)
    if pose3d.ndim != 3 or pose3d.shape[1:] != (17, 3):
        return False

    json_2d_path = os.path.join(json_2d_dir, f"{video_id}_2d.json")
    
    # STEP 1: Fill natural tracking gaps in the RAW ARRAY before anything else
    pose3d_interp = interpolate_nan_gaps(pose3d, max_gap=5)
    
    # STEP 2: Apply 2D Gate mask (nullify occluded limbs)
    pose3d_masked = apply_2d_mask(pose3d_interp, json_2d_path)

    # STEP 3: Run through Bone Normalizer & Kinematics
    pose_final = processor.process(pose3d_masked)

    # STEP 4: Extract windows safely
    windows, valid_indices = create_windows(pose_final, window_size=8)

    if len(windows) == 0:
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for w, true_frame_id in zip(windows, valid_indices):
            record = {
                "window_id": int(true_frame_id),
                "states": to_safe_json_list(w)
            }
            f.write(json.dumps(record, allow_nan=False) + "\n")

    return True

# -------------------------
# ENTRY POINT (SLURM ARRAY)
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process 3D pose .npy files into state windows for large-scale SLURM jobs."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing input .npy files.")
    parser.add_argument("--output-dir", required=True, help="Directory to write output .jsonl files.")
    parser.add_argument(
        "--json-2d-dir",
        required=True,
        help="Directory containing 2D pose json files named as <video_id>_2d.json.",
    )
    parser.add_argument(
        "--fps-json",
        default=None,
        help="Path to fps_lookup.json from tools/extract_fps.py. "
             "If provided, per-video fps is used for kinematics instead of fixed 30.",
    )
    args = parser.parse_args()

    fps_lookup = {}
    if args.fps_json:
        import json as _json
        with open(args.fps_json) as _f:
            fps_lookup = {k: v for k, v in _json.load(_f).items() if v}

    # processor is now created per-video inside the loop (see below)
    default_processor = KinematicPreprocessor(fps=30.0)

    input_dir = args.input_dir
    output_dir = args.output_dir
    json_2d_dir = args.json_2d_dir
    os.makedirs(output_dir, exist_ok=True)

    all_npy_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    npy_files = [f for i, f in enumerate(all_npy_files) if i % num_tasks == task_id]
    total_files = len(npy_files)

    print(f"\n🚀 [Worker {task_id}/{num_tasks}] Processing {total_files} files.")
    print("=" * 60)

    processed = 0
    skipped = 0

    for idx, npy_path in enumerate(npy_files, start=1):
        video_id = os.path.basename(npy_path).split(".")[0]
        final_jsonl = os.path.join(output_dir, f"{video_id}_states.jsonl")
        temp_jsonl = f"{final_jsonl}.tmp"

        if os.path.exists(final_jsonl):
            skipped += 1
            print(f"⏩ [Worker {task_id}] Checked: {idx}/{total_files} (Resumed: {skipped})", end="\r")
            continue

        try:
            video_fps = fps_lookup.get(video_id, 30.0) if fps_lookup else 30.0
            processor = (
                KinematicPreprocessor(fps=video_fps)
                if video_fps != 30.0
                else default_processor
            )
            success = process_file(
                npy_path,
                temp_jsonl,
                processor,
                video_id,
                json_2d_dir=json_2d_dir,
            )

            if success:
                os.replace(temp_jsonl, final_jsonl)
                processed += 1
                progress = (processed + skipped) / total_files * 100 if total_files > 0 else 100.0
                print(f"✅ [Worker {task_id}] {progress:.2f}% | Processed: {processed} | Skipped: {skipped}")
            else:
                if os.path.exists(temp_jsonl):
                    os.remove(temp_jsonl)

        except Exception as e:
            print(f"❌ Error processing {video_id}: {e}")
            if os.path.exists(temp_jsonl):
                os.remove(temp_jsonl)
            continue

    print("\n" + "=" * 60)
    print(f"🎉 WORKER {task_id} FINISHED! (Processed: {processed}, Skipped: {skipped})")
