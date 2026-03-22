import cv2
import json
import numpy as np
from mmpose.apis import (inference_top_down_pose_model, init_pose_model, process_mmdet_results)
from mmdet.apis import inference_detector, init_detector

# ================= MODEL CONFIGURATION =================
# Trỏ trực tiếp vào các file .py bạn vừa dùng wget tải về
pose_config = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'

det_config = 'faster-rcnn_r50_fpn_1x_coco.py'
det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

device = 'cpu' # Change to 'cuda:0' when running on HPC/JUPITER

print("Initializing HRNet & Faster R-CNN models...")
pose_model = init_pose_model(pose_config, pose_checkpoint, device=device)
det_model = init_detector(det_config, det_checkpoint, device=device)


# ================= COCO -> H36M CONVERSION FUNCTION =================
CONF_THRESHOLD = 0.5

def coco_to_h36m(coco_kpts):
    """
    coco_kpts: Numpy array shape (17, 3) containing [x, y, confidence]
    Returns: H36M array shape (17, 3) using MotionBERT standard coordinate format.
    """
    h36m = np.zeros((17, 3), dtype=np.float32)

    # Internal helper function to safely retrieve keypoints
    def get_pt(idx):
        x, y, c = coco_kpts[idx]
        if c < CONF_THRESHOLD:
            return np.array([0.0, 0.0]), 0.0
        return np.array([x, y]), 1.0

    # Extract original COCO joints
    nose, c_nose = get_pt(0)
    lsho, c_lsho = get_pt(5)
    rsho, c_rsho = get_pt(6)
    lhip, c_lhip = get_pt(11)
    rhip, c_rhip = get_pt(12)

    # --- DIRECTLY MAPPED JOINTS ---
    # Right leg
    h36m[1, :2], h36m[1, 2] = get_pt(12)  # RHip
    h36m[2, :2], h36m[2, 2] = get_pt(14)  # RKnee
    h36m[3, :2], h36m[3, 2] = get_pt(16)  # RAnkle

    # Left leg
    h36m[4, :2], h36m[4, 2] = get_pt(11)  # LHip
    h36m[5, :2], h36m[5, 2] = get_pt(13)  # LKnee
    h36m[6, :2], h36m[6, 2] = get_pt(15)  # LAnkle

    # Left arm
    h36m[11, :2], h36m[11, 2] = get_pt(5) # LShoulder
    h36m[12, :2], h36m[12, 2] = get_pt(7) # LElbow
    h36m[13, :2], h36m[13, 2] = get_pt(9) # LWrist

    # Right arm
    h36m[14, :2], h36m[14, 2] = get_pt(6) # RShoulder
    h36m[15, :2], h36m[15, 2] = get_pt(8) # RElbow
    h36m[16, :2], h36m[16, 2] = get_pt(10)# RWrist

    # Nose
    h36m[9, :2], h36m[9, 2] = nose, c_nose

    # --- VECTOR-BASED COMPUTATION FOR DERIVED JOINTS ---
    # Pelvis (midpoint of left and right hips)
    if c_lhip > 0 and c_rhip > 0:
        pelvis = (lhip + rhip) / 2.0
        h36m[0, :2], h36m[0, 2] = pelvis, 1.0
    else:
        pelvis = np.array([0.0, 0.0])
        h36m[0, 2] = 0.0 # Occluded

    # Neck (midpoint of shoulders)
    if c_lsho > 0 and c_rsho > 0:
        neck = (lsho + rsho) / 2.0
        h36m[8, :2], h36m[8, 2] = neck, 1.0
    else:
        neck = np.array([0.0, 0.0])
        h36m[8, 2] = 0.0

    # Torso (spine - midpoint between pelvis and neck)
    if h36m[0, 2] > 0 and h36m[8, 2] > 0:
        h36m[7, :2] = (pelvis + neck) / 2.0
        h36m[7, 2] = 1.0

    # Head Top (extrapolated from neck toward nose)
    if c_nose > 0 and h36m[8, 2] > 0:
        vec_neck_to_nose = nose - neck
        h36m[10, :2] = nose + 0.8 * vec_neck_to_nose
        h36m[10, 2] = 1.0

    return h36m


# ================= VIDEO PROCESSING LOOP =================
def process_video_to_json(video_path, output_json_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    # Lấy tổng số frame để in tiến độ (nếu video hỗ trợ đọc property này)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    all_frames_data = []
    frame_idx = 0
    
    while True:
        if max_frames and frame_idx >= max_frames:
            break
            
        ok, frame = cap.read()
        if not ok:
            break
            
        rgb_frame = frame[:, :, ::-1]
        
        # 1. Detect human bounding boxes
        mmdet_results = inference_detector(det_model, rgb_frame)
        person_results = process_mmdet_results(mmdet_results, cat_id=1)
        
        # 2. Extract COCO 2D Pose
        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            rgb_frame,
            person_results,
            bbox_thr=0.3,
            format='xyxy',
            dataset=pose_model.cfg.data.test.type
        )
        
        frame_data = {"frame_id": frame_idx, "keypoints": []}
        
        if len(pose_results) > 0:
            # Select the most confident detected person
            best_person = max(pose_results, key=lambda x: x['bbox'][4])
            coco_keypoints = np.array(best_person['keypoints'])
            
            # 3. CONVERT TO DEFAULT H36M FORMAT FOR MOTIONBERT
            h36m_keypoints = coco_to_h36m(coco_keypoints)
            frame_data["keypoints"] = h36m_keypoints.tolist()
        else:
            frame_data["keypoints"] = [[0.0, 0.0, 0.0]] * 17
            
        all_frames_data.append(frame_data)
        frame_idx += 1
        
        # In tiến độ frame trên cùng 1 dòng (tránh spam console)
        if frame_idx % 10 == 0:
            print(f"   -> Extracted frames: {frame_idx}/{total_frames if total_frames > 0 else 'Unknown'}", end='\r')

    cap.release()
    print() # Xuống dòng sau khi in đè xong
    
    # Save JSON output
    with open(output_json_path, 'w') as f:
        json.dump(all_frames_data, f, indent=4)

import os
import glob

if __name__ == "__main__":
    input_dir = 'videos_staging/'
    output_dir = 'outputs/2d_keypoints/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all mp4 files
    video_files = glob.glob(os.path.join(input_dir, '*.mp4'))[:100]
    total_videos = len(video_files)
    
    print(f"\n🚀 Found {total_videos} videos in '{input_dir}' to process.")
    print("=" * 60)
    
    for idx, video_path in enumerate(video_files, start=1):
        video_id = os.path.basename(video_path).split('.')[0]
        output_file = os.path.join(output_dir, f'{video_id}.json')
        
        # Fault Tolerance: Skip if already processed
        if os.path.exists(output_file):
            print(f"⏩ [{idx}/{total_videos}] Skipping {video_id} (already extracted).")
            continue
            
        print(f"⏳ [{idx}/{total_videos}] Processing video: {video_id}")
        process_video_to_json(video_path, output_file)
        print(f"✅ [{idx}/{total_videos}] Saved to: {output_file}\n")
        
    print("🎉 ALL VIDEOS PROCESSED SUCCESSFULLY!")