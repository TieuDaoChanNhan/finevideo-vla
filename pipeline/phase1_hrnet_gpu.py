import cv2
import json
import numpy as np
import torch
import os
import glob

# IMPORT FOR 1.X VÀ 3.X
from mmpose.apis import init_model as init_pose_model, inference_topdown
from mmdet.apis import init_detector, inference_detector
from mmengine.registry import init_default_scope
from datasets import load_from_disk
import argparse

# ================= MODEL CONFIGURATION (FIXED PATHS) =================
pose_config = '../hrnet_storage/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
pose_checkpoint = '../hrnet_storage/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
det_config = '../hrnet_storage/faster-rcnn_r50_fpn_1x_coco.py'
det_checkpoint = '../hrnet_storage/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

local_id = int(os.environ.get('SLURM_LOCALID', 0)) 
device = f'cuda:{local_id}' 

print(f"🚀 Initializing HRNet & Faster R-CNN on {device}...")

print("🚀 Initializing HRNet & Faster R-CNN on GPU H100...")
pose_model = init_pose_model(pose_config, pose_checkpoint, device=device)
det_model = init_detector(det_config, det_checkpoint, device=device)

# ================= COCO -> H36M CONVERSION =================
CONF_THRESHOLD = 0.5

def coco_to_h36m(coco_kpts):
    h36m = np.zeros((17, 3), dtype=np.float32)
    def get_pt(idx):
        x, y, c = coco_kpts[idx]
        if c < CONF_THRESHOLD: return np.array([0.0, 0.0]), 0.0
        return np.array([x, y]), 1.0

    nose, c_nose = get_pt(0); lsho, c_lsho = get_pt(5); rsho, c_rsho = get_pt(6)
    lhip, c_lhip = get_pt(11); rhip, c_rhip = get_pt(12)

    h36m[1,:2], h36m[1,2] = get_pt(12); h36m[2,:2], h36m[2,2] = get_pt(14); h36m[3,:2], h36m[3,2] = get_pt(16)
    h36m[4,:2], h36m[4,2] = get_pt(11); h36m[5,:2], h36m[5,2] = get_pt(13); h36m[6,:2], h36m[6,2] = get_pt(15)
    h36m[11,:2], h36m[11,2] = get_pt(5); h36m[12,:2], h36m[12,2] = get_pt(7); h36m[13,:2], h36m[13,2] = get_pt(9)
    h36m[14,:2], h36m[14,2] = get_pt(6); h36m[15,:2], h36m[15,2] = get_pt(8); h36m[16,:2], h36m[16,2] = get_pt(10)
    h36m[9,:2], h36m[9,2] = nose, c_nose

    if c_lhip > 0 and c_rhip > 0: h36m[0, :2], h36m[0, 2] = (lhip + rhip) / 2.0, 1.0
    if c_lsho > 0 and c_rsho > 0: h36m[8, :2], h36m[8, 2] = (lsho + rsho) / 2.0, 1.0
    if h36m[0, 2] > 0 and h36m[8, 2] > 0: h36m[7, :2], h36m[7, 2] = (h36m[0, :2] + h36m[8, :2]) / 2.0, 1.0
    if c_nose > 0 and h36m[8, 2] > 0: h36m[10, :2], h36m[10, 2] = nose + 0.8 * (nose - h36m[8, :2]), 1.0

    return h36m

def process_video_to_json(video_path, output_json_path):
    cap = cv2.VideoCapture(video_path)
    all_frames_data = []
    frame_idx = 0
    
    while True:
        ok, frame = cap.read()
        if not ok: break
        
        # CHUYỂN CÔNG TẮC 1: Dùng MMDet tìm người
        init_default_scope('mmdet')
        det_result = inference_detector(det_model, frame)
        pred_instances = det_result.pred_instances
        person_mask = (pred_instances.labels == 0) & (pred_instances.scores > 0.5)
        bboxes = pred_instances.bboxes[person_mask].cpu().numpy()
        
        frame_data = {"frame_id": frame_idx, "keypoints": []}
        if len(bboxes) > 0:
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            best_bbox = bboxes[np.argmax(areas)]
            
            # CHUYỂN CÔNG TẮC 2: Dùng MMPose vẽ khung xương
            init_default_scope('mmpose')
            pose_results = inference_topdown(pose_model, frame, bboxes=[best_bbox])
            
            if len(pose_results) > 0:
                kpts = pose_results[0].pred_instances.keypoints[0]
                scores = pose_results[0].pred_instances.keypoint_scores[0]
                coco_final = np.concatenate([kpts, scores[:, None]], axis=1)
                h36m_keypoints = coco_to_h36m(coco_final)
                frame_data["keypoints"] = h36m_keypoints.tolist()
        
        if not frame_data["keypoints"]:
            frame_data["keypoints"] = [[0.0, 0.0, 0.0]] * 17
            
        all_frames_data.append(frame_data)
        frame_idx += 1
        if frame_idx % 1000 == 0:
            print(f"   -> Worker {task_id} | Frame: {frame_idx}", end='\r')
    
    cap.release()
    with open(output_json_path, 'w') as f:
        json.dump(all_frames_data, f)

if __name__ == "__main__":
    # 2. Xử lý và ghi vào file JSON TẠM
    process_video_to_json('../videos/tmp4bo9xir3.mp4', '../outputs/keypoints.json')