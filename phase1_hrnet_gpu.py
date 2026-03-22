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

# ================= MODEL CONFIGURATION (FIXED PATHS) =================
pose_config = 'hrnet_storage/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
pose_checkpoint = 'hrnet_storage/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
det_config = 'hrnet_storage/faster-rcnn_r50_fpn_1x_coco.py'
det_checkpoint = 'hrnet_storage/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

device = 'cuda:0'

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

# ================= VIDEO PROCESSING LOOP =================
def process_video_to_json(video_path, output_json_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_frames_data = []
    frame_idx = 0
    detected_count = 0
    
    while True:
        if max_frames and frame_idx >= max_frames: break
        ok, frame = cap.read()
        if not ok: break
        
        init_default_scope('mmdet')
        det_result = inference_detector(det_model, frame)
        pred_instances = det_result.pred_instances
        
        person_mask = (pred_instances.labels == 0) & (pred_instances.scores > 0.5)
        bboxes = pred_instances.bboxes[person_mask].cpu().numpy()
        
        frame_data = {"frame_id": frame_idx, "keypoints": []}
        
        if len(bboxes) > 0:
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            best_bbox = bboxes[np.argmax(areas)]
            
            pose_results = inference_topdown(pose_model, frame, bboxes=[best_bbox])
            
            if len(pose_results) > 0:
                kpts = pose_results[0].pred_instances.keypoints[0]
                scores = pose_results[0].pred_instances.keypoint_scores[0]
                coco_final = np.concatenate([kpts, scores[:, None]], axis=1)
                
                h36m_keypoints = coco_to_h36m(coco_final)
                frame_data["keypoints"] = h36m_keypoints.tolist()
                detected_count += 1
        
        if not frame_data["keypoints"]:
            frame_data["keypoints"] = [[0.0, 0.0, 0.0]] * 17
            
        all_frames_data.append(frame_data)
        frame_idx += 1
        
        if frame_idx % 1000 == 0:
            print(f"   -> Extracted: {frame_idx}/{total_frames if total_frames > 0 else '??'}", end='\r')

    cap.release()
    print() 
    
    success_rate = (detected_count / frame_idx) * 100 if frame_idx > 0 else 0
    print(f"📊 Extraction Success Rate: {success_rate:.2f}% ({detected_count}/{frame_idx} frames)")
    
    with open(output_json_path, 'w') as f:
        json.dump(all_frames_data, f, indent=4)

# ================= BATCH PROCESSING =================
if __name__ == "__main__":
    input_dir = 'videos_staging/'
    output_dir = 'outputs/2d_keypoints/'
    os.makedirs(output_dir, exist_ok=True)
    
    all_video_files = sorted(glob.glob(os.path.join(input_dir, '*.mp4')))[:100]
    
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    num_tasks = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
    
    video_files = [v for i, v in enumerate(all_video_files) if i % num_tasks == task_id]
    
    total_videos = len(video_files)
    print(f"\n🚀 [Worker {task_id}/{num_tasks}] Assigned {total_videos} videos.")
    print(f"🚀 [TieuDaoChanNhan] Resume System Activated.") 
    print("=" * 60)
    
    for idx, video_path in enumerate(video_files, start=1):
        video_id = os.path.basename(video_path).split('.')[0]
        final_output = os.path.join(output_dir, f'{video_id}.json')
        temp_output = os.path.join(output_dir, f'{video_id}.json.tmp')
        
        if os.path.exists(final_output):
            if idx % 5 == 0 or idx == total_videos:
                print(f"⏩ Resumed: {idx}/{total_videos}", end='\r')
            continue
            
        print(f"\n⏳ [{idx}/{total_videos}] Processing: {video_id}")
        
        try:
            process_video_to_json(video_path, temp_output)
            os.rename(temp_output, final_output)
            print(f"✅ Completed: {video_id}")
        except Exception as e:
            print(f"❌ Error {video_id}: {str(e)}")
            if os.path.exists(temp_output): os.remove(temp_output)
            continue
        
    print("\n" + "=" * 60)
    print(f"🎉 BATCH FINISHED!")