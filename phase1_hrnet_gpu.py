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
pose_config = 'hrnet_storage/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
pose_checkpoint = 'hrnet_storage/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
det_config = 'hrnet_storage/faster-rcnn_r50_fpn_1x_coco.py'
det_checkpoint = 'hrnet_storage/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

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
    OUT_2D = "outputs/2d_json"
    WORKSPACE = "workspace_temp"
    DATASET_PATH = "/e/scratch/reformo/nguyen38/finevideo_disk"
    
    os.makedirs(OUT_2D, exist_ok=True)
    
    # Thêm parser để nhận offset từ lệnh srun
    parser = argparse.ArgumentParser()
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--total_workers', type=int, default=160)
    args = parser.parse_known_args()[0]

    # Tính toán ID toàn cầu dựa trên Offset của từng Job
    local_proc_id = int(os.environ.get('SLURM_PROCID', 0))
    global_task_id = local_proc_id + args.offset
    total_global_tasks = args.total_workers 
    
    task_id = global_task_id # Dùng cái này cho các lệnh print

    # 1. TẠO THƯ MỤC TẠM RIÊNG BIỆT CHO TỪNG WORKER
    worker_tmp_dir = os.path.join(WORKSPACE, f"worker_{task_id}_hrnet_tmp")
    os.makedirs(worker_tmp_dir, exist_ok=True)

    # Dùng File IO an toàn
    try:
        with open("cached_video_ids.json", "r") as f:
            all_ids = json.load(f)
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file cached_video_ids.json!")
        exit(1)

    # Chia bài: Phân đoạn video dựa trên Global Rank
    my_ids = set([vid for i, vid in enumerate(all_ids) if i % total_global_tasks == global_task_id])

    print(f"🚀 [Global Worker {task_id}/{total_global_tasks}] Gánh {len(my_ids)} videos...")
    dataset = load_from_disk(DATASET_PATH)

    for item in dataset:
        raw = item.get('json', {})
        vid_id = raw.get("original_video_filename", "unknown").replace(".mp4", "")
        if vid_id == "unknown": 
            vid_id = raw.get("youtube_title", "video").replace(" ", "_").lower()

        if vid_id in my_ids:
            final_json_2d = os.path.join(OUT_2D, f"{vid_id}_2d.json")
            
            # 2. ĐẶT FILE TẠM VÀO THƯ MỤC RIÊNG CỦA WORKER
            tmp_json_2d = os.path.join(worker_tmp_dir, f"{vid_id}_2d.json.tmp") 
            tmp_mp4 = os.path.join(worker_tmp_dir, f"{vid_id}.mp4")
            
            # CƠ CHẾ RESUME CHI TIẾT
            if os.path.exists(final_json_2d):
                print(f"⏩ [Worker {task_id}] Skip (2D exists): {vid_id}")
                continue
            
            if os.path.exists(f"outputs/3d_npy/{vid_id}.npy"):
                print(f"⏩ [Worker {task_id}] Skip (3D exists): {vid_id}")
                continue
                
            if os.path.exists(f"outputs/final_states/{vid_id}_states.jsonl"):
                print(f"⏩ [Worker {task_id}] Skip (States exist): {vid_id}")
                continue

            video_bytes = item.get('mp4')
            if not video_bytes: continue
            
            try:
                # 1. Giải nén bytes ra mp4 vào thư mục riêng
                with open(tmp_mp4, "wb") as f: 
                    f.write(video_bytes)
                
                # 2. Xử lý và ghi vào file JSON TẠM trong thư mục riêng
                process_video_to_json(tmp_mp4, tmp_json_2d)
                
                # 3. ATOMIC RENAME: Bắn file JSON đã hoàn thiện ra ngoài OUT_2D chung
                if os.path.exists(tmp_json_2d):
                    os.rename(tmp_json_2d, final_json_2d)
                    
                print(f"✅ [Worker {task_id}] Khớp xương 2D thành công: {vid_id}")
                
            except Exception as e:
                print(f"❌ [Worker {task_id}] Lỗi {vid_id}: {e}")
                if os.path.exists(tmp_json_2d): 
                    os.remove(tmp_json_2d) # Xóa file json rác
            finally:
                if os.path.exists(tmp_mp4): 
                    os.remove(tmp_mp4) # Dọn dẹp MP4

    # 3. DỌN DẸP THƯ MỤC TẠM SAU KHI XONG VIỆC
    if not os.listdir(worker_tmp_dir):
        os.rmdir(worker_tmp_dir)