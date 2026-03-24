import os
import json
import glob
import subprocess
from datasets import load_from_disk

# ================= CONFIGURATION =================
JSONL_DIR = "../prototype/FineVideo-VLA"
DATASET_PATH = "/e/scratch/reformo/nguyen38/finevideo_disk"
WORKSPACE = "workspace_temp"

TASK_ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
NUM_TASKS = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))

os.makedirs(WORKSPACE, exist_ok=True)
os.makedirs("outputs/final_states", exist_ok=True)

def load_target_video_ids(cache_path="cached_video_ids.json"):
    with open(cache_path, "r") as f:
        return json.load(f)

def run_isolated_command(cmd_string):
    """
    Mở một shell Bash thực thụ, chạy chuỗi lệnh (bao gồm cả source env), 
    đợi chạy xong và đóng lại.
    """
    subprocess.run(
        cmd_string, 
        shell=True, 
        executable='/bin/bash', # Bắt buộc dùng bash để hỗ trợ lệnh 'source'
        check=True,
        stdout=subprocess.DEVNULL, # Ẩn log phụ để Terminal đỡ rác
        stderr=subprocess.PIPE     # Giữ lại lỗi để debug nếu crash
    )

def main():
    print(f"🚀 [Worker {TASK_ID}/{NUM_TASKS}] Khởi động Nhạc Trưởng E2E...")
    
    if not os.path.exists("cached_video_ids.json"):
        raise RuntimeError("Missing cached_video_ids.json. Run build script first.")

    all_target_ids = load_target_video_ids()
    if not all_target_ids: return
        
    # Chia bài cho GPU này
    my_target_ids = set([vid for i, vid in enumerate(all_target_ids) if i % NUM_TASKS == TASK_ID])

    print(f"🔄 Assigned xử lí {len(my_target_ids)} videos")
    
    dataset = load_from_disk(DATASET_PATH)
    
    for item in dataset:
        raw_metadata = item.get('json', {})
        video_id = raw_metadata.get("original_video_filename", "unknown").replace(".mp4", "")
        if video_id == "unknown":
            video_id = raw_metadata.get("youtube_title", "video").replace(" ", "_").lower()
            
        if video_id in my_target_ids:
            final_states = f"outputs/final_states/{video_id}_states.jsonl"
            
            if os.path.exists(final_states):
                continue # Đã làm xong từ trước, bỏ qua
                
            print(f"\n🔄 Đang xử lý E2E: {video_id}")
            
            # Khai báo đường dẫn file tạm cho riêng video này
            tmp_mp4 = os.path.join(WORKSPACE, f"{video_id}.mp4")
            tmp_2d_json = os.path.join(WORKSPACE, f"{video_id}_2d.json")
            tmp_3d_npy = os.path.join(WORKSPACE, f"{video_id}.npy") # Giả định MB xuất ra tên này
            
            try:
                # ========================================================
                # CƠ CHẾ RESUME TỪNG BƯỚC (BOTTOM-UP CHECK)
                # ========================================================
                
                # 1. TRÍCH XUẤT VIDEO
                # Chỉ tải MP4 nếu chưa có file 3D NPY, chưa có 2D JSON VÀ chưa có MP4
                if not os.path.exists(tmp_3d_npy) and not os.path.exists(tmp_2d_json) and not os.path.exists(tmp_mp4):
                    video_bytes = item.get('mp4')
                    if not video_bytes:
                        print(f"⚠️ Video {video_id} bị hỏng trên HuggingFace (None). Bỏ qua.")
                        continue
                        
                    with open(tmp_mp4, "wb") as f:
                        f.write(video_bytes)
                    
                # 2. PHASE 1: HRNET (2D)
                # Chỉ chạy nếu chưa có 3D NPY VÀ chưa có 2D JSON
                if not os.path.exists(tmp_3d_npy) and not os.path.exists(tmp_2d_json):
                    print(f"   -> Chạy HRNet (2D)...")
                    cmd_hrnet = f"source setup_hrnet_gpu.sh && python run_single_hrnet.py {tmp_mp4} {tmp_2d_json}"
                    run_isolated_command(cmd_hrnet)
                
                # 3. PHASE 2: MOTIONBERT (3D)
                # Chỉ chạy nếu chưa có 3D NPY
                if not os.path.exists(tmp_3d_npy):
                    print(f"   -> Chạy MotionBERT (3D)...")
                    cmd_mb = f"source setup_motionbert.sh && python run_single_motionbert.py {tmp_2d_json} {tmp_mp4} {WORKSPACE}"
                    run_isolated_command(cmd_mb)
                    
                    # CLEANUP 1: Đã có 3D thì xóa ngay 2D JSON để nhẹ ổ đĩa
                    if os.path.exists(tmp_2d_json): os.remove(tmp_2d_json)
                
                # 4. PHASE 3: KINEMATICS & STATES
                # Chắc chắn chạy vì ở đầu vòng lặp ta đã check final_states chưa tồn tại
                print(f"   -> Chạy Kinematics & States...")
                cmd_kin = f"python run_single_kinematics.py {tmp_3d_npy} {final_states} {video_id}"
                run_isolated_command(cmd_kin)
                
                # CLEANUP 2: Xóa MP4 và 3D NPY sau khi đã ra được final states
                if os.path.exists(tmp_mp4): os.remove(tmp_mp4)
                if os.path.exists(tmp_3d_npy): os.remove(tmp_3d_npy)
                
                print(f"✅ Xong trọn gói: {video_id}")
                
            except subprocess.CalledProcessError as e:
                print(f"❌ Lỗi Subprocess tại {video_id}: {e.stderr.decode('utf-8')}")
                # KHÔNG XÓA FILE TẠM Ở ĐÂY NỮA!
                # Giữ lại các file đã hoàn thành (như .mp4 hoặc _2d.json) 
                # để lần chạy sau orchestrator có thể resume tiếp.

if __name__ == "__main__":
    main()