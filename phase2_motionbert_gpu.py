import os
import glob
import subprocess

if __name__ == "__main__":
    json_dir = "outputs/2d_keypoints/"
    video_dir = "videos_staging/"
    output_dir = "outputs/3d_npy/"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. ĐỌC VÀ CHIA BÀI (LOGIC TỪ PHASE 1)
    all_json_files = sorted(glob.glob(os.path.join(json_dir, '*.json')))
    
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    num_tasks = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
    
    json_files = [f for i, f in enumerate(all_json_files) if i % num_tasks == task_id]
    total_files = len(json_files)
    
    # 2. TẠO PHÒNG CÁCH LY (ISOLATED WORKSPACE) ĐỂ CHỐNG RACE CONDITION
    worker_tmp_dir = os.path.join(output_dir, f"worker_{task_id}_tmp")
    os.makedirs(worker_tmp_dir, exist_ok=True)
    
    print(f"\n🚀 [Worker {task_id}/{num_tasks}] Assigned {total_files} JSON files.")
    print("=" * 60)
    
    skipped = 0
    processed = 0
    
    for idx, json_path in enumerate(json_files, start=1):
        video_id = os.path.basename(json_path).split('.')[0]
        video_path = os.path.join(video_dir, f'{video_id}.mp4')
        
        final_npy = os.path.join(output_dir, f'{video_id}.npy')
        final_mp4 = os.path.join(output_dir, f'{video_id}.mp4')
        
        # 3. CƠ CHẾ RESUME
        if os.path.exists(final_npy) and os.path.exists(final_mp4):
            skipped += 1
            if skipped % 5 == 0 or idx == total_files:
                print(f"⏩ [Worker {task_id}] Resumed: {skipped}/{total_files}", end='\r')
            continue
            
        print(f"\n⏳ [{idx}/{total_files}] Lifting 2D to 3D for: {video_id}")
        
        # SỬ DỤNG PHÒNG CÁCH LY CHO OUT_PATH
        cmd = [
            "python", "MotionBERT/infer_wild.py",
            "--config", "MotionBERT/configs/pose3d/MB_ft_h36m.yaml",
            "--evaluate", "MotionBERT/checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin",
            "--json_path", json_path,
            "--vid_path", video_path,
            "--out_path", worker_tmp_dir, # <-- Trỏ về thư mục tạm của riêng worker này
            "--pixel"
        ]
        
        try:
            # Chạy MotionBERT
            subprocess.run(cmd, check=True)
            
            # 4. ATOMIC MOVE TỪ PHÒNG CÁCH LY RA THƯ MỤC CHÍNH
            x3d_npy = os.path.join(worker_tmp_dir, 'X3D.npy')
            x3d_mp4 = os.path.join(worker_tmp_dir, 'X3D.mp4')
            
            if os.path.exists(x3d_npy): os.rename(x3d_npy, final_npy)
            if os.path.exists(x3d_mp4): os.rename(x3d_mp4, final_mp4)
                
            processed += 1
            print(f"✅ Saved 3D data for -> {video_id}")
            
        except Exception as e:
            print(f"❌ Error processing {video_id}: {e}")
            continue

    # Dọn dẹp phòng cách ly sau khi xong việc
    if not os.listdir(worker_tmp_dir):
        os.rmdir(worker_tmp_dir)

    print("\n" + "=" * 60)
    print(f"🎉 WORKER {task_id} COMPLETED! (Processed: {processed}, Resumed: {skipped})")