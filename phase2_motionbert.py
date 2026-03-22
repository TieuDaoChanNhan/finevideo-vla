import os
import glob
import subprocess

if __name__ == "__main__":
    json_dir = "outputs/2d_keypoints/"
    video_dir = "videos_staging/"
    output_dir = "outputs/3d_npy/"
    
    os.makedirs(output_dir, exist_ok=True)
    
    json_files = glob.glob(os.path.join(json_dir, '*.json'))
    total_files = len(json_files)
    
    print(f"\n🚀 Found {total_files} JSON files to process with MotionBERT.")
    print("=" * 60)
    
    for idx, json_path in enumerate(json_files, start=1):
        video_id = os.path.basename(json_path).split('.')[0]
        video_path = os.path.join(video_dir, f'{video_id}.mp4')
        
        # Định nghĩa tên file đích cho cả NPY và MP4
        final_npy = os.path.join(output_dir, f'{video_id}.npy')
        final_mp4 = os.path.join(output_dir, f'{video_id}.mp4')
        
        # Fault Tolerance: Bỏ qua nếu CẢ HAI file đã tồn tại
        if os.path.exists(final_npy) and os.path.exists(final_mp4):
            print(f"⏩ [{idx}/{total_files}] Skipping {video_id} (already generated).")
            continue
            
        print(f"⏳ [{idx}/{total_files}] Lifting 2D to 3D for video: {video_id}")
        
        cmd = [
            "python", "MotionBERT/infer_wild.py",
            "--config", "MotionBERT/configs/pose3d/MB_ft_h36m.yaml",
            "--evaluate", "MotionBERT/checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin",
            "--json_path", json_path,
            "--vid_path", video_path,
            "--out_path", output_dir, 
            "--pixel"
        ]
        
        # Chạy MotionBERT
        subprocess.run(cmd, check=True)
        
        # Xử lý đổi tên file X3D.npy một cách an toàn
        x3d_npy = os.path.join(output_dir, 'X3D.npy')
        if os.path.exists(x3d_npy):
            os.rename(x3d_npy, final_npy)
            
        # Xử lý đổi tên file X3D.mp4 một cách an toàn
        x3d_mp4 = os.path.join(output_dir, 'X3D.mp4')
        if os.path.exists(x3d_mp4):
            os.rename(x3d_mp4, final_mp4)
            
        print(f"✅ Saved 3D array -> {final_npy}")
        print(f"✅ Saved 3D video -> {final_mp4}\n")

    print("🎉 PHASE 2 COMPLETED! ALL 3D POSES READY.")