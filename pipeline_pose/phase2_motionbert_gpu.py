import os
import json
import subprocess
import argparse
import glob
from datasets import load_from_disk

# ================= CONFIGURATION =================
OUT_2D = "outputs/2d_json"
OUT_3D = "outputs/3d_npy"
WORKSPACE = "workspace_temp"
DATASET_PATH = "/e/scratch/reformo/nguyen38/finevideo_disk"

CONFIG = "MotionBERT/configs/pose3d/MB_ft_h36m.yaml"
CHECKPOINT = "MotionBERT/checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin"

if __name__ == "__main__":
    os.makedirs(OUT_3D, exist_ok=True)
    os.makedirs(WORKSPACE, exist_ok=True)

    # 1. NODE/GPU PARTITIONING LOGIC
    parser = argparse.ArgumentParser()
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--total_workers', type=int, default=160)
    args = parser.parse_known_args()[0]

    local_proc_id = int(os.environ.get('SLURM_PROCID', 0))
    global_task_id = local_proc_id + args.offset
    total_global_tasks = args.total_workers

    task_id = global_task_id

    try:
        with open("cached_video_ids.json", "r") as f:
            all_ids = json.load(f)
    except FileNotFoundError:
        print("❌ Error: cached_video_ids.json not found!")
        exit(1)

    my_ids = set([vid for i, vid in enumerate(all_ids) if i % total_global_tasks == task_id])

    print(f"\n🚀 [Global Worker {task_id}/{total_global_tasks}] Starting 3D lifting for {len(my_ids)} videos...")
    print("=" * 60)

    worker_tmp_dir = os.path.join(WORKSPACE, f"worker_{task_id}_mb_tmp")
    os.makedirs(worker_tmp_dir, exist_ok=True)

    dataset = load_from_disk(DATASET_PATH)
    processed = 0
    skipped = 0

    for item in dataset:
        raw = item.get('json', {})
        vid_id = raw.get("original_video_filename", "unknown").replace(".mp4", "")
        if vid_id == "unknown":
            vid_id = raw.get("youtube_title", "video").replace(" ", "_").lower()

        if vid_id in my_ids:
            json_2d = os.path.join(OUT_2D, f"{vid_id}_2d.json")
            final_npy = os.path.join(OUT_3D, f"{vid_id}.npy")
            final_mp4 = os.path.join(OUT_3D, f"{vid_id}.mp4")

            # RESUME: skip only when both .npy and .mp4 outputs exist
            if (os.path.exists(final_npy) and os.path.exists(final_mp4)) or os.path.exists(f"outputs/final_states/{vid_id}_states.jsonl"):
                print(f"⏩ [Worker {task_id}] Skip (Already finished): {vid_id}")
                skipped += 1
                continue

            if not os.path.exists(json_2d):
                print(f"⏳ [Worker {task_id}] Wait (2D not ready): {vid_id}")
                continue

            video_bytes = item.get('mp4')
            if not video_bytes: continue

            tmp_mp4 = os.path.join(WORKSPACE, f"{vid_id}_worker{task_id}.mp4")

            try:
                with open(tmp_mp4, "wb") as f:
                    f.write(video_bytes)

                # Pin MotionBERT to the correct GPU via CUDA_VISIBLE_DEVICES
                local_id = int(os.environ.get('SLURM_LOCALID', 0))
                my_env = os.environ.copy()
                my_env["CUDA_VISIBLE_DEVICES"] = str(local_id)

                cmd = [
                    "python", "MotionBERT/infer_wild.py",
                    "--config", CONFIG,
                    "--evaluate", CHECKPOINT,
                    "--json_path", json_2d,
                    "--vid_path", tmp_mp4,
                    "--out_path", worker_tmp_dir,
                    "--pixel"
                ]

                subprocess.run(cmd, env=my_env, check=True, stdout=subprocess.DEVNULL)

                # 3. ATOMIC RENAME (both NPY and MP4)
                x3d_npy = os.path.join(worker_tmp_dir, 'X3D.npy')
                x3d_mp4 = os.path.join(worker_tmp_dir, 'X3D.mp4')

                success_npy = False

                # Handle NPY
                if os.path.exists(x3d_npy):
                    os.rename(x3d_npy, final_npy)
                    success_npy = True

                # Handle MP4 (if present)
                if os.path.exists(x3d_mp4):
                    os.rename(x3d_mp4, final_mp4)

                if success_npy:
                    processed += 1
                    print(f"✅ [Worker {task_id}] 3D lifting done: {vid_id}")
                else:
                    print(f"⚠️ [Worker {task_id}] X3D.npy not found for {vid_id}")

            except Exception as e:
                print(f"❌ [Worker {task_id}] Error {vid_id}: {e}")
            finally:
                if os.path.exists(tmp_mp4):
                    os.remove(tmp_mp4)
                for f in glob.glob(os.path.join(worker_tmp_dir, "*")):
                    os.remove(f)

    if not os.listdir(worker_tmp_dir):
        os.rmdir(worker_tmp_dir)

    print("\n" + "=" * 60)
    print(f"🎉 WORKER {task_id} COMPLETED! (Processed: {processed}, Skipped: {skipped})")
