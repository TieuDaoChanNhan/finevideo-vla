import sys
import subprocess

CONFIG = "MotionBERT/configs/pose3d/MB_ft_h36m.yaml"
CHECKPOINT = "MotionBERT/checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin"

if __name__ == "__main__":
    if len(sys.argv) < 4: sys.exit(1)
    json_path = sys.argv[1]
    vid_path = sys.argv[2]
    out_dir = sys.argv[3]
    
    cmd = [
        "python", "MotionBERT/infer_wild.py",
        "--config", CONFIG,
        "--evaluate", CHECKPOINT,
        "--json_path", json_path,
        "--vid_path", vid_path,
        "--out_path", out_dir,
        "--pixel"
    ]
    subprocess.run(cmd, check=True)