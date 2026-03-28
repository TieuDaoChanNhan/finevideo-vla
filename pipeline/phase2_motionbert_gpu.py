import os
import json
import subprocess
import argparse
import glob
from datasets import load_from_disk

# ================= CONFIGURATION =================

CONFIG = "../MotionBERT/configs/pose3d/MB_ft_h36m.yaml"
CHECKPOINT = "../MotionBERT/checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin"

if __name__ == "__main__":
    local_id = int(os.environ.get('SLURM_LOCALID', 0))
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = str(local_id)
    # ----------------------------------------------------

    cmd = [
        "python", "../MotionBERT/infer_wild.py",
        "--config", CONFIG,
        "--evaluate", CHECKPOINT,
        "--json_path", '../outputs/keypoints.json',
        "--vid_path", '../videos/martial_art.mp4',
        "--out_path", '../outputs',
        "--pixel"
    ]
    
    # CHÚ Ý THÊM env=my_env VÀO ĐÂY
    subprocess.run(cmd, env=my_env, check=True, stdout=subprocess.DEVNULL)