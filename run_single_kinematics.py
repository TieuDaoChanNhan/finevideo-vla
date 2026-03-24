import sys
import os
import json
import numpy as np
from phase3_kinematics_processor import KinematicPreprocessor, process_file # Tái sử dụng code có sẵn

if __name__ == "__main__":
    if len(sys.argv) < 4: sys.exit(1)
    input_npy = sys.argv[1]
    output_jsonl = sys.argv[2]
    video_id = sys.argv[3]
    
    processor = KinematicPreprocessor()
    
    g_mean, g_std = None, None
        
    process_file(input_npy, output_jsonl, processor, video_id, g_mean, g_std)