import os
import glob
import numpy as np
from phase4_kinematics_processor import KinematicPreprocessor

def compute_global_stats():
    input_dir = "outputs/3d_npy/"
    npy_files = glob.glob(os.path.join(input_dir, '*.npy'))
    
    processor = KinematicPreprocessor()
    all_raw_states = []
    
    print(f"🔍 Scanning {len(npy_files)} files to compute Global Stats...")
    
    for idx, npy_path in enumerate(npy_files):
        print(f"⏳ Reading [{idx+1}/{len(npy_files)}]...", end='\r')
        pose3d = np.load(npy_path)
        if pose3d.ndim != 3 or pose3d.shape[1:] != (17, 3): continue
            
        # Reproduce preprocessing steps exactly as in the pipeline but WITHOUT normalization
        pose = processor.smooth(pose3d)
        centered, _ = processor.split_root_motion(pose)
        norm_pose = processor.normalize_bone_lengths(centered)
        vel, acc = processor.compute_derivatives(norm_pose)
        
        # Apply physical clipping as recommended
        vel = np.clip(vel, -20.0, 20.0)
        acc = np.clip(acc, -10.0, 10.0)
        
        pos = norm_pose.reshape(len(norm_pose), -1)
        vel = vel.reshape(len(vel), -1)
        acc = acc.reshape(len(acc), -1)
        
        raw_state = np.concatenate([pos, vel, acc], axis=1) # Shape: (N, 153)
        all_raw_states.append(raw_state)
        
    # Concatenate everything into one large matrix
    mega_raw_state = np.concatenate(all_raw_states, axis=0)
    
    global_mean = mega_raw_state.mean(axis=0, keepdims=True)
    global_std = mega_raw_state.std(axis=0, keepdims=True) + 1e-6
    
    # Save to file for permanent reuse
    np.savez("outputs/global_stats.npz", mean=global_mean, std=global_std)
    
    print("\n✅ GLOBAL STATS COMPUTED!")
    print(f"   -> Global Mean shape: {global_mean.shape}")
    print(f"   -> Global Std shape : {global_std.shape}")

if __name__ == "__main__":
    compute_global_stats()