import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import os

# Import the new adaptive tokenizer
from phase4_interpolation_tokenizer import AdaptiveInterpolationTokenizer

def load_real_chunk(video_id):
    """
    Load the first 8-frame chunk from the processed states JSONL file.
    """
    target_file = f"outputs/states/{video_id}_states.jsonl"
    
    if not os.path.exists(target_file):
        raise FileNotFoundError(f"File not found: {target_file}")
        
    with open(target_file, 'r') as f:
        first_line = f.readline()
        data = json.loads(first_line)
        
        # State vector contains 153 variables; extract the first 51 (Positions)
        state_153 = np.array(data["states"], dtype=np.float32)
        chunk_positions = state_153[:, 0:51].reshape(8, 17, 3)
        
        return chunk_positions

def visualize_real_trajectory(video_id):
    print(f"📦 Loading data for video: {video_id}...")
    chunk_positions = load_real_chunk(video_id)
    
    # Extract Right Wrist (Index 16) for visualization
    wrist_raw = chunk_positions[:, 16, :]
    x_raw, y_raw, z_raw = wrist_raw[:, 0], wrist_raw[:, 1], wrist_raw[:, 2]

    # 1. Initialize the Adaptive Tokenizer
    tokenizer = AdaptiveInterpolationTokenizer(frames_per_chunk=8)
    
    # 2. Encode to get tokens, sample indices, and parameterized time (t_cp)
    encoded = tokenizer.encode_chunk(chunk_positions)
    sample_idx = encoded["sample_idx"]
    t_cp = encoded["t_cp"]
    
    print(f"🎯 Adaptive Indices chosen: {sample_idx}")
    
    # 3. Decode back to 3D Control Points
    recovered_cp = tokenizer.decode_chunk(encoded) 
    
    # Extract the 4 control points specifically for the Right Wrist
    cp_wrist = recovered_cp[:, 16, :]
    cp_x, cp_y, cp_z = cp_wrist[:, 0], cp_wrist[:, 1], cp_wrist[:, 2]
    
    # 4. Reconstruct the full timeline to calculate exact errors
    # We need the full arc-length parameterized time array (8 frames)
    anchor = chunk_positions[0].copy()
    rel = chunk_positions - anchor
    scale = np.max(np.abs(rel)) + 1e-6
    norm = rel / scale
    t_eval_full = tokenizer.compute_arc_length_param(norm)
    
    # 5. Interpolate using PCHIP (Monotonic & Safe)
    high_res_t = np.linspace(0, 1, 50)
    spline_x = PchipInterpolator(t_cp, cp_x)
    spline_y = PchipInterpolator(t_cp, cp_y)
    spline_z = PchipInterpolator(t_cp, cp_z)
    
    smooth_x = spline_x(high_res_t)
    smooth_y = spline_y(high_res_t)
    smooth_z = spline_z(high_res_t)

    # ================= VISUALIZATION PLOTS =================
    fig = plt.figure(figsize=(18, 6))

    # Plot 1: 3D Trajectory
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.plot(x_raw, y_raw, z_raw, 'ro--', label='Raw Frames (8 points)', markersize=6, alpha=0.5)
    ax1.plot(smooth_x, smooth_y, smooth_z, 'b-', label='PCHIP Trajectory', linewidth=3)
    ax1.scatter(cp_x, cp_y, cp_z, c='green', s=100, marker='s', label='Adaptive Tokens', zorder=5)
    
    ax1.set_title(f'Right Wrist 3D - Adaptive PCHIP ({video_id})')
    ax1.legend()

    # Plot 2: 2D Monotonicity Check (X-axis)
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(t_eval_full, x_raw, 'ro--', label='Raw X (Jittery)', alpha=0.5)
    ax2.plot(high_res_t, smooth_x, 'b-', label='Smooth X (PCHIP)', linewidth=2)
    ax2.scatter(t_cp, cp_x, c='green', s=80, marker='s', label='Tokens (On curve)', zorder=5)
    
    ax2.set_title('X Coordinate - No Overshoot Guarantee')
    ax2.set_xlabel('Arc-length Parameterized Time')
    ax2.set_ylabel('Position X')
    ax2.legend()

    # Plot 3: Reconstruction Error 
    ax3 = fig.add_subplot(1, 3, 3)
    
    # Evaluate PCHIP exactly at the 8 original frame timestamps
    smooth_x_at_raw = spline_x(t_eval_full)
    error = x_raw - smooth_x_at_raw

    ax3.plot(t_eval_full, error, 'm-o', label='Error (Raw - PCHIP)')
    ax3.axhline(0, color='black', linestyle='--', linewidth=1)

    ax3.set_title('Reconstruction Error (X axis)')
    ax3.set_xlabel('Arc-length Parameterized Time')
    ax3.set_ylabel('Error')
    ax3.legend()
    
    # Print Mean Absolute Error
    mae = np.mean(np.abs(error))
    print(f"📊 Mean Absolute Error (X-axis): {mae:.6f}")

    plt.tight_layout()
    output_img = f'adaptive_pchip_{video_id}.png'
    plt.savefig(output_img, dpi=300)
    print(f"✅ Saved visualization to '{output_img}'")

if __name__ == "__main__":
    video_id = "05xCYOyY1bg"
    visualize_real_trajectory(video_id)