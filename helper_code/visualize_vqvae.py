import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import các class từ hệ thống của bạn
from dataset import KinematicDataset
from model import GRFSQ_VAE

# ================= 1. CẤU HÌNH =================
SKELETON_TREE = [
    (0, 1), (1, 2), (2, 3),        # Right Leg
    (0, 4), (4, 5), (5, 6),        # Left Leg
    (0, 7), (7, 8), (8, 9),        # Spine & Head
    (8, 11), (11, 12), (12, 13),   # Left Arm
    (8, 14), (14, 15), (15, 16)    # Right Arm
]

def load_global_stats(npz_path="outputs/global_stats.npz"):
    stats = np.load(npz_path)
    return torch.tensor(stats['mean'], dtype=torch.float32), torch.tensor(stats['std'], dtype=torch.float32)

def plot_pose3d(ax, pose, title="", color='blue'):
    x = pose[:, 0]
    y = pose[:, 2] 
    z = -pose[:, 1] 

    ax.scatter(x, y, z, c='black', s=8)

    for parent, child in SKELETON_TREE:
        ax.plot([x[parent], x[child]], [y[parent], y[child]], [z[parent], z[child]], 
                c=color, linewidth=2, alpha=0.8)

    ax.set_title(title, fontsize=10, pad=2)
    
    limit = 0.8
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    
    # Clean up axes for a professional look
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)

# ================= 2. MAIN SCRIPT =================
def generate_executive_report(epoch_checkpoint="outputs/grfsq_vae_epoch_50.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating Executive Visualization on {device}...")

    # Load Data & Model
    global_mean, global_std = load_global_stats()
    global_mean = global_mean.to(device)
    global_std = global_std.to(device)

    # 🚀 THAY ĐỔI Ở ĐÂY: NHẮM THẲNG MỤC TIÊU VÀO VIDEO 05xCYOyY1bg
    video_id = "05xCYOyY1bg"
    target_file = f"outputs/states/{video_id}_states.jsonl"
    
    if not os.path.exists(target_file):
        print(f"❌ Không tìm thấy file {target_file}!")
        return

    print(f"🎯 Đang trích xuất 8 frames đầu tiên từ video: {video_id}")
    with open(target_file, 'r') as f:
        first_line = f.readline() # Đọc đúng dòng đầu tiên (Window 0)
        data = json.loads(first_line)
        state_array = np.array(data["states"], dtype=np.float32)
        
    # Tạo Tensor đầu vào (Batch=1, Time=8, Features=153)
    x_input = torch.tensor(state_array).unsqueeze(0).to(device)

    model = GRFSQ_VAE(input_dim=153, latent_dim=128, codebook_size=512, num_quantizers=3).to(device)
    
    if os.path.exists(epoch_checkpoint):
        # Sửa lỗi cảnh báo bảo mật bằng weights_only=True
        model.load_state_dict(torch.load(epoch_checkpoint, map_location=device, weights_only=True))
    
    model.eval()
    with torch.no_grad():
        x_recon, tokens, _ = model(x_input)
        token_list = tokens[0].cpu().numpy()

    # Un-normalize & Reshape
    x_input_real = (x_input * global_std) + global_mean
    x_recon_real = (x_recon * global_std) + global_mean

    pos_input = x_input_real[0, :, 0:51].cpu().numpy().reshape(8, 17, 3)
    pos_recon = x_recon_real[0, :, 0:51].cpu().numpy().reshape(8, 17, 3)

    # Setup the Canvas
    fig = plt.figure(figsize=(20, 8))
    fig.patch.set_facecolor('#f8f9fa') # Light gray background for professional look
    
    # MAIN TITLE & TOKEN DISPLAY
    token_str = f"[{token_list[0]}, {token_list[1]}, {token_list[2]}]"
    fig.suptitle("MOTION TOKENIZATION VIA GRFSQ-VAE", fontsize=22, fontweight='bold', y=0.95)
    
    fig.text(0.5, 0.88, 
             f"Continuous 3D Motion (8 frames) successfully compressed into 3 Discrete Tokens:\n"
             f"Motion Tokens = {token_str}", 
             ha='center', fontsize=16, color='#dc3545', fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='#ced4da', boxstyle='round,pad=0.5'))

    # Plot Ground Truth (Row 1)
    for i in range(8):
        ax = fig.add_subplot(2, 8, i + 1, projection='3d')
        plot_pose3d(ax, pos_input[i], title=f"Real Frame {i+1}", color='#0d6efd') # Blue

    # Plot Reconstruction (Row 2)
    for i in range(8):
        ax = fig.add_subplot(2, 8, i + 9, projection='3d')
        plot_pose3d(ax, pos_recon[i], title=f"Recon Frame {i+1} (from tokens)", color='#dc3545') # Red

    plt.subplots_adjust(top=0.82, bottom=0.05, hspace=0.1, wspace=0.1)
    
    # Add a subtle footer
    fig.text(0.02, 0.02, "Data: Z-score Global Normalized | Model: Conv1D + Residual Vector Quantization", 
             fontsize=10, color='gray', style='italic')

    # Save
    output_img = "outputs/executive_token_visualization.png"
    plt.savefig(output_img, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"✅ Executive Report successfully saved to: {output_img}")
    plt.close()

if __name__ == "__main__":
    generate_executive_report()