import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.interpolate import PchipInterpolator

from phase4_interpolation_tokenizer import AdaptiveInterpolationTokenizer

# ================= CONFIGURATION =================
VIDEO_IN = "../videos/--5iwqOe8G8.mp4"
TOKENS_IN = "../outputs/agent_tokens_test/--5iwqOe8G8_tokens.jsonl"
VIDEO_OUT = "../outputs/filtered_skeleton.mp4"

skeleton_tree = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16)
]

def interpolate_trajectory(trajectory):
    """Linear interpolation to fill gaps caused by Phase 4 stride skipping."""
    N = trajectory.shape[0]
    traj_flat = trajectory.reshape(N, -1)

    for j in range(traj_flat.shape[1]):
        y = traj_flat[:, j]
        nans = np.isnan(y)
        if not nans.any() or nans.all():
            continue
        x = np.arange(N)
        traj_flat[nans, j] = np.interp(x[nans], x[~nans], y[~nans])

    return traj_flat.reshape(trajectory.shape)

def load_and_reconstruct_tokens(total_video_frames):
    print("🔍 Decoding agent tokens with PCHIP...")
    tokenizer = AdaptiveInterpolationTokenizer()
    chunks = []

    with open(TOKENS_IN, "r") as f:
        for line in f:
            chunks.append(json.loads(line))

    if not chunks:
        raise ValueError("No tokens found to render!")

    full_trajectory = np.full((total_video_frames, 17, 3), np.nan, dtype=np.float32)

    for data in chunks:
        # Use exact time coordinates from extraction
        window_id = data["window_id"]
        package = data["package"]
        cp = tokenizer.decode_chunk(package)

        # Use original t_cp to preserve real motion acceleration
        t_cp = package["t_cp"]

        t_8 = np.linspace(0, 1, 8, dtype=np.float32)
        recon = np.zeros((8, 17, 3), dtype=np.float32)

        for j in range(17):
            for d in range(3):
                spline = PchipInterpolator(t_cp, cp[:, j, d])
                recon[:, j, d] = spline(t_8)

        # Place the 8 reconstructed frames back onto the absolute timeline
        end_frame = min(window_id + 8, total_video_frames)
        length = end_frame - window_id
        if length > 0:
            full_trajectory[window_id:end_frame] = recon[:length]

    # Fill gaps (when stride > 8)
    full_trajectory = interpolate_trajectory(full_trajectory)
    return full_trajectory

def draw_3d_pose(ax, pose):
    ax.clear()
    ax.view_init(elev=15, azim=-90)

    x = pose[:, 0]
    y = pose[:, 2]
    z = -pose[:, 1]

    # --- DYNAMIC CAMERA: Focus on the Pelvis (Joint 0) ---
    pelvis_x, pelvis_y, pelvis_z = x[0], y[0], z[0]
    radius = 1.0  # 1-metre box radius
    ax.set_xlim3d([pelvis_x - radius, pelvis_x + radius])
    ax.set_ylim3d([pelvis_y - radius, pelvis_y + radius])
    ax.set_zlim3d([pelvis_z - radius, pelvis_z + radius])

    ax.scatter(x, y, z, c="cyan", s=30, edgecolors="white")
    for parent, child in skeleton_tree:
        ax.plot(
            [x[parent], x[child]],
            [y[parent], y[child]],
            [z[parent], z[child]],
            c="lime",
            linewidth=3
        )

    ax.set_facecolor("black")
    ax.axis("off")

def main():
    # Open the source video only to read metadata (frame count & FPS)
    cap = cv2.VideoCapture(VIDEO_IN)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    poses = load_and_reconstruct_tokens(total_video_frames)
    print(f"✅ Successfully reconstructed {total_video_frames} frames on the absolute timeline.")

    render_size = 600
    fig = plt.figure(figsize=(6, 6), dpi=100)
    fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111, projection="3d")
    canvas = FigureCanvasAgg(fig)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (render_size, render_size))

    print(f"🎥 Rendering skeleton-only video: {VIDEO_OUT}...")

    for frame_idx in range(total_video_frames):
        pose = poses[frame_idx]

        if not np.isnan(pose).all():
            draw_3d_pose(ax, pose)
        else:
            ax.clear()
            ax.set_facecolor("black")
            ax.axis("off")

        canvas.draw()
        plot_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Convert colour space and resize if needed
        plot_bgr = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
        if plot_bgr.shape[0] != render_size or plot_bgr.shape[1] != render_size:
            plot_bgr = cv2.resize(plot_bgr, (render_size, render_size))

        out.write(plot_bgr)

        if frame_idx % 20 == 0:
            print(f"   ⏳ Rendered {frame_idx}/{total_video_frames} frames...", end="\r")

    out.release()
    plt.close(fig)
    print(f"\n✅ Done! Skeleton-only video saved to: {VIDEO_OUT}")

if __name__ == "__main__":
    main()
