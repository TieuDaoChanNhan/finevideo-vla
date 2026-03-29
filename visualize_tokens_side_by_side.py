import argparse
import json
import os
from typing import List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm

from phase5_interpolation_tokenizer_fixed import AdaptiveInterpolationTokenizer


SKELETON_TREE: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Phase 5 tokens side-by-side with the real video."
    )
    parser.add_argument(
        "--tokens-jsonl",
        required=True,
        help="Path to a Phase 5 tokens JSONL file.",
    )
    parser.add_argument(
        "--video-real",
        required=True,
        help="Path to the original video file.",
    )
    parser.add_argument(
        "--output-mp4",
        required=True,
        help="Path to save the side-by-side MP4 output.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override output FPS. Defaults to the source video FPS, or 30 if unavailable.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Figure DPI for output video rendering.",
    )
    return parser.parse_args()


def load_token_records(tokens_jsonl: str):
    records = []
    with open(tokens_jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            data = json.loads(line)
            if "window_id" not in data or "package" not in data:
                raise ValueError(
                    f"Record at line {line_no} is missing 'window_id' or 'package'."
                )
            records.append(data)

    if not records:
        raise ValueError(f"No valid records found in {tokens_jsonl}")

    records.sort(key=lambda x: (x.get("video_id", ""), x["window_id"]))
    return records


def reconstruct_8_frames(cp: np.ndarray, t_cp: List[float]) -> np.ndarray:
    t_cp = np.asarray(t_cp, dtype=float)
    t = np.linspace(0.0, 1.0, 8)
    recon = np.zeros((8, 17, 3), dtype=np.float32)

    for j in range(17):
        for d in range(3):
            spline = PchipInterpolator(t_cp, cp[:, j, d])
            recon[:, j, d] = spline(t)

    return recon


def read_window_frames(cap: cv2.VideoCapture, window_id: int, num_frames: int = 8) -> List[np.ndarray]:
    ok = cap.set(cv2.CAP_PROP_POS_FRAMES, window_id)
    if not ok:
        # Some backends return False even though seeking still works, so continue.
        pass

    frames = []
    for _ in range(num_frames):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    return frames


def convert_pose_for_plot(pose_xyz: np.ndarray) -> np.ndarray:
    x = pose_xyz[:, 0]
    y = pose_xyz[:, 2]
    z = -pose_xyz[:, 1]
    return np.stack([x, y, z], axis=1)


def compute_axis_limits(all_points_xyz: np.ndarray):
    mins = all_points_xyz.min(axis=0)
    maxs = all_points_xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    spans = maxs - mins
    max_span = float(np.max(spans))
    if max_span <= 0:
        max_span = 1.0
    half = max_span / 2.0

    xlim = (center[0] - half, center[0] + half)
    ylim = (center[1] - half, center[1] + half)
    zlim = (center[2] - half, center[2] + half)
    return xlim, ylim, zlim


def build_synced_sequences(records, cap, tokenizer):
    video_frames = []
    skeleton_frames = []

    for record in tqdm(records, desc="Syncing tokens with real video", unit="window"):
        window_id = int(record["window_id"])
        package = record["package"]

        cp = tokenizer.decode_chunk(package)
        recon8 = reconstruct_8_frames(cp, package["t_cp"])
        real_frames = read_window_frames(cap, window_id, num_frames=8)

        synced_len = min(len(real_frames), recon8.shape[0])
        if synced_len == 0:
            continue

        video_frames.extend(real_frames[:synced_len])
        skeleton_frames.extend(recon8[:synced_len])

    if not video_frames or not skeleton_frames:
        raise ValueError("No synchronized frames could be built from the inputs.")

    return video_frames, np.asarray(skeleton_frames, dtype=np.float32)


def init_figure(first_frame: np.ndarray):
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    img_artist = ax1.imshow(first_frame)
    ax1.set_title("Real Video")
    ax1.axis("off")

    ax2.set_title("Reconstructed 3D Skeleton")
    scatter = ax2.scatter([], [], [], s=18)
    line_artists = [ax2.plot([], [], [], linewidth=2)[0] for _ in SKELETON_TREE]

    return fig, ax1, ax2, img_artist, scatter, line_artists


def configure_3d_axes(ax2, xlim, ylim, zlim):
    ax2.set_xlim(*xlim)
    ax2.set_ylim(*ylim)
    ax2.set_zlim(*zlim)
    ax2.set_box_aspect((1, 1, 1))
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")
    ax2.set_zlabel("-Y")
    ax2.view_init(elev=18, azim=-60)
    try:
        ax2.dist = 8
    except Exception:
        pass


def update_skeleton_artists(pose_xyz: np.ndarray, scatter, line_artists):
    pose_plot = convert_pose_for_plot(pose_xyz)
    xs = pose_plot[:, 0]
    ys = pose_plot[:, 1]
    zs = pose_plot[:, 2]

    scatter._offsets3d = (xs, ys, zs)

    for line_artist, (a, b) in zip(line_artists, SKELETON_TREE):
        line_artist.set_data([xs[a], xs[b]], [ys[a], ys[b]])
        line_artist.set_3d_properties([zs[a], zs[b]])


def render_video(video_frames, skeleton_frames, output_mp4: str, fps: float, dpi: int):
    all_plot_points = np.asarray(
        [convert_pose_for_plot(pose) for pose in skeleton_frames], dtype=np.float32
    ).reshape(-1, 3)
    xlim, ylim, zlim = compute_axis_limits(all_plot_points)

    fig, _ax1, ax2, img_artist, scatter, line_artists = init_figure(video_frames[0])
    configure_3d_axes(ax2, xlim, ylim, zlim)
    update_skeleton_artists(skeleton_frames[0], scatter, line_artists)

    ffmpeg_path = os.environ.get("FFMPEG_PATH")
    if ffmpeg_path:
        plt.rcParams["animation.ffmpeg_path"] = ffmpeg_path

    os.makedirs(os.path.dirname(os.path.abspath(output_mp4)), exist_ok=True)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=2400)

    with writer.saving(fig, output_mp4, dpi=dpi):
        for frame_rgb, pose_xyz in tqdm(
            zip(video_frames, skeleton_frames),
            total=len(video_frames),
            desc="Rendering MP4",
            unit="frame",
        ):
            img_artist.set_data(frame_rgb)
            update_skeleton_artists(pose_xyz, scatter, line_artists)
            writer.grab_frame()

    plt.close(fig)


def main():
    args = parse_args()

    tokenizer = AdaptiveInterpolationTokenizer()
    records = load_token_records(args.tokens_jsonl)

    cap = cv2.VideoCapture(args.video_real)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_real}")

    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = args.fps if args.fps is not None else (src_fps if src_fps and src_fps > 0 else 30.0)

        video_frames, skeleton_frames = build_synced_sequences(records, cap, tokenizer)
        render_video(
            video_frames=video_frames,
            skeleton_frames=skeleton_frames,
            output_mp4=args.output_mp4,
            fps=fps,
            dpi=args.dpi,
        )
    finally:
        cap.release()

    print(f"Saved side-by-side validation video to: {args.output_mp4}")
    print(f"Total synced frames: {len(skeleton_frames)}")


if __name__ == "__main__":
    main()
