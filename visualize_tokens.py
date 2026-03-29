import argparse
import json
import os
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['animation.ffmpeg_path'] = os.environ.get('FFMPEG_PATH', 'ffmpeg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from phase5_interpolation_tokenizer_fixed import AdaptiveInterpolationTokenizer


SKELETON_TREE: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize reconstructed 3D skeleton motion from Phase 5 token JSONL."
    )
    parser.add_argument(
        "--tokens-jsonl",
        required=True,
        help="Path to a Phase 5 output JSONL file.",
    )
    parser.add_argument(
        "--output-mp4",
        required=True,
        help="Path to save the output MP4 video.",
    )
    return parser.parse_args()


def load_motion_sequence(tokens_jsonl: str, tokenizer: AdaptiveInterpolationTokenizer) -> np.ndarray:
    records = []
    with open(tokens_jsonl, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading JSONL", unit="line"):
            if not line.strip():
                continue
            records.append(json.loads(line))

    if not records:
        raise ValueError(f"No valid JSONL records found in: {tokens_jsonl}")

    records.sort(key=lambda x: (str(x.get("video_id", "")), int(x.get("window_id", 0))))

    video_ids = sorted({str(r.get("video_id", "unknown")) for r in records})
    if len(video_ids) > 1:
        print(
            f"[WARN] File contains multiple video_id values: {video_ids}. "
            "They will be stitched in sorted order."
        )

    all_recon = []
    for record in tqdm(records, desc="Decoding + reconstructing", unit="chunk"):
        package = record["package"]
        cp = tokenizer.decode_chunk(package)
        recon = tokenizer.reconstruct(cp, package["t_cp"])
        all_recon.append(recon)

    motion = np.concatenate(all_recon, axis=0)
    if motion.ndim != 3 or motion.shape[1:] != (17, 3):
        raise ValueError(f"Unexpected motion shape: {motion.shape}, expected (T, 17, 3)")

    return motion


def compute_axes_limits(sequence: np.ndarray, padding_ratio: float = 0.1):
    mins = sequence.min(axis=(0, 1))
    maxs = sequence.max(axis=(0, 1))
    center = (mins + maxs) / 2.0
    spans = maxs - mins
    max_span = float(np.max(spans))
    if max_span <= 1e-6:
        max_span = 1.0

    half_range = (max_span * (1.0 + padding_ratio)) / 2.0
    xlim = (center[0] - half_range, center[0] + half_range)
    ylim = (center[1] - half_range, center[1] + half_range)
    zlim = (center[2] - half_range, center[2] + half_range)
    return xlim, ylim, zlim, center


def render_video(sequence: np.ndarray, output_mp4: str, fps: int = 30) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_mp4)), exist_ok=True)

    xlim, ylim, zlim, center = compute_axes_limits(sequence)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    def setup_axes():
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Phase 5 Token Reconstruction Review")

        # Góc nhìn ổn định để dễ nghiệm thu dáng.
        ax.view_init(elev=18, azim=-65)

        # Đặt tâm nhìn vào giữa chuyển động.
        ax.set_proj_type("persp")
        try:
            ax.dist = 9
        except Exception:
            pass

    points = ax.scatter([], [], [], s=28)
    lines = [ax.plot([], [], [], linewidth=2)[0] for _ in SKELETON_TREE]
    frame_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def init():
        setup_axes()
        points._offsets3d = ([], [], [])
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        frame_text.set_text("")
        return [points, frame_text, *lines]

    def update(frame_idx: int):
        joints = sequence[frame_idx]
        xs, ys, zs = joints[:, 0], joints[:, 1], joints[:, 2]
        points._offsets3d = (xs, ys, zs)

        for line, (a, b) in zip(lines, SKELETON_TREE):
            line.set_data([xs[a], xs[b]], [ys[a], ys[b]])
            line.set_3d_properties([zs[a], zs[b]])

        frame_text.set_text(
            f"Frame: {frame_idx + 1}/{len(sequence)}\n"
            f"Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})"
        )
        return [points, frame_text, *lines]

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(sequence),
        interval=1000 / fps,
        blit=False,
    )

    # Lấy đường dẫn ffmpeg từ biến môi trường mà cậu đã export
    ffmpeg_bin = os.environ.get('FFMPEG_PATH')
    
    # Khởi tạo writer với tham số executable
    writer = FFMpegWriter(fps=fps, bitrate=2500)
    
    with tqdm(total=len(sequence), desc="Rendering MP4", unit="frame") as pbar:
        anim.save(output_mp4, writer=writer, progress_callback=lambda i, n: pbar.update(1))

    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()

    tokenizer = AdaptiveInterpolationTokenizer()
    sequence = load_motion_sequence(args.tokens_jsonl, tokenizer)

    print(f"Loaded reconstructed motion with shape: {sequence.shape}")
    render_video(sequence, args.output_mp4)
    print(f"Saved MP4 to: {args.output_mp4}")
