#!/usr/bin/env python3
"""
Render `<agent>` pose tokens (decoded by tools/eval/decode_agent_tokens.py)
as a standalone 3D skeleton animation -- no source video required.

Unlike tools/visualize/render_filtered_skeleton.py (which overlays a decoded
skeleton on its original video, for pipeline sanity-checking against real
footage), this is for the case decode_agent_tokens.py already handles but
never visualizes: a bare (n_windows, 8, 17, 3) trajectory with no video to
overlay on -- e.g. a sequence the model generated from scratch.

Joint order / skeleton edges match render_filtered_skeleton.py's convention
(H36M 17-joint layout, see decode_agent_tokens.py's JOINT_NAMES docstring).

Usage:
    python tools/visualize/render_agent_pose.py --input tokens.txt --output pose.mp4 [--fps 5]
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
from decode_agent_tokens import decode, JOINT_NAMES  # noqa: E402

SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]


def render(trajectory: np.ndarray, output_path: str, fps: int) -> None:
    """trajectory: (n_frames, 17, 3) in metres. Renders one PNG per frame,
    then stitches with ffmpeg at `fps` (deliberately slow relative to the
    native 30fps window rate, same "few real frames -> slow fps for
    visibility" convention as decode_cosmos.py, since a short generated
    snippet is only a handful of frames)."""
    import imageio_ffmpeg
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = os.path.abspath(output_path)
    frame_dir = f"/tmp/agent_pose_render_{os.getpid()}"
    os.makedirs(frame_dir, exist_ok=True)

    lo, hi = trajectory.min(), trajectory.max()
    pad = 0.1 * max(hi - lo, 1e-3)
    lo, hi = lo - pad, hi + pad

    for i, frame in enumerate(trajectory):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        xs, ys, zs = frame[:, 0], frame[:, 1], frame[:, 2]
        # Pose coords are (x, y, z) with y as the image-plane vertical axis
        # pointing down (quantized straight from 2D+depth, see
        # decode_agent_tokens.py's dequantize()) -- flip for a natural
        # "up is up" 3D view.
        ax.scatter(xs, zs, -ys, c="tab:blue", s=25)
        for a, b in SKELETON_EDGES:
            ax.plot([xs[a], xs[b]], [zs[a], zs[b]], [-ys[a], -ys[b]], c="tab:blue")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_zlim(-hi, -lo)
        ax.set_title(f"frame {i}")
        ax.set_axis_off()
        fig.savefig(f"{frame_dir}/frame_{i:03d}.png", dpi=100)
        plt.close(fig)

    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg_bin, "-y", "-framerate", str(fps), "-i", f"{frame_dir}/frame_%03d.png",
         "-pix_fmt", "yuv420p", output_path],
        check=True, capture_output=True,
    )
    for i in range(len(trajectory)):
        os.remove(f"{frame_dir}/frame_{i:03d}.png")
    os.rmdir(frame_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Text file containing <agent>...</agent> token block(s)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--fps", type=int, default=5, help="Output mp4 framerate")
    args = ap.parse_args()

    token_str = open(args.input, encoding="utf-8").read()
    trajectories = decode(token_str)
    if not trajectories:
        ap.error("No agent windows decoded from input")

    stacked = np.concatenate(trajectories, axis=0)  # (n_windows*8, 17, 3)
    print(f"Decoded {len(trajectories)} window(s), {stacked.shape[0]} frames total, "
          f"range [{stacked.min():.3f}, {stacked.max():.3f}] m")
    render(stacked, args.output, args.fps)
    print(f"Saved: {args.output} ({stacked.shape[0]} frames @ {args.fps}fps = "
          f"{stacked.shape[0] / args.fps:.1f}s playback)")


if __name__ == "__main__":
    main()
