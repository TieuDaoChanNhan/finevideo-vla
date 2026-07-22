#!/usr/bin/env python3
"""
One-off experiment (not part of the production pipeline): take ONE real short
video, pull one real 8-frame chunk from it, encode+decode that same chunk at
several different Cosmos target_size values, and save each reconstruction as
its own mp4 (plus the original frames as a reference) so a human can compare
quality directly and pick a resolution -- rather than judging on synthetic
noise frames (which the earlier quick functional check in REPORT.md #35 used).

Per user request 2026-07-22: "chọn 1 video ngắn vài giây, chọn từng target
size, tokenize đếm token/chunk, rồi retokenize lại thành video để check kết
quả, rồi chọn."

Usage:
    python tools/cosmos_resolution_experiment.py \
        --video videos/good.mp4 --start-frame 40 \
        --sizes 160,224,256,320,384,448,512 \
        --output-dir samples/cosmos_resolution_experiment
"""
import argparse
import os
import sys

PROTOTYPE_DIR = "/e/project1/reformo/nguyen38/prototype"


def extract_chunk(video_path: str, start_frame: int, n_frames: int = 8):
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(n_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = frame_bgr[:, :, ::-1]
        frames.append(Image.fromarray(frame_rgb.copy()))
    cap.release()
    if len(frames) != n_frames:
        raise RuntimeError(f"Only got {len(frames)}/{n_frames} frames from {video_path} at start_frame={start_frame}")
    return frames


def save_reference_video(frames, output_path: str, view_size: int = 512, fps: int = 6):
    import subprocess
    import tempfile
    import imageio_ffmpeg
    tmp_dir = tempfile.mkdtemp(prefix="cosmos_ref_")
    for i, f in enumerate(frames):
        f.save(os.path.join(tmp_dir, f"frame_{i:02d}.png"))
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg_bin, "-y", "-framerate", str(fps), "-i", os.path.join(tmp_dir, "frame_%02d.png"),
         "-vf", f"scale=-2:{view_size}", "-pix_fmt", "yuv420p", os.path.abspath(output_path)],
        check=True, capture_output=True,
    )
    for i in range(len(frames)):
        os.remove(os.path.join(tmp_dir, f"frame_{i:02d}.png"))
    os.rmdir(tmp_dir)


def decode_grid(token_ids: list, grid: tuple, output_path: str, view_size: int = 512, fps: int = 6):
    """Generalized version of decode_cosmos.py's decode_cosmos_chunk() that accepts
    any (T', H', W') grid instead of the hardcoded (2, 10, 10) for target_size=160,
    and always upscales to a common view_size (nearest-neighbor, so blockiness from
    a low source resolution stays visible/honest rather than smoothed away)."""
    import subprocess
    import torch
    import torchvision.transforms as T
    import imageio_ffmpeg
    from cosmos_tokenizer.video_lib import CausalVideoTokenizer

    checkpoint_dec = os.path.join(PROTOTYPE_DIR, "pretrained_ckpts/Cosmos-Tokenizer-DV8x16x16/decoder.jit")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dec = CausalVideoTokenizer(checkpoint_dec=checkpoint_dec).to(device)

    indices = torch.tensor(token_ids, dtype=torch.int64, device=device).view(1, *grid)
    with torch.no_grad():
        out = dec.decode(indices)  # (1, 3, T, H, W)
    out = ((out.float() + 1.0) / 2.0).clamp(0, 1).squeeze(0)
    n_frames = out.shape[1]

    frame_dir = f"/tmp/cosmos_res_exp_{os.getpid()}"
    os.makedirs(frame_dir, exist_ok=True)
    to_pil = T.ToPILImage()
    for i in range(n_frames):
        to_pil(out[:, i, :, :].cpu()).save(f"{frame_dir}/frame_{i:02d}.png")

    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg_bin, "-y", "-framerate", str(fps), "-i", f"{frame_dir}/frame_%02d.png",
         "-vf", f"scale=-2:{view_size}:flags=neighbor", "-pix_fmt", "yuv420p",
         os.path.abspath(output_path)],
        check=True, capture_output=True,
    )
    for i in range(n_frames):
        os.remove(f"{frame_dir}/frame_{i:02d}.png")
    os.rmdir(frame_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="videos/good.mp4")
    ap.add_argument("--start-frame", type=int, default=40)
    ap.add_argument("--sizes", default="160,224,256,320,384,448,512")
    ap.add_argument("--output-dir", default="samples/cosmos_resolution_experiment")
    ap.add_argument("--view-size", type=int, default=512)
    args = ap.parse_args()

    args.output_dir = os.path.abspath(args.output_dir)  # resolve before any os.chdir() below
    args.video = os.path.abspath(args.video)
    os.makedirs(args.output_dir, exist_ok=True)
    sizes = [int(x) for x in args.sizes.split(",")]

    print(f"Extracting 8 real frames from {args.video} starting at frame {args.start_frame}...")
    frames = extract_chunk(args.video, args.start_frame)
    print(f"  frame size: {frames[0].size}")

    ref_path = os.path.join(args.output_dir, "00_original_reference.mp4")
    save_reference_video(frames, ref_path, view_size=args.view_size)
    print(f"  saved reference (native res, upscaled for viewing): {ref_path}")

    sys.path.insert(0, PROTOTYPE_DIR)
    os.chdir(PROTOTYPE_DIR)
    os.environ.setdefault("SLURM_NTASKS", "1")
    os.environ.setdefault("SLURM_PROCID", "0")
    os.environ.setdefault("SLURM_LOCALID", "0")
    from pipeline import CosmosVideoTokenizer

    cosmos = CosmosVideoTokenizer()
    print(f"Encoder loaded: {cosmos.encoder is not None}\n")

    w0, h0 = frames[0].size
    results = []
    for size in sizes:
        ids = cosmos.encode_video_chunk(frames, target_size=size)
        # Mirror encode_video_chunk()'s own aspect-preserving H/W computation
        # (2026-07-22 update: no longer a square crop) to get the right grid
        # shape for decode.
        if w0 >= h0:
            new_h = size
            new_w = max(16, round(size * w0 / h0 / 16) * 16)
        else:
            new_w = size
            new_h = max(16, round(size * h0 / w0 / 16) * 16)
        grid = (2, new_h // 16, new_w // 16)
        out_path = os.path.join(args.output_dir, f"target_{size:04d}_{len(ids)}tok.mp4")
        decode_grid(ids, grid, out_path, view_size=args.view_size)
        print(f"  target_size={size:4d} -> {len(ids):5d} tokens/chunk -> {out_path}")
        results.append((size, len(ids), out_path))

    summary_path = os.path.join(args.output_dir, "SUMMARY.md")
    with open(summary_path, "w") as f:
        f.write("# Cosmos resolution experiment\n\n")
        f.write(f"Source: `{args.video}`, frames {args.start_frame}-{args.start_frame + 7} "
                f"(native {frames[0].size[0]}x{frames[0].size[1]})\n\n")
        f.write("Reference (original, native res, upscaled for viewing): "
                "[00_original_reference.mp4](00_original_reference.mp4)\n\n")
        f.write("| target_size | tokens/chunk | file |\n|---|---|---|\n")
        for size, n, path in results:
            fname = os.path.basename(path)
            f.write(f"| {size} | {n} | [{fname}]({fname}) |\n")
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
