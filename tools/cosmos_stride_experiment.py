#!/usr/bin/env python3
"""
One-off experiment: "Cach A" from the 2026-07-22 discussion (REPORT.md #35
follow-up) -- speed up the source video before feeding Cosmos, instead of (or
in addition to) changing resolution. Since Cosmos's causal temporal
compression always turns exactly 8 input frames into 2 temporal codes
regardless of source fps, sampling every Nth native frame instead of every
frame keeps token count IDENTICAL while making each chunk represent N times
more real-world elapsed time -- directly targeting Huu's "8 frames might be
way too slow" observation at zero extra token cost (unlike the resolution
fix, which is token-expensive).

Picks 1 real short clip with continuous motion, extracts 8-frame chunks at
several strides (1x/2x/3x/4x = same real span as if playback were sped up by
that factor), encodes+decodes each at a fixed resolution so only the stride
varies, for direct visual A/B.

Usage:
    python tools/cosmos_stride_experiment.py --video videos/boxing.mp4 \
        --start-frame 100 --strides 1,2,3,4 --output-dir samples/cosmos_stride_experiment
"""
import argparse
import os
import sys

PROTOTYPE_DIR = "/e/project1/reformo/nguyen38/prototype"


def extract_strided_chunk(video_path: str, start_frame: int, stride: int, n_frames: int = 8):
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    for i in range(n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i * stride)
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(frame_bgr[:, :, ::-1].copy()))
    cap.release()
    if len(frames) != n_frames:
        raise RuntimeError(f"Only got {len(frames)}/{n_frames} frames (stride={stride})")
    real_span = (n_frames - 1) * stride / fps if fps else 0
    return frames, fps, real_span


def decode_grid(token_ids, grid, output_path, view_size=512, fps=6):
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
        out = dec.decode(indices)
    out = ((out.float() + 1.0) / 2.0).clamp(0, 1).squeeze(0)
    n_frames = out.shape[1]
    frame_dir = f"/tmp/cosmos_stride_exp_{os.getpid()}"
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


def save_reference_video(frames, output_path, view_size=512, fps=6):
    import subprocess
    import tempfile
    import imageio_ffmpeg
    tmp_dir = tempfile.mkdtemp(prefix="cosmos_stride_ref_")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="videos/boxing.mp4")
    ap.add_argument("--start-frame", type=int, default=100)
    ap.add_argument("--strides", default="1,2,3,4")
    ap.add_argument("--target-size", type=int, default=256)
    ap.add_argument("--output-dir", default="samples/cosmos_stride_experiment")
    ap.add_argument("--view-size", type=int, default=512)
    args = ap.parse_args()

    args.output_dir = os.path.abspath(args.output_dir)
    args.video = os.path.abspath(args.video)
    os.makedirs(args.output_dir, exist_ok=True)
    strides = [int(x) for x in args.strides.split(",")]

    sys.path.insert(0, PROTOTYPE_DIR)
    os.chdir(PROTOTYPE_DIR)
    os.environ.setdefault("SLURM_NTASKS", "1")
    os.environ.setdefault("SLURM_PROCID", "0")
    os.environ.setdefault("SLURM_LOCALID", "0")
    from pipeline import CosmosVideoTokenizer
    cosmos = CosmosVideoTokenizer()
    print(f"Encoder loaded: {cosmos.encoder is not None}\n")

    results = []
    for stride in strides:
        frames, fps, real_span = extract_strided_chunk(args.video, args.start_frame, stride)
        w0, h0 = frames[0].size
        ref_path = os.path.join(args.output_dir, f"stride{stride}_00_original_frames.mp4")
        save_reference_video(frames, ref_path, view_size=args.view_size)

        ids = cosmos.encode_video_chunk(frames, target_size=args.target_size)
        if w0 >= h0:
            new_h = args.target_size
            new_w = max(16, round(args.target_size * w0 / h0 / 16) * 16)
        else:
            new_w = args.target_size
            new_h = max(16, round(args.target_size * h0 / w0 / 16) * 16)
        grid = (2, new_h // 16, new_w // 16)

        out_path = os.path.join(args.output_dir, f"stride{stride}_decoded_{len(ids)}tok.mp4")
        decode_grid(ids, grid, out_path, view_size=args.view_size)
        print(f"  stride={stride} (native fps={fps:.1f}, chunk spans {real_span:.3f}s real time) "
              f"-> {len(ids)} tokens -> {out_path}")
        results.append((stride, fps, real_span, len(ids), out_path))

    summary_path = os.path.join(args.output_dir, "SUMMARY.md")
    with open(summary_path, "w") as f:
        f.write("# Cosmos stride (window-duration) experiment\n\n")
        f.write(f"Source: `{args.video}`, target_size={args.target_size} (fixed, so only stride varies)\n\n")
        f.write("| stride | native fps | real time/chunk | tokens | original | decoded |\n|---|---|---|---|---|---|\n")
        for stride, fps, span, n, path in results:
            ref = f"stride{stride}_00_original_frames.mp4"
            dec = os.path.basename(path)
            f.write(f"| {stride}x | {fps:.1f} | {span:.3f}s | {n} | [{ref}]({ref}) | [{dec}]({dec}) |\n")
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
