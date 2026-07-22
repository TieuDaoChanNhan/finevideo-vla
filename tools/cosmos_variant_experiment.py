#!/usr/bin/env python3
"""
One-off experiment: compare NVIDIA's 3 Cosmos discrete-video-tokenizer
variants (same suite we already use, different compression tradeoffs) on the
SAME real 8-frame chunk, at the same aspect-preserving resize, to see if a
less-aggressively-compressed variant genuinely reconstructs sharper detail
(faces, signage) than our current Cosmos-Tokenizer-DV8x16x16 -- per the
user's 2026-07-22 observation that reconstructions look blurry/unnatural
regardless of resolution or aspect ratio, raising "is Cosmos itself just
weak, are there better options" (accepting a higher token cost if so).

NVIDIA's own published benchmark (pretrained_ckpts/*/README.md) already
ranks these by PSNR/SSIM/rFVD: DV4x8x8 > DV8x8x8 > DV8x16x16 (our current
one, the most-compressed/lowest-fidelity of the three) -- this script
verifies that ranking holds visually on our own real data, and gives the
real token-cost multiplier for each (not just NVIDIA's compression-ratio
labels), so a decision can be made with actual numbers.

Usage:
    python tools/cosmos_variant_experiment.py --video videos/good.mp4 --start-frame 40
"""
import argparse
import os
import sys

PROTOTYPE_DIR = "/e/project1/reformo/nguyen38/prototype"
VARIANTS = ["Cosmos-Tokenizer-DV8x16x16", "Cosmos-Tokenizer-DV8x8x8", "Cosmos-Tokenizer-DV4x8x8"]


def extract_chunk(video_path, start_frame, n_frames=8):
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(n_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(frame_bgr[:, :, ::-1].copy()))
    cap.release()
    if len(frames) != n_frames:
        raise RuntimeError(f"Only got {len(frames)}/{n_frames} frames")
    return frames


def save_reference_video(frames, output_path, view_size=512, fps=6):
    import subprocess, tempfile, imageio_ffmpeg
    tmp_dir = tempfile.mkdtemp(prefix="cosmos_variant_ref_")
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
    ap.add_argument("--video", default="videos/good.mp4")
    ap.add_argument("--start-frame", type=int, default=40)
    ap.add_argument("--target-size", type=int, default=256, help="shorter-side target, aspect-preserving")
    ap.add_argument("--output-dir", default="samples/cosmos_variant_experiment")
    ap.add_argument("--view-size", type=int, default=512)
    args = ap.parse_args()

    args.output_dir = os.path.abspath(args.output_dir)
    args.video = os.path.abspath(args.video)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Extracting 8 real frames from {args.video} starting at frame {args.start_frame}...")
    frames = extract_chunk(args.video, args.start_frame)
    w0, h0 = frames[0].size
    if w0 >= h0:
        new_h = args.target_size
        new_w = max(16, round(args.target_size * w0 / h0 / 16) * 16)
    else:
        new_w = args.target_size
        new_h = max(16, round(args.target_size * h0 / w0 / 16) * 16)
    print(f"  native {w0}x{h0} -> resized {new_w}x{new_h}")

    ref_path = os.path.join(args.output_dir, "00_original_reference.mp4")
    save_reference_video(frames, ref_path, view_size=args.view_size)
    print(f"  saved reference: {ref_path}")

    sys.path.insert(0, PROTOTYPE_DIR)
    os.chdir(PROTOTYPE_DIR)
    import torch
    from torchvision import transforms as T
    import torchvision.transforms as TT
    import imageio_ffmpeg
    import subprocess
    from cosmos_tokenizer.video_lib import CausalVideoTokenizer

    transform = T.Compose([T.Resize((new_h, new_w)), T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    tensors = torch.stack([transform(f.convert("RGB")) for f in frames])
    tensors = tensors.permute(1, 0, 2, 3).unsqueeze(0).to("cuda")

    results = []
    for variant in VARIANTS:
        enc_path = f"pretrained_ckpts/{variant}/encoder.jit"
        dec_path = f"pretrained_ckpts/{variant}/decoder.jit"
        enc = CausalVideoTokenizer(checkpoint_enc=enc_path).to("cuda")
        with torch.no_grad():
            indices, _ = enc.encode(tensors.to(torch.float16))
        n_tokens = indices.numel()
        del enc
        torch.cuda.empty_cache()

        dec = CausalVideoTokenizer(checkpoint_dec=dec_path).to("cuda")
        with torch.no_grad():
            out = dec.decode(indices)
        del dec
        torch.cuda.empty_cache()

        out = ((out.float() + 1.0) / 2.0).clamp(0, 1).squeeze(0)
        n_frames_out = out.shape[1]
        frame_dir = f"/tmp/cosmos_variant_{os.getpid()}"
        os.makedirs(frame_dir, exist_ok=True)
        to_pil = TT.ToPILImage()
        for i in range(n_frames_out):
            to_pil(out[:, i, :, :].cpu()).save(f"{frame_dir}/frame_{i:02d}.png")
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        out_path = os.path.join(args.output_dir, f"{variant}_{n_tokens}tok.mp4")
        subprocess.run(
            [ffmpeg_bin, "-y", "-framerate", "6", "-i", f"{frame_dir}/frame_%02d.png",
             "-vf", f"scale=-2:{args.view_size}:flags=neighbor", "-pix_fmt", "yuv420p",
             os.path.abspath(out_path)],
            check=True, capture_output=True,
        )
        for i in range(n_frames_out):
            os.remove(f"{frame_dir}/frame_{i:02d}.png")
        os.rmdir(frame_dir)

        print(f"  {variant}: {n_tokens} tokens -> {out_path}")
        results.append((variant, n_tokens, out_path))

    summary_path = os.path.join(args.output_dir, "SUMMARY.md")
    with open(summary_path, "w") as f:
        f.write("# Cosmos variant comparison\n\n")
        f.write(f"Source: `{args.video}`, frames {args.start_frame}-{args.start_frame+7}, "
                f"resized to {new_w}x{new_h} (target_size={args.target_size}, aspect-preserving)\n\n")
        f.write("Reference: [00_original_reference.mp4](00_original_reference.mp4)\n\n")
        f.write("| variant | tokens/chunk | ratio vs current | file |\n|---|---|---|---|\n")
        base = results[0][1]
        for variant, n, path in results:
            f.write(f"| {variant} | {n} | {n/base:.1f}x | [{os.path.basename(path)}]({os.path.basename(path)}) |\n")
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
