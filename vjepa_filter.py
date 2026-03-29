#!/usr/bin/env python3
"""
Semantic Video Filtering with V-JEPA 2 (Meta / Hugging Face)

This script compares a "real" video stream against a "skeleton / agent-token render"
video stream using a frozen V-JEPA 2 encoder and cosine similarity over sliding clips.

Key features
------------
- Backbone: facebook/vjepa2-vitl-fpc64-256
- Sliding window over video
- Clip definition:
    - sample_span = 64 real frames
    - sampled_frames = 16
    - temporal_stride = 4
- Frozen encoder
- Global Average Pooling or learnable attentive pooling
- Mixed precision on CUDA
- Batch inference
- Suitable as a starting point for large-scale filtering jobs
- Saves pooled clip embeddings for both real and skeleton videos to a .pt file

Notes
-----
1) The Hugging Face V-JEPA 2 checkpoint `facebook/vjepa2-vitl-fpc64-256` is configured
   around 64-frame clips and its official video processor uses resize/crop settings that
   land on 256x256 crops, not 224x224. If you must force 224x224 for your pipeline, set
   `force_size=224`, but the most faithful option is to keep `force_size=None` and let
   `AutoVideoProcessor` apply the model-native preprocessing.
2) For "semantic video filtering", this script uses encoder features only and computes
   clip-level cosine similarity between the real video and skeleton-rendered video.
3) `AttentivePooler` here is an inference-time aggregation head over encoder tokens. It is
   not part of the pretrained checkpoint. By default, the script uses frozen attention
   weights as initialized. For a stronger system, train/calibrate this pooler on your own
   matched real-vs-skeleton dataset, or keep `pooling="gap"` for a simple no-training path.
4) The saved `.pt` file stores pooled clip embeddings before L2 normalization, which is
   generally more useful as downstream features. Cosine similarity is still computed using
   L2-normalized embeddings, preserving the original filtering behavior.

Install
-------
pip install -U "torch>=2.2" torchvision timm av huggingface_hub einops numpy
pip install -U git+https://github.com/huggingface/transformers

Example
-------
python vjepa_filter.py \
    --video-real video_real.mp4 \
    --video-skeleton video_skeleton.mp4 \
    --threshold 0.70 \
    --batch-size 8 \
    --pooling gap \
    --output-json result.json \
    --output-embeddings outputs/vjepa_embeddings.pt
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import av
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoVideoProcessor


# -----------------------------
# Data containers
# -----------------------------

@dataclass
class ClipScore:
    clip_index: int
    start_frame: int
    end_frame_inclusive: int
    sampled_indices: List[int]
    similarity: float
    is_anomaly: bool


@dataclass
class FilterResult:
    video_real: str
    video_skeleton: str
    model_name: str
    threshold: float
    sample_span: int
    sampled_frames: int
    temporal_stride: int
    window_step: int
    pooling: str
    total_clips: int
    anomaly_ratio: float
    clip_scores: List[ClipScore]


# -----------------------------
# Utility helpers
# -----------------------------

def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def infer_autocast_dtype(device: torch.device, prefer_bf16: bool = True) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if prefer_bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def format_float(x: float, ndigits: int = 4) -> float:
    return round(float(x), ndigits)


# -----------------------------
# Video decoding
# -----------------------------

class VideoReaderPyAV:
    """
    Lightweight frame reader backed by PyAV.

    Decodes the full video into memory as uint8 RGB frames in T x H x W x C format.
    This is simple and reliable for moderately sized clips. For production-scale 40k-video
    pipelines, replace this with a shard-aware dataset + worker-local decoder strategy.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.frames = self._decode_all(path)  # uint8, T x H x W x C
        self.num_frames = int(self.frames.shape[0])

    @staticmethod
    def _decode_all(path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video not found: {path}")
        container = av.open(path)
        frames: List[np.ndarray] = []
        try:
            for frame in container.decode(video=0):
                img = frame.to_rgb().to_ndarray()  # H x W x C, uint8
                frames.append(img)
        finally:
            container.close()

        if not frames:
            raise RuntimeError(f"No frames decoded from video: {path}")
        return np.stack(frames, axis=0)

    def get_frames(self, indices: Sequence[int]) -> np.ndarray:
        idx = np.asarray(indices, dtype=np.int64)
        if idx.min() < 0 or idx.max() >= self.num_frames:
            raise IndexError(
                f"Requested frame indices out of range. "
                f"min={idx.min()}, max={idx.max()}, num_frames={self.num_frames}"
            )
        return self.frames[idx]  # T x H x W x C


# -----------------------------
# Pooling
# -----------------------------

class AttentivePooler(nn.Module):
    """
    Simple attention pooling over token embeddings.

    Input:  x -> [B, N, D]
    Output: pooled -> [B, D]

    This module is optional. If left untrained, prefer GAP for stability.
    """

    def __init__(self, dim: int, num_heads: int = 8, qk_dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim or dim
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(x.shape[0], -1, -1)  # [B, 1, D]
        pooled, _ = self.attn(q, x, x, need_weights=False)
        pooled = self.norm(pooled[:, 0, :])
        return pooled


# -----------------------------
# Main filtering class
# -----------------------------

MY_HF_TOKEN = "..."
MY_CACHE_DIR = "/e/project1/reformo/nguyen38/hf_cache"


class VJEPAFilter(nn.Module):
    """
    Semantic video filtering using a frozen V-JEPA 2 encoder.

    Expected clip sampling
    ----------------------
    sampled_frames = 16
    temporal_stride = 4
    sample_span = 64

    sampled indices inside a window:
        [start, start+4, start+8, ..., start+60]
    """

    def __init__(
        self,
        model_name: str = "facebook/vjepa2-vitl-fpc64-256",
        pooling: str = "gap",
        device: str = "auto",
        amp: bool = True,
        force_size: Optional[int] = None,
        attn_implementation: str = "sdpa",
        use_torch_compile: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = resolve_device(device)
        self.amp = amp and self.device.type == "cuda"
        self.autocast_dtype = infer_autocast_dtype(self.device)

        self.processor = AutoVideoProcessor.from_pretrained(
            model_name,
            token=MY_HF_TOKEN,
            cache_dir=MY_CACHE_DIR,
        )

        model_dtype = self.autocast_dtype if self.device.type == "cuda" else torch.float32
        self.encoder = AutoModel.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
            attn_implementation=attn_implementation,
            token=MY_HF_TOKEN,
            cache_dir=MY_CACHE_DIR,
        )
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        self.encoder.to(self.device)

        hidden_size = int(self.encoder.config.hidden_size)

        pooling = pooling.lower()
        if pooling not in {"gap", "attn"}:
            raise ValueError("pooling must be one of: {'gap', 'attn'}")
        self.pooling = pooling

        if self.pooling == "attn":
            num_heads = min(8, int(self.encoder.config.num_attention_heads))
            self.pooler = AttentivePooler(dim=hidden_size, num_heads=num_heads).to(self.device)
            self.pooler.eval()
            self.pooler.requires_grad_(False)
        else:
            self.pooler = None

        self.force_size = force_size

        if use_torch_compile and hasattr(torch, "compile"):
            self.encoder = torch.compile(self.encoder)  # type: ignore[assignment]

    @staticmethod
    def build_sample_indices(
        start_frame: int,
        sampled_frames: int = 16,
        temporal_stride: int = 4,
    ) -> List[int]:
        return [start_frame + i * temporal_stride for i in range(sampled_frames)]

    @staticmethod
    def build_windows(
        num_frames: int,
        sampled_frames: int = 16,
        temporal_stride: int = 4,
        window_step: int = 16,
    ) -> List[Tuple[int, int, List[int]]]:
        """
        Returns list of:
            (start_frame, end_frame_inclusive, sampled_indices)

        The sampled clip covers:
            sample_span = sampled_frames * temporal_stride
        but because indexing is inclusive, the last sampled index is:
            start + (sampled_frames - 1) * temporal_stride
        """
        assert sampled_frames > 0
        assert temporal_stride > 0
        assert window_step > 0

        last_required = (sampled_frames - 1) * temporal_stride
        max_start = num_frames - 1 - last_required
        if max_start < 0:
            return []

        windows: List[Tuple[int, int, List[int]]] = []
        for start in range(0, max_start + 1, window_step):
            idx = VJEPAFilter.build_sample_indices(
                start_frame=start,
                sampled_frames=sampled_frames,
                temporal_stride=temporal_stride,
            )
            end_frame = idx[-1]
            windows.append((start, end_frame, idx))
        return windows

    def _preprocess_numpy_batch(self, clips_uint8: List[np.ndarray]) -> Dict[str, torch.Tensor]:
        processor_kwargs: Dict[str, Any] = {"return_tensors": "pt"}

        if self.force_size is not None:
            processor_kwargs.update(
                {
                    "do_resize": True,
                    "size": {"height": self.force_size, "width": self.force_size},
                    "do_center_crop": False,
                }
            )

        batch = self.processor(clips_uint8, **processor_kwargs)

        device_batch: Dict[str, torch.Tensor] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
        return device_batch

    @torch.inference_mode()
    def encode_clips(self, clips_uint8: List[np.ndarray], normalize: bool = True) -> torch.Tensor:
        """
        Returns pooled clip embeddings of shape [B, D].

        - normalize=True: returns L2-normalized embeddings for cosine similarity.
        - normalize=False: returns pooled raw embeddings for downstream feature export.
        """
        batch = self._preprocess_numpy_batch(clips_uint8)

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.amp,
        ):
            outputs = self.encoder(**batch)
            tokens = outputs.last_hidden_state  # [B, N, D]

            if self.pooling == "gap":
                emb = tokens.mean(dim=1)
            else:
                assert self.pooler is not None
                emb = self.pooler(tokens)

        emb = emb.float()
        if normalize:
            emb = F.normalize(emb, dim=-1)
        return emb

    @torch.inference_mode()
    def encode_clip_pairs(
        self,
        real_clips_uint8: List[np.ndarray],
        skeleton_clips_uint8: List[np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            real_emb_raw: [B, D] pooled embeddings before normalization
            skel_emb_raw: [B, D] pooled embeddings before normalization
            sim:          [B] cosine similarity computed on normalized embeddings
        """
        if len(real_clips_uint8) != len(skeleton_clips_uint8):
            raise ValueError("real and skeleton clip batches must have the same length")

        real_emb_raw = self.encode_clips(real_clips_uint8, normalize=False)
        skel_emb_raw = self.encode_clips(skeleton_clips_uint8, normalize=False)

        real_emb_norm = F.normalize(real_emb_raw, dim=-1)
        skel_emb_norm = F.normalize(skel_emb_raw, dim=-1)
        sim = F.cosine_similarity(real_emb_norm, skel_emb_norm, dim=-1)

        return real_emb_raw, skel_emb_raw, sim.float()

    @torch.inference_mode()
    def similarity_scores(
        self,
        real_clips_uint8: List[np.ndarray],
        skeleton_clips_uint8: List[np.ndarray],
    ) -> torch.Tensor:
        _, _, sim = self.encode_clip_pairs(real_clips_uint8, skeleton_clips_uint8)
        return sim

    @torch.inference_mode()
    def filter_pair(
        self,
        video_real: str,
        video_skeleton: str,
        threshold: float = 0.70,
        sampled_frames: int = 16,
        temporal_stride: int = 4,
        window_step: int = 16,
        batch_size: int = 8,
        output_embeddings_path: Optional[str] = None,
    ) -> FilterResult:
        reader_real = VideoReaderPyAV(video_real)
        reader_skel = VideoReaderPyAV(video_skeleton)

        num_frames = min(reader_real.num_frames, reader_skel.num_frames)

        windows = self.build_windows(
            num_frames=num_frames,
            sampled_frames=sampled_frames,
            temporal_stride=temporal_stride,
            window_step=window_step,
        )

        if not windows:
            raise RuntimeError(
                "Video too short for the requested clip sampling. "
                f"Need at least {(sampled_frames - 1) * temporal_stride + 1} decoded frames."
            )

        clip_scores: List[ClipScore] = []
        embedding_dict: Dict[str, Dict[str, torch.Tensor]] = {}

        for batch_start in range(0, len(windows), batch_size):
            batch_windows = windows[batch_start: batch_start + batch_size]

            real_batch = [reader_real.get_frames(idx) for _, _, idx in batch_windows]
            skel_batch = [reader_skel.get_frames(idx) for _, _, idx in batch_windows]

            real_emb_raw, skel_emb_raw, sims = self.encode_clip_pairs(real_batch, skel_batch)
            real_emb_raw = real_emb_raw.detach().cpu()
            skel_emb_raw = skel_emb_raw.detach().cpu()
            sims_list = sims.detach().cpu().tolist()

            for local_i, ((start_frame, end_frame, idx), sim) in enumerate(zip(batch_windows, sims_list)):
                clip_index = batch_start + local_i
                is_anomaly = bool(sim < threshold)

                clip_scores.append(
                    ClipScore(
                        clip_index=clip_index,
                        start_frame=int(start_frame),
                        end_frame_inclusive=int(end_frame),
                        sampled_indices=[int(x) for x in idx],
                        similarity=float(sim),
                        is_anomaly=is_anomaly,
                    )
                )

                embedding_dict[str(clip_index)] = {
                    "clip_index": torch.tensor(clip_index, dtype=torch.int64),
                    "start_frame": torch.tensor(start_frame, dtype=torch.int64),
                    "end_frame_inclusive": torch.tensor(end_frame, dtype=torch.int64),
                    "sampled_indices": torch.tensor(idx, dtype=torch.int64),
                    "real_embedding": real_emb_raw[local_i].clone(),
                    "skeleton_embedding": skel_emb_raw[local_i].clone(),
                }

        if output_embeddings_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_embeddings_path)), exist_ok=True)
            torch.save(embedding_dict, output_embeddings_path)
            print(f"\nSaved V-JEPA embeddings to: {output_embeddings_path}")

        anomaly_ratio = sum(1 for x in clip_scores if x.is_anomaly) / max(1, len(clip_scores))
        sample_span = sampled_frames * temporal_stride

        return FilterResult(
            video_real=video_real,
            video_skeleton=video_skeleton,
            model_name=self.model_name,
            threshold=float(threshold),
            sample_span=int(sample_span),
            sampled_frames=int(sampled_frames),
            temporal_stride=int(temporal_stride),
            window_step=int(window_step),
            pooling=self.pooling,
            total_clips=len(clip_scores),
            anomaly_ratio=float(anomaly_ratio),
            clip_scores=clip_scores,
        )


# -----------------------------
# Serialization / reporting
# -----------------------------

def result_to_dict(result: FilterResult) -> Dict[str, Any]:
    data = asdict(result)
    data["anomaly_ratio"] = format_float(data["anomaly_ratio"], 6)
    for item in data["clip_scores"]:
        item["similarity"] = format_float(item["similarity"], 6)
    return data


def print_summary(result: FilterResult) -> None:
    print("=" * 88)
    print("Semantic Video Filtering Result")
    print("=" * 88)
    print(f"real video      : {result.video_real}")
    print(f"skeleton video  : {result.video_skeleton}")
    print(f"model           : {result.model_name}")
    print(f"pooling         : {result.pooling}")
    print(f"threshold       : {result.threshold:.4f}")
    print(f"sample span     : {result.sample_span} real frames")
    print(f"clip sampling   : {result.sampled_frames} frames, stride={result.temporal_stride}")
    print(f"window step     : {result.window_step}")
    print(f"total clips     : {result.total_clips}")
    print(f"anomaly ratio   : {result.anomaly_ratio:.4%}")
    print("-" * 88)
    print("Per-clip scores")
    print("-" * 88)
    for item in result.clip_scores:
        tag = "ANOMALY" if item.is_anomaly else "OK"
        print(
            f"[{item.clip_index:04d}] "
            f"frames {item.start_frame:06d}-{item.end_frame_inclusive:06d} | "
            f"sim={item.similarity:.4f} | {tag}"
        )


# -----------------------------
# CLI
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Semantic Video Filtering with V-JEPA 2"
    )
    parser.add_argument(
        "--video-real",
        type=str,
        required=True,
        help="Path to the real-person video.",
    )
    parser.add_argument(
        "--video-skeleton",
        type=str,
        required=True,
        help="Path to the rendered 3D skeleton / agent-token video.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/vjepa2-vitl-fpc64-256",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Cosine similarity threshold below which a clip is flagged as anomaly.",
    )
    parser.add_argument(
        "--sampled-frames",
        type=int,
        default=16,
        help="Number of sampled frames per clip.",
    )
    parser.add_argument(
        "--temporal-stride",
        type=int,
        default=4,
        help="Temporal stride between sampled frames.",
    )
    parser.add_argument(
        "--window-step",
        type=int,
        default=16,
        help="Sliding window step in decoded-frame units.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["gap", "attn"],
        default="gap",
        help="Clip embedding aggregation method.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto | cuda | cpu | cuda:0 ...",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision on CUDA.",
    )
    parser.add_argument(
        "--force-size",
        type=int,
        default=None,
        help=(
            "Optional manual resize override (e.g. 224). "
            "Leave unset to use the official HF processor config."
        ),
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa"],
        help="Attention backend for the HF model.",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile for the encoder where supported.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save the result JSON.",
    )
    parser.add_argument(
        "--output-embeddings",
        type=str,
        default="outputs/vjepa_embeddings.pt",
        help="Path to save pooled V-JEPA embeddings as a .pt dictionary.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)

    if args.sampled_frames != 16 or args.temporal_stride != 4:
        print(
            "[warning] You changed the default clip sampling. "
            "The requested V-JEPA setup in this pipeline is 16 sampled frames with stride=4."
        )

    vf = VJEPAFilter(
        model_name=args.model_name,
        pooling=args.pooling,
        device=args.device,
        amp=args.amp,
        force_size=args.force_size,
        attn_implementation=args.attn_implementation,
        use_torch_compile=args.torch_compile,
    )

    result = vf.filter_pair(
        video_real=args.video_real,
        video_skeleton=args.video_skeleton,
        threshold=args.threshold,
        sampled_frames=args.sampled_frames,
        temporal_stride=args.temporal_stride,
        window_step=args.window_step,
        batch_size=args.batch_size,
        output_embeddings_path=args.output_embeddings,
    )

    print_summary(result)

    if args.output_json is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result_to_dict(result), f, ensure_ascii=False, indent=2)
        print(f"\nSaved JSON result to: {args.output_json}")


if __name__ == "__main__":
    main()
