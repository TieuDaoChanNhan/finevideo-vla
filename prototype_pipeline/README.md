# prototype_pipeline — Seed2 / Cosmos / AVC-LM Tokenization (Step A)

This folder implements **Step A** of the FineVideo-VLA pipeline: it reads videos
from the HuggingFace FineVideo dataset and produces the interleaved
`<seed2>`, `<cosmos>`, and `<avc_lm>` token blocks written into
`training_ready_rank_*.jsonl`.

These files are the input for both the legacy merge (`pipeline/merge_agent_tokens.py`)
and the current XYZT merge (`pipeline/merge_xyzt_tokens.py`), which inject 3D pose
tokens from the pose pipeline branch.

---

## The one file you need to read

**`pipeline.py`** — the production multi-GPU tokenization script.

It does everything in one pass per activity segment:

1. Extracts frames at exactly **30 FPS** via ffmpeg.
2. Iterates frame-by-frame, building three token streams in parallel:
   - **Seed2** (1 fps semantic keyframe) — fires every 30 frames via `idx % 30 == 0`
   - **Cosmos** (8-frame spatial tokens) — fires when the rolling 8-frame buffer fills
   - **AVC-LM** (8-frame H.264 BPE tokens) — fires together with Cosmos, same 8-frame window
3. Emits the token blocks in frame order:
   ```
   <seed2> ... </seed2>  ← every 1 s
   <cosmos> ... </cosmos> <avc_lm> ... </avc_lm>  ← every 8 frames (≈0.267 s)
   <cosmos> ... </cosmos> <avc_lm> ... </avc_lm>
   ...
   ```

**Token alignment**: because all three tokenizers share the same 30 fps frame grid,
every `<cosmos>`/`<avc_lm>` chunk k covers frames `[8k, 8k+7]` = `[8k/30 s, (8k+7)/30 s]`.
The XYZT agent tokens from the 3D pose branch (Steps B–G) use the same 8-frame window,
so they map 1-to-1 with each `<cosmos>`/`<avc_lm>` block — no separate timestamps needed.

To run on the cluster:
```bash
sbatch submit_official.sbatch   # 40 nodes × 4 GPU
```

---

## Output format

Each `training_ready_rank_*.jsonl` file contains one JSON object per video. Structure:

```json
{
  "video_id": "abc123",
  "global_context": "A video about ...",
  "scenes": [
    {
      "scene_title": "...",
      "scene_thematic": "...",
      "scene_mood": "...",
      "activities": [
        {
          "text_prompt": "...",
          "speech_transcript": "...",
          "time_range_sec": [0.0, 12.5],
          "video_tokens": "<seed2> 3758 2157 </seed2> <cosmos> 58567 ... </cosmos> <avc_lm> 100 200 </avc_lm> ..."
        }
      ]
    }
  ]
}
```

160 shards total, one per SLURM rank.

---

## Other files

| File | Purpose |
|------|---------|
| `pipeline_1gpu.py` | Single-GPU version of `pipeline.py` — useful for debugging |
| `video_pipeline.py` | Earlier prototype; video-only, no Seed2 |
| `image_pipeline.py` | Image-only tokenization (not used in production) |
| `pipeline_benchmark.py` / `benchmark_pipeline.py` | Throughput benchmarking scripts |
| `convert_jsonl.py` | Converts between JSONL output formats |
| `count_tokens.py` | Counts token density in output JSONL files |
| `upload_hf.py` | Uploads completed output shards to HuggingFace |
| `download.py` | Downloads Cosmos model weights from HuggingFace (run once) |
| `delete_files.py` | Removes files from a HuggingFace repo |
| `fix.py` | One-off post-processing fix script |
| `submit_demo.sbatch` | SLURM script for a small demo run (1 node) |
| `submit_official.sbatch` | SLURM script for the full production run (40 nodes) |

## Tokenizer dependencies

| Directory | What it is |
|-----------|-----------|
| `seed2/` | Seed2 tokenizer source + vocab (weights downloaded separately to `seed2/model.safetensors`) |
| `cosmos_tokenizer/` | Cosmos tokenizer source code (weights in `pretrained_ckpts/`, downloaded via `download.py`) |
| `avc-lm/` | AVC-LM BPE tokenizer vocab (original, 8192 tokens) |
| `avc_lm_v2/` | AVC-LM BPE tokenizer vocab v2 — used by `pipeline.py` |
| `jpeg_tokenizer/` | JPEG BPE vocab — experimental, not used in production |
| `pretrained_ckpts/` | Cosmos model configs; `.jit` weight files are gitignored, download with `download.py` |

## Environment setup

```bash
module --force purge
module load Stages/2025 GCC/13.3.0 Python/3.12.3 CUDA/12 PyTorch/2.5.1 torchvision/0.20.1
source /e/project1/reformo/nguyen38/env_stable_vla/bin/activate
export FFMPEG_PATH=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
```

Model weights are **not committed** — download once:
```bash
export HF_TOKEN='hf_...'
python download.py
```
