# FineVideo-VLA Dataset Pipeline

This repository contains the **complete pipeline** for building the FineVideo-VLA pretraining dataset (~25B tokens) from HuggingFace's [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo) dataset (~40K YouTube videos). The output is a Megatron-LM `.bin/.idx` token file.

There are two parallel branches that produce different token types and are merged at the end:

| Branch | What it produces | Entry point |
|--------|-----------------|-------------|
| **Prototype pipeline** (Step A) | Seed2 + Cosmos + AVC-LM video tokens | `prototype_pipeline/pipeline.py` |
| **3D pose pipeline** (Steps B–F) | Agent tokens (3D human pose, 256 uint8 per 8-frame chunk) | `pipeline/phase1_hrnet_gpu.py` … |

Each video activity in the final dataset produces an interleaved token sequence:
```
USER: <activity_description> [Speech: ...]  ASSISTANT:
  <seed2> <seed2_N> ... </seed2>       # 1 FPS semantic keyframe     (vocab: 8192)
  <cosmos> <cosmos_N> ... </cosmos>    # every 8 frames, spatial     (vocab: 64000)
  <avc_lm> <avclm_N> ... </avc_lm>    # every 8 frames, H.264 BPE   (vocab: 8192)
  <agent> 0 127 255 ... </agent>       # every 8 frames, 3D pose     (vocab: 256)
```
After flattening, `<tag> N </tag>` becomes `<tag_N>` — each is a single vocabulary token.

---

## Repository Layout

```
├── prototype_pipeline/         # Step A: Seed2, Cosmos, AVC-LM tokenization
│   ├── pipeline.py                 ← main entry point (read this first)
│   ├── pipeline_1gpu.py            single-GPU debug version
│   ├── submit_official.sbatch      SLURM job: 40 nodes × 4 GPU
│   ├── submit_demo.sbatch          SLURM job: 1 node demo run
│   ├── cosmos_tokenizer/           Cosmos tokenizer source
│   ├── seed2/                      Seed2 tokenizer source + vocab
│   ├── avc_lm_v2/                  AVC-LM BPE vocab (used by pipeline.py)
│   ├── pretrained_ckpts/           Cosmos model configs (weights gitignored)
│   └── README.md                   per-file descriptions
│
├── pipeline/                   # Steps B–F: 3D pose → agent tokens
│   ├── phase1_hrnet_gpu.py         2D pose estimation (HRNet)
│   ├── phase2_motionbert_gpu.py    3D pose lifting (MotionBERT)
│   ├── phase2_5_resample_30fps.py  Resample native-fps poses to 30fps
│   ├── phase3_kinematics_processor.py  Signal filter + kinematics
│   ├── phase4_yolo_cleaner.py      YOLO person-presence filter
│   ├── phase5_interpolation_tokenizer.py  PCHIP + uint8 quantisation
│   ├── phase6_macro_filter_dataset.py     Quality filter
│   ├── merge_agent_tokens.py       Inject <agent> blocks into training_ready
│   └── flatten_dataset.py          Hierarchical JSON → Megatron flat JSONL
│
├── slurm/                      # SLURM submit scripts for the 3D pose branch
│   ├── submit_hrnet.sh
│   ├── submit_motionbert.sh
│   ├── submit_phase2_5.sh
│   ├── submit_kinematics.sh
│   ├── submit_yolo.sh
│   ├── submit_beast.sh
│   ├── submit_phase5_resume.sh
│   └── submit_integration.sh
│
├── tools/                      # Standalone utilities
│   ├── expand_vocab.py             Extend GPT-NeoX-20b vocab with VLA tokens
│   ├── decode_agent_tokens.py      Decode agent uint8 tokens → 3D poses
│   ├── extract_fps.py              Read native fps for all videos → fps_lookup.json
│   ├── rebuild_parquet_fps.py      Rebuild parquet shards with 30fps poses + fps column
│   ├── upload_3d_npy_to_hf.py      Upload 3d_npy/ arrays as parquet shards to HuggingFace
│   ├── upload_parquet_hf.py        Upload rebuilt parquet shards to HuggingFace (resume-safe)
│   ├── check_vocab.py
│   ├── check_flattened_data.py
│   ├── extract_sample.py
│   ├── fetch_data.py
│   └── render_filtered_skeleton.py
│
├── dev/                        # Single-video dev/demo scripts
├── envs/                       # Conda environment YAML specs
├── vocab/                      # vocab.json (GPT-NeoX-20b) + vocab_expanded.json
├── samples/                    # decoded_agent_sample.json
├── setup_motionbert.sh         # Activate env for Steps B–F
└── setup_hrnet_gpu.sh          # Activate env for Step A (HRNet)
```

---

## End-to-End Pipeline

```
HuggingFace FineVideo disk
/e/scratch/reformo/nguyen38/finevideo_disk

──────────────── BRANCH A: Video Tokens ────────────────────────────────────

Step A   prototype_pipeline/pipeline.py   (submit_official.sbatch, 40 nodes × 4 GPU)
         Extracts frames at 30fps; tokenizes every activity segment with
         Seed2 (1fps), Cosmos (8-frame), and AVC-LM (8-frame).
         → /e/scratch/reformo/nguyen38/FineVideo-VLA/training_ready_rank_*.jsonl
           Each activity has a flat video_tokens string and time_range_sec.

──────────────── BRANCH B: 3D Pose / Agent Tokens ──────────────────────────

Step B   pipeline/phase1_hrnet_gpu.py          (slurm/submit_hrnet.sh)
         HRNet + Faster R-CNN 2D pose estimation
         → outputs/2d_json/{video_id}_2d.json

Step C   pipeline/phase2_motionbert_gpu.py     (slurm/submit_motionbert.sh)
         MotionBERT 3D lifting (native video fps)
         → outputs/3d_npy/{video_id}.npy

Step D   pipeline/phase2_5_resample_30fps.py   (slurm/submit_phase2_5.sh)
         Resample native-fps 3D poses to 30fps via linear interpolation.
         Required so Steps E–F share the same time grid as Branch A.
         → outputs/3d_npy_30fps/{video_id}.npy

Step E   pipeline/phase3_kinematics_processor.py  (slurm/submit_kinematics.sh)
         Signal filter, bone normalisation, kinematics (pos/vel/acc)
         → outputs/states_jsonl/{video_id}_states.jsonl   (windows × 8 × 153)
           153 = 17 joints × 3 dims × 3 kinematics

Step F   pipeline/phase4_yolo_cleaner.py       (slurm/submit_yolo.sh)
         Drop 8-frame windows where ≥ 4 frames have no detected person.
         Windows that pass keep their original window_id (frame offset).
         → outputs/yolo_cleaned/{video_id}_cleaned.jsonl

Step G   pipeline/phase5_interpolation_tokenizer.py  (slurm/submit_beast.sh)
         Adaptive PCHIP interpolation + uint8 quantisation → 256 tokens/window
         → outputs/agent_tokens/{video_id}_tokens.jsonl
         Resume incomplete run: slurm/submit_phase5_resume.sh

Step H   pipeline/phase6_macro_filter_dataset.py
         Macro-level quality filter (min token yield per video)
         → outputs/agent_tokens_filtered/

──────────────── MERGE ──────────────────────────────────────────────────────

Step I   pipeline/merge_agent_tokens.py        (slurm/submit_integration.sh)
         Injects <agent> blocks after each <avc_lm> block in training_ready files.
         → /e/scratch/reformo/nguyen38/FineVideo-VLA/final_dataset/final_vla_rank_*.jsonl

Step J   pipeline/flatten_dataset.py
         Hierarchical JSON → Megatron flat JSONL
         → /p/data1/mmlaion/shared/vla/vla_25b/flat_*.jsonl
```

Steps B–H run from `3d-human-pose/` as working directory. `outputs/` is a symlink to `/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs/`.

---

## Token Alignment

All four token types share the same **30fps frame grid**. Step A resamples videos to exactly 30fps via ffmpeg; Step D resamples 3D poses to 30fps. This means:

| Token type | Fires at | Covers frames | Timestamp formula |
|------------|----------|---------------|-------------------|
| Seed2 | every 30 frames | single frame | `activity_start + k × 1.0 s` |
| Cosmos | every 8 frames | frames `[8k, 8k+7]` | `activity_start + k × 8/30 s` |
| AVC-LM | every 8 frames | frames `[8k, 8k+7]` | `activity_start + k × 8/30 s` |
| Agent | every 8 frames | frames `[8k, 8k+7]` | `activity_start + window_id × 8/30 s` |

`activity_start` is stored in each activity's `time_range_sec` field in the JSONL.

**Seed2/Cosmos/AVC-LM tokens are contiguous** — every frame of every activity is tokenized with no gaps. Timestamps are not stored explicitly but are recoverable by position math above.

**Agent tokens may be non-contiguous** — Phase 4 (YOLO) drops junk windows. Each surviving record stores `window_id` (original frame offset from video start), so the exact timestamp and video segment are always recoverable.

To extract the video segment for any token block:
```bash
ffmpeg -ss <timestamp> -t <duration> -i video.mp4 segment.mp4
# Cosmos/AVC-LM: duration = 8/30 ≈ 0.267s
# Seed2: duration = 1/30 ≈ 0.033s (single frame)
```

---

## Environment Setup

**Step A — prototype pipeline** (Seed2 + Cosmos + AVC-LM):
```bash
module --force purge
module load Stages/2025 GCC/13.3.0 Python/3.12.3 CUDA/12 PyTorch/2.5.1 torchvision/0.20.1
source /e/project1/reformo/nguyen38/env_stable_vla/bin/activate
export FFMPEG_PATH=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
```
Model weights are **not committed** — download once with `prototype_pipeline/download.py` (requires `HF_TOKEN`).

**Steps B–H — 3D pose pipeline** (HRNet, MotionBERT, YOLO, tokenizer):
```bash
source setup_motionbert.sh    # activates env_motion_final/
```

**Step B only — HRNet** (different env):
```bash
source setup_hrnet_gpu.sh
```

Environment YAML specs are in `envs/`.

---

## Agent Token Format

Each 8-frame chunk is encoded as **256 `uint8` tokens**:

| Index | Content | Encoding |
|-------|---------|----------|
| `0` | scale (1 token) | `clip(scale / 2.0 * 255, 0, 255)` |
| `1–51` | anchor — frame-0 pose, root-centred (51 tokens) | `clip((x + 2) / 4 * 255, 0, 255)` per dim |
| `52–255` | motion control points (204 tokens) | `uint8` from Phase 5 |

```
token[0]       scale   decode: t / 255.0 * 2.0          (metres, max 2.0 m)
tokens[1–51]   anchor  decode: t / 255.0 * 4.0 - 2.0    (metres, root-centred, ±2.0 m)
tokens[52–255] motion  decode: t / 127.5 - 1.0           (normalised [-1,1], 4 CPs × 17 joints × 3)
```

Reconstruction: `cp_absolute = motion_CPs * scale + anchor`, then PCHIP interpolation over `t ∈ [0, 1]`.

---

## Cluster (JUPITER — `booster` partition)

- **Hardware:** GH200 nodes, 4 GPUs per node, 288 CPU cores per node
- **Account:** `reformo`
- GPU assignment: `SLURM_LOCALID` → `cuda:{local_id}`
- File partitioning: `SLURM_ARRAY_TASK_ID` + `SLURM_ARRAY_TASK_COUNT`
- All phases are **safe to re-run** — each script skips already-completed output files

---

## Key Data Paths

| What | Path |
|------|------|
| FineVideo HF dataset | `/e/scratch/reformo/nguyen38/finevideo_disk` |
| Intermediate pose outputs | `/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs/` |
| `training_ready` JSONL (Step A output) | `/e/scratch/reformo/nguyen38/FineVideo-VLA/training_ready_rank_*.jsonl` |
| Final merged JSONL | `/e/scratch/reformo/nguyen38/FineVideo-VLA/final_dataset/final_vla_rank_*.jsonl` |
| Flat Megatron target | `/p/data1/mmlaion/shared/vla/vla_25b/` |

---

## Vocabulary

`tools/expand_vocab.py` extends the GPT-NeoX-20b base (`vocab/vocab.json`) with:

| Token range | Count |
|-------------|-------|
| `<agent_0>` … `<agent_255>` | 256 |
| `<avclm_0>` … `<avclm_8191>` | 8192 |
| `<seed2_0>` … `<seed2_8191>` | 8192 |
| `<cosmos_0>` … `<cosmos_63999>` | 64000 |
| Wrapper tags (`<agent>`, `</agent>`, …) | 8 |

Output: `vocab/vocab_expanded.json`.
```bash
python tools/check_vocab.py   # verify vocab_size (rounds to nearest 128 for Megatron)
```

---

## Useful Commands

```bash
# Decode a random agent token block from a final_vla file
python tools/decode_agent_tokens.py --seed 42

# Count token density across all training_ready shards
python prototype_pipeline/count_tokens.py

# Check agent token coverage in a merged file
python3 -c "
import json
with_agent = without_agent = 0
for line in open('/e/scratch/reformo/nguyen38/FineVideo-VLA/final_dataset/final_vla_rank_0.jsonl'):
    d = json.loads(line)
    has = any('<agent>' in a.get('video_tokens','') for s in d['scenes'] for a in s['activities'])
    with_agent += has; without_agent += not has
print(with_agent, 'with agent /', without_agent, 'without')
"

# Sanity-check a flat Megatron dataset
python tools/check_flattened_data.py

# Render a skeleton-only video from a states JSONL
python tools/render_filtered_skeleton.py \
    --video-real videos/sample.mp4 \
    --jsonl outputs/states_jsonl/sample_states.jsonl \
    --output outputs/skeleton.mp4
```

---

## Third-Party Dependencies

- **[MotionBERT](https://github.com/Walter0807/MotionBERT)** — 3D pose lifting (Step C). Checkpoint: `third_party/MotionBERT/checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin`
- **HRNet** — model weights in `hrnet_storage/` (gitignored)
- **YOLO** — `yolo26n.pt` expected in working directory for Step F
- **Cosmos tokenizer weights** — download via `prototype_pipeline/download.py`; stored in `prototype_pipeline/pretrained_ckpts/` (gitignored)
- **Seed2 model weights** — `prototype_pipeline/seed2/model.safetensors` and `ae.safetensors` (gitignored)
