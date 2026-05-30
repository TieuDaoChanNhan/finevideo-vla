# 3D Human Pose — VLA Agent Token Pipeline

This repository implements the **3D pose branch** of the FineVideo-VLA dataset pipeline. It takes HuggingFace's [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo) dataset (~40K YouTube videos) and produces compact **256-token agent blocks** — one per 8-frame chunk — that are injected into the main VLA training JSONL alongside Seed2, Cosmos, and AVC-LM video tokens.

The final dataset target is ~25B tokens for Megatron-LM pretraining.

---

## Agent Token Format

Each 8-frame chunk is encoded as **256 `uint8` tokens** laid out as:

| Index | Content | Encoding |
|-------|---------|----------|
| `0` | scale (1 token) | `clip(scale / 2.0 * 255, 0, 255)` |
| `1–51` | anchor — frame-0 pose, root-centred (51 tokens) | `clip((x + 2) / 4 * 255, 0, 255)` per dim |
| `52–255` | motion control points (204 tokens) | existing `uint8` from Phase 5 |

```
token[0]       scale   uint8  decode: t / 255.0 * 2.0          (metres, max 2.0 m)
tokens[1–51]   anchor  uint8  decode: t / 255.0 * 4.0 - 2.0    (metres, root-centred, ±2.0 m)
tokens[52–255] motion  uint8  decode: t / 127.5 - 1.0           (normalised [-1,1], 4 CPs × 17 joints × 3)
```

Reconstruction: `cp_absolute = motion_CPs * scale + anchor`, then PCHIP interpolation over `t ∈ [0, 1]`.

---

## Repository Layout

```
3d-human-pose/
├── pipeline/               # Main processing scripts (Phases 1–6 + integration)
│   ├── phase1_hrnet_gpu.py
│   ├── phase2_motionbert_gpu.py
│   ├── phase3_kinematics_processor.py
│   ├── phase4_yolo_cleaner.py
│   ├── phase5_interpolation_tokenizer.py
│   ├── phase6_macro_filter_dataset.py
│   ├── merge_agent_tokens.py
│   └── flatten_dataset.py
├── slurm/                  # SLURM job submission scripts
│   ├── submit_hrnet.sh
│   ├── submit_motionbert.sh
│   ├── submit_kinematics.sh
│   ├── submit_yolo.sh
│   ├── submit_beast.sh
│   ├── submit_phase5_resume.sh
│   └── submit_integration.sh
├── tools/                  # Standalone utilities
│   ├── expand_vocab.py
│   ├── decode_agent_tokens.py
│   ├── check_vocab.py
│   ├── check_flattened_data.py
│   ├── extract_sample.py
│   ├── fetch_data.py
│   └── render_filtered_skeleton.py
├── dev/                    # Single-video dev/demo scripts
│   ├── pipeline.sh
│   ├── cut_video.py
│   ├── check_states.py
│   └── visualize_demo.py
├── envs/                   # Conda environment specs
│   ├── env_motion_final.yaml
│   ├── env_hrnet_datasets_v1.yaml
│   ├── environment.yml
│   └── base_miniforge3.yaml
├── vocab/                  # Vocabulary files
│   ├── vocab.json              (GPT-NeoX-20b base)
│   └── vocab_expanded.json     (extended with VLA tokens)
├── samples/
│   └── decoded_agent_sample.json
├── tests/
├── setup_motionbert.sh     # Environment activation (Phases 2–6)
└── setup_hrnet_gpu.sh      # Environment activation (Phase 1)
```

---

## End-to-End Pipeline

```
HuggingFace FineVideo disk
/e/scratch/reformo/nguyen38/finevideo_disk

Phase 1  pipeline/phase1_hrnet_gpu.py          (slurm/submit_hrnet.sh)
         HRNet + Faster R-CNN 2D pose estimation
         → outputs/2d_json/{video_id}_2d.json

Phase 2  pipeline/phase2_motionbert_gpu.py     (slurm/submit_motionbert.sh)
         MotionBERT 3D lifting
         → outputs/3d_npy/{video_id}.npy

Phase 3  pipeline/phase3_kinematics_processor.py  (slurm/submit_kinematics.sh)
         Signal filtering, bone normalisation, kinematics, stiff-leg correction
         → outputs/states_jsonl/{video_id}_states.jsonl   (windows × 8 × 17 × 3)

Phase 4  pipeline/phase4_yolo_cleaner.py       (slurm/submit_yolo.sh)
         YOLO-based person-presence filtering
         → outputs/yolo_cleaned/{video_id}_cleaned.jsonl

Phase 5  pipeline/phase5_interpolation_tokenizer.py  (slurm/submit_beast.sh)
         Adaptive PCHIP interpolation + uint8 quantisation
         → outputs/agent_tokens/{video_id}_tokens.jsonl
         Resume: slurm/submit_phase5_resume.sh

Phase 6  pipeline/phase6_macro_filter_dataset.py
         Macro-level quality filter (min tokens per video, yield rate)
         → outputs/agent_tokens_filtered/

Merge    pipeline/merge_agent_tokens.py        (slurm/submit_integration.sh)
         Injects <agent> blocks after each <avc_lm> block
         → final_dataset/final_vla_rank_*.jsonl

Flatten  pipeline/flatten_dataset.py
         Hierarchical JSON → Megatron flat JSONL
         → /p/data1/mmlaion/shared/vla/vla_25b/flat_*.jsonl
```

All phases run from `3d-human-pose/` as the working directory. `outputs/` is a symlink to `/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs/`.

---

## Environment Setup

There are two separate conda environments — **never mix them**.

**Phase 1** (HRNet + MMDet/MMPose):
```bash
source setup_hrnet_gpu.sh
```

**Phases 2–6** (MotionBERT, YOLO, tokeniser, merge, flatten):
```bash
source setup_motionbert.sh
# conda env: env_motion_final/
```

Environment YAML specs are in `envs/`.

---

## Cluster (JUPITER — `booster` partition)

- **Hardware:** GH200 nodes, 4 GPUs per node, 288 CPU cores per node
- **Account:** `reformo`
- GPU assignment: `SLURM_LOCALID` → `cuda:{local_id}`
- File partitioning: `SLURM_ARRAY_TASK_ID` + `SLURM_ARRAY_TASK_COUNT` (modulo split)
- CPU-parallel phases use `for i in ...; do python ... & done; wait`

All phases are **safe to re-run** — each script checks for existing output files and skips them.

---

## Key Data Paths

| What | Path |
|------|------|
| FineVideo HF dataset | `/e/scratch/reformo/nguyen38/finevideo_disk` |
| Intermediate pose outputs | `/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs/` |
| `training_ready` JSONL | `/e/scratch/reformo/nguyen38/FineVideo-VLA/training_ready_rank_*.jsonl` |
| Final merged JSONL | `/e/scratch/reformo/nguyen38/FineVideo-VLA/final_dataset/final_vla_rank_*.jsonl` |
| Flat Megatron target | `/p/data1/mmlaion/shared/vla/vla_25b/` |

---

## Vocabulary

`tools/expand_vocab.py` extends the GPT-NeoX-20b base vocabulary (`vocab/vocab.json`) with:

| Token range | Count |
|-------------|-------|
| `<agent_0>` … `<agent_255>` | 256 |
| `<avclm_0>` … `<avclm_8191>` | 8192 |
| `<seed2_0>` … `<seed2_8191>` | 8192 |
| `<cosmos_0>` … `<cosmos_63999>` | 64000 |
| Wrapper tags (`<agent>`, `</agent>`, etc.) | 8 |

Output: `vocab/vocab_expanded.json`. Check required Megatron `vocab_size` (rounded to nearest multiple of 128):
```bash
python tools/check_vocab.py
```

---

## Useful Commands

```bash
# Decode a random agent token block from a final_vla file
python tools/decode_agent_tokens.py --seed 42

# Sanity-check a flat Megatron dataset
python tools/check_flattened_data.py

# Render a skeleton-only video from a states JSONL
python tools/render_filtered_skeleton.py \
    --video-real videos/sample.mp4 \
    --jsonl outputs/states_jsonl/sample_states.jsonl \
    --output outputs/skeleton.mp4

# Check agent token coverage in a merged file
python3 -c "
import json
with_agent = without_agent = 0
for line in open('path/to/final_vla_rank_0.jsonl'):
    d = json.loads(line)
    has = any('<agent>' in a.get('video_tokens','') for s in d['scenes'] for a in s['activities'])
    with_agent += has; without_agent += not has
print(with_agent, 'with agent /', without_agent, 'without')
"
```

---

## Third-Party Dependencies

- **[MotionBERT](https://github.com/Walter0807/MotionBERT)** — 3D pose lifting (Phase 2). Place checkpoint at `third_party/MotionBERT/checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin`.
- **HRNet** model weights — place configs and `.pth` files in `hrnet_storage/` (gitignored).
- **YOLO** — `yolo26n.pt` expected in the working directory for Phase 4.
