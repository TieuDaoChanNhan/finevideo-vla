# FineVideo-VLA Dataset Pipeline

This repository contains the **complete pipeline** for building the FineVideo-VLA pretraining dataset (~25B tokens) from HuggingFace's [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo) dataset (~40K YouTube videos). The output is a Megatron-LM-ready flat JSONL dataset.

There are two parallel branches that produce different token types and are merged at the end:

| Branch | What it produces | Entry point |
|--------|-----------------|-------------|
| **Prototype pipeline** (Step A) | Seed2 + Cosmos + AVC-LM video tokens | `prototype_pipeline/pipeline.py` |
| **3D pose pipeline** (Steps B–F) | Per-joint XYZT tokens (17 joints × 3 dims, 30fps) | `pipeline/phase1_hrnet_gpu.py` … `pipeline/phase5b_xyzt_tokenizer.py` |

Each video activity in the final dataset produces an interleaved token sequence:
```
USER: <activity_description> [Speech: ...]  ASSISTANT:
  <seed2> <seed2_N> ... </seed2>       # 1 FPS semantic keyframe     (vocab: 8192)
  <cosmos> <cosmos_N> ... </cosmos>    # every 8 frames, spatial     (vocab: 64000)
  <avc_lm> <avclm_N> ... </avc_lm>    # every 8 frames, H.264 BPE   (vocab: 8192)
  <agent> <fps_30> <joint_0_x_N> <joint_0_y_N> ... </agent>  # every 8 frames, 3D pose
```

After flattening, wrapper tags are removed and individual tokens become single vocabulary entries:
```
<seed2_3758> <seed2_2157> <cosmos_58567> <avclm_100> <fps_30> <joint_0_x_127> <joint_0_y_200> ...
```

---

## HuggingFace Datasets

| Dataset | Description | Records | Size |
|---------|-------------|---------|------|
| [EmpathicRobotics/FineVideo-VLA-flattened](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-VLA-flattened) | Flattened Megatron-LM JSONL (final output, ready for pretraining) | 69,844 | ~24 GB |
| [EmpathicRobotics/FineVideo-VLA-Agent](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-VLA-Agent) | Per-joint XYZT merged dataset (pre-flattening, hierarchical) | — | — |

Both are split 152 train / 8 test shards (95/5, seed 42), gzip compressed.

---

## Repository Layout

```
├── prototype_pipeline/         # Step A: Seed2, Cosmos, AVC-LM tokenization
│   ├── pipeline.py                 ← main entry point (read this first)
│   ├── pipeline_1gpu.py            single-GPU debug version
│   ├── submit_official.sbatch      SLURM job: 40 nodes × 4 GPU
│   ├── cosmos_tokenizer/           Cosmos tokenizer source
│   ├── seed2/                      Seed2 tokenizer source + vocab
│   ├── avc_lm_v2/                  AVC-LM BPE vocab (used by pipeline.py)
│   ├── pretrained_ckpts/           Cosmos model configs (weights gitignored)
│   └── README.md
│
├── pipeline/                   # Steps B–G: 3D pose → XYZT tokens → merge → flatten
│   ├── phase1_hrnet_gpu.py         2D pose estimation (HRNet)
│   ├── phase2_motionbert_gpu.py    3D pose lifting (MotionBERT)
│   ├── phase2_5_resample_30fps.py  Resample native-fps poses to 30fps
│   ├── phase3_kinematics_processor.py  Signal filter + kinematics
│   ├── phase4_yolo_cleaner.py      YOLO person-presence filter
│   ├── phase5b_xyzt_tokenizer.py   Per-joint XYZ quantisation (409 tokens/window)
│   ├── phase5_interpolation_tokenizer.py  Legacy: PCHIP + uint8 (256 tokens/window)
│   ├── merge_xyzt_tokens.py        Inject XYZT <agent> blocks into training_ready
│   ├── merge_agent_tokens.py       Legacy: inject opaque <agent> blocks
│   ├── flatten_dataset.py          Legacy: flatten for old token format
│   └── README.md
│
├── tools/                      # Standalone utilities (see tools/README.md)
│   ├── flatten.py                  Flatten XYZT merged → Megatron JSONL (with augmentation)
│   ├── expand_vocab.py             Extend GPT-NeoX-20b vocab with all VLA tokens
│   ├── upload_flattened_hf.py      Upload flattened dataset to HuggingFace
│   ├── upload_vla_agent_hf.py      Upload XYZT agent dataset to HuggingFace
│   ├── cleanup_flattened_hf.py     Remove old files from HF repo
│   ├── decode_agent_tokens.py      Decode agent uint8 tokens → 3D poses
│   ├── check_flattened_data.py     Validate flattened Megatron files
│   ├── check_vocab.py              Verify expanded vocab
│   ├── extract_fps.py              Read native fps for all videos
│   ├── extract_sample.py           Extract sample records
│   ├── fetch_data.py               Fetch video data from HuggingFace
│   ├── rebuild_parquet_fps.py      Rebuild parquet shards with 30fps poses
│   ├── render_filtered_skeleton.py Render skeleton overlay video
│   ├── upload_3d_npy_to_hf.py      Upload 3d_npy/ arrays as parquet
│   ├── upload_parquet_hf.py        Upload parquet shards to HuggingFace
│   └── README.md
│
├── slurm/                      # SLURM submit scripts
│   ├── submit_hrnet.sh             Phase 1
│   ├── submit_motionbert.sh        Phase 2
│   ├── submit_phase2_5.sh          Phase 2.5
│   ├── submit_kinematics.sh        Phase 3
│   ├── submit_yolo.sh              Phase 4
│   ├── submit_phase5b.sh           Phase 5b (XYZT)
│   ├── submit_beast.sh             Phase 5 (legacy)
│   ├── submit_phase5_resume.sh     Phase 5 resume
│   ├── submit_merge_xyzt.sh        XYZT merge
│   └── submit_integration.sh       Legacy merge
│
├── dev/                        # Single-video dev/demo scripts
├── envs/                       # Conda environment YAML specs
├── vocab/                      # vocab.json (GPT-NeoX-20b) + vocab_expanded.json
├── samples/                    # Sample outputs for inspection
├── setup_motionbert.sh         # Activate env for Steps B–G
└── setup_hrnet_gpu.sh          # Activate env for Step B (HRNet only)
```

---

## End-to-End Pipeline

```
$DATA = /e/data1/datasets/playground/mmlaion/shared/nguyen38

──────────────── BRANCH A: Video Tokens ────────────────────────────────────

Step A   prototype_pipeline/pipeline.py   (submit_official.sbatch, 40 nodes × 4 GPU)
         Extracts frames at 30fps; tokenizes every activity segment with
         Seed2 (1fps), Cosmos (8-frame), and AVC-LM (8-frame).
         → $DATA/FineVideo-VLA/training_ready_rank_*.jsonl

──────────────── BRANCH B: 3D Pose / XYZT Tokens ──────────────────────────

Step B   pipeline/phase1_hrnet_gpu.py          (slurm/submit_hrnet.sh)
         HRNet + Faster R-CNN 2D pose estimation
         → outputs/2d_json/{video_id}_2d.json

Step C   pipeline/phase2_motionbert_gpu.py     (slurm/submit_motionbert.sh)
         MotionBERT 3D lifting (native video fps)
         → outputs/3d_npy/{video_id}.npy

Step D   pipeline/phase2_5_resample_30fps.py   (slurm/submit_phase2_5.sh)
         Resample native-fps 3D poses to 30fps via linear interpolation.
         Required so Steps E–G share the same time grid as Branch A.
         → outputs/3d_npy_30fps/{video_id}.npy

Step E   pipeline/phase3_kinematics_processor.py  (slurm/submit_kinematics.sh)
         Signal filter, bone normalisation, kinematics (pos/vel/acc)
         → outputs/states_jsonl/{video_id}_states.jsonl   (windows × 8 × 153)

Step F   pipeline/phase4_yolo_cleaner.py       (slurm/submit_yolo.sh)
         Drop 8-frame windows where ≥ 4 frames have no detected person.
         → outputs/yolo_cleaned_30fps/{video_id}_cleaned.jsonl

Step G   pipeline/phase5b_xyzt_tokenizer.py    (slurm/submit_phase5b.sh)
         Per-joint XYZ quantisation → 409 self-describing tokens/window
         → outputs/agent_tokens_xyzt/{video_id}_tokens.jsonl
         → outputs/agent_xyzt_npy/{video_id}_xyzt.npy

──────────────── MERGE + FLATTEN ───────────────────────────────────────────

Step H   pipeline/merge_xyzt_tokens.py         (slurm/submit_merge_xyzt.sh)
         Injects <agent> blocks (with per-joint tokens) after each <avc_lm>
         block in training_ready files.
         → $DATA/FineVideo-VLA/final_dataset_xyzt/final_vla_xyzt_rank_*.jsonl

Step I   tools/flatten.py                      (run on login node or SLURM)
         Hierarchical JSON → Megatron flat JSONL with data augmentation:
         synonym replacement, stopword dropout, sentence permutation,
         modality dropout (99% avc_lm, 90% cosmos, 0% seed2), and
         speech/token interleaving.
         → $DATA/flat_xyzt/flat_final_vla_xyzt_rank_*.jsonl

Step J   tools/upload_flattened_hf.py
         Compress + upload to EmpathicRobotics/FineVideo-VLA-flattened
```

Steps B–G run from `3d-human-pose/` as working directory. `outputs/` is a symlink to `$DATA/outputs/`.

---

## XYZT Token Format (current)

Each 8-frame chunk produces **409 self-describing tokens**:

```
<fps_30> <joint_0_x_127> <joint_0_y_200> <joint_0_z_143> <joint_1_x_130> ...
```

| Component | Count | Encoding |
|-----------|-------|----------|
| `<fps_N>` | 1 | Frame rate (always 30) |
| `<joint_J_d_V>` | 408 = 8 frames × 17 joints × 3 dims | `V = clip(round((v + 2.0) / 4.0 * 255), 0, 255)` |

- **Joint order** (H36M 17-joint): pelvis, r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle, spine, thorax, nose, head_top, l_shoulder, l_elbow, l_wrist, r_shoulder, r_elbow, r_wrist
- **Coordinate range**: [-2.0 m, +2.0 m], precision ~15.7 mm
- **Root-centred**: pelvis is always at origin [0, 0, 0]
- All poses are at 30 fps — no variable frame rate

### Legacy Agent Token Format (256 tokens)

The older `phase5_interpolation_tokenizer.py` produced 256 opaque uint8 tokens per chunk (scale + anchor + motion control points). This has been superseded by the per-joint XYZT format above.

---

## Token Alignment

All four token types share the same **30fps frame grid**:

| Token type | Fires at | Covers frames | Timestamp formula |
|------------|----------|---------------|-------------------|
| Seed2 | every 30 frames | single frame | `activity_start + k × 1.0 s` |
| Cosmos | every 8 frames | frames `[8k, 8k+7]` | `activity_start + k × 8/30 s` |
| AVC-LM | every 8 frames | frames `[8k, 8k+7]` | `activity_start + k × 8/30 s` |
| Agent (XYZT) | every 8 frames | frames `[8k, 8k+7]` | `activity_start + window_id × 8/30 s` |

**Agent tokens may be non-contiguous** — Phase F (YOLO) drops windows with no detected person. Each surviving record stores `window_id` (original frame offset), so timestamps are always recoverable.

---

## Flattened Dataset Statistics

The final flattened output (`flat_xyzt/`) contains:

| Metric | Value |
|--------|-------|
| Total files | 160 |
| Total records | 69,844 |
| Total size | ~24 GB |
| Avg record size | ~348 KB |
| Records per file | 321–648 (avg 437) |
| Bad JSON / missing structure | 0 |

**Modality distribution** (sampled every 10th record):

| Modality | Avg tokens/record | Zero rate | Dropout |
|----------|-------------------|-----------|---------|
| seed2 | 1,282 | 0% | 0% |
| cosmos | 2,995 | 8% | 90% |
| avclm | 6,738 | 49% | 99% |
| fps | 31 | 0% | — |
| joint_xyz | 12,549 | 0% | — |

The high dropout for avclm (99%) and cosmos (90%) balances the token ratio relative to agent/seed2 tokens.

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

**Steps B–G — 3D pose pipeline** (HRNet, MotionBERT, YOLO, tokenizer):
```bash
source setup_motionbert.sh    # activates env_motion_final/
```

**Step B only — HRNet** (different env):
```bash
source setup_hrnet_gpu.sh
```

**Flatten + upload** (tools):
```bash
source setup_motionbert.sh    # or any env with huggingface_hub installed
```

Environment YAML specs are in `envs/`.

---

## Cluster (JUPITER — `booster` partition)

- **Hardware:** GH200 nodes, 4 GPUs per node, 288 CPU cores per node
- **Account:** `reformo`
- GPU assignment: `SLURM_LOCALID` → `cuda:{local_id}`
- File partitioning: `SLURM_ARRAY_TASK_ID` + `SLURM_ARRAY_TASK_COUNT`
- All phases are **safe to re-run** — each script skips already-completed output files

---

## Key Data Paths

All data lives under `$DATA = /e/data1/datasets/playground/mmlaion/shared/nguyen38`.

| What | Path |
|------|------|
| FineVideo HF dataset | `$DATA/finevideo_disk` |
| Intermediate pose outputs | `$DATA/outputs/` |
| `training_ready` JSONL (Step A output) | `$DATA/FineVideo-VLA/training_ready_rank_*.jsonl` |
| XYZT merged JSONL (Step H output) | `$DATA/FineVideo-VLA/final_dataset_xyzt/final_vla_xyzt_rank_*.jsonl` |
| Flat Megatron JSONL (Step I output) | `$DATA/flat_xyzt/flat_final_vla_xyzt_rank_*.jsonl` |
| HF upload staging | `$DATA/flat_xyzt_hf_upload/` |
| Legacy final merged JSONL | `$DATA/FineVideo-VLA/final_dataset/final_vla_rank_*.jsonl` |

---

## Vocabulary

`tools/expand_vocab.py` extends the GPT-NeoX-20b base (`vocab/vocab.json`) with:

| Token range | Count |
|-------------|-------|
| `<agent_0>` … `<agent_255>` | 256 |
| `<avclm_0>` … `<avclm_8191>` | 8,192 |
| `<seed2_0>` … `<seed2_8191>` | 8,192 |
| `<cosmos_0>` … `<cosmos_63999>` | 64,000 |
| `<fps_N>`, `<joint_J_d_V>` | per-joint XYZT tokens |
| Wrapper tags (`<agent>`, `</agent>`, …) | 8 |

Output: `vocab/vocab_expanded.json`.
```bash
python tools/check_vocab.py   # verify vocab_size (rounds to nearest 128 for Megatron)
```

---

## Useful Commands

```bash
# Flatten the XYZT merged dataset (Step I)
python tools/flatten.py

# Upload flattened dataset to HuggingFace
export HF_TOKEN='hf_...'
python tools/upload_flattened_hf.py

# Clean old files from HF repo
python tools/cleanup_flattened_hf.py

# Decode a random agent token block from a final_vla file
python tools/decode_agent_tokens.py --seed 42

# Count token density across training_ready shards
python prototype_pipeline/count_tokens.py

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
