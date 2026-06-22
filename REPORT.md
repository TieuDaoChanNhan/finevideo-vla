# FineVideo-VLA: Full Project Report

**Author:** Van Khue Nguyen  
**Date:** June 2025 – June 2026  
**Cluster:** JUPITER (JSC), `booster` partition, GH200 nodes

---

## 1. Goal

Build a multimodal Vision-Language-Action pretraining dataset from ~40K YouTube videos (HuggingFace [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo)). The final output is a Megatron-LM-ready flat JSONL dataset where each record interleaves four token modalities:

- **Seed2** — semantic keyframe tokens (1 FPS, vocab 8192)
- **Cosmos** — spatial video tokens (every 8 frames, vocab 64000)
- **AVC-LM** — H.264 BPE tokens (every 8 frames, vocab 8192)
- **Agent** — 3D human pose tokens (every 8 frames, adaptive PCHIP, 17 joints)

---

## 2. What Was Done

### 2.1 Branch A: Video Token Extraction (prototype pipeline)

**Script:** `prototype_pipeline/pipeline.py`  
**Compute:** 40 SLURM nodes × 4 GPUs = 160 GPUs  

Processed all ~40K FineVideo videos:
- Extracted frames at 30fps
- Tokenised each activity segment with Seed2 (1fps keyframes), Cosmos (8-frame spatial), and AVC-LM (8-frame H.264 BPE)
- Output: 160 `training_ready_rank_*.jsonl` files with hierarchical JSON (video → scenes → activities → tokens)

Each activity contains: `text_prompt`, `speech_transcript`, `video_tokens` (with `<seed2>`, `<cosmos>`, `<avc_lm>` blocks).

### 2.2 Branch B: 3D Human Pose Pipeline

#### Phase 1 — 2D Pose Detection (HRNet + Faster R-CNN)
**Script:** `pipeline/phase1_hrnet_gpu.py` | **SLURM:** `slurm/submit_hrnet.sh`

- Ran HRNet with Faster R-CNN person detection on all videos
- Output: `outputs/2d_json/{video_id}_2d.json` — 2D joint coordinates per frame
- **40,804 videos** processed, **145 GB**

#### Phase 2 — 3D Pose Lifting (MotionBERT)
**Script:** `pipeline/phase2_motionbert_gpu.py` | **SLURM:** `slurm/submit_motionbert.sh`

- Lifted 2D poses to 3D using MotionBERT (pretrained on Human3.6M)
- Processed at native video fps
- Output: `outputs/3d_npy/{video_id}.npy` — 3D joint arrays
- **40,804 videos**, **259 GB**

#### Phase 2.5 — 30fps Resampling
**Script:** `pipeline/phase2_5_resample_30fps.py` | **SLURM:** `slurm/submit_phase2_5.sh`

- Resampled all 3D poses from native video fps to uniform 30fps via linear interpolation
- Required so pose tokens align to the same time grid as Seed2/Cosmos/AVC-LM (all at 30fps)
- Output: `outputs/3d_npy_30fps/{video_id}.npy`
- **40,804 videos**, **67 GB**

#### Phase 3 — Kinematics Processing
**Script:** `pipeline/phase3_kinematics_processor.py` | **SLURM:** `slurm/submit_kinematics.sh`

- Applied temporal smoothing (Butterworth filter)
- Bone length normalisation to canonical Human3.6M skeleton
- Root centering (pelvis at origin)
- Anti-teleportation filter (removes sudden jumps)
- Windowed into 8-frame chunks with position/velocity/acceleration
- Output: `outputs/states_jsonl_30fps/{video_id}_states.jsonl` — shape `(windows, 8, 153)`
- **40,200 videos** (604 dropped due to too-short sequences), **193 GB**

#### Phase 4 — YOLO Person-Presence Cleaning
**Script:** `pipeline/phase4_yolo_cleaner.py` | **SLURM:** `slurm/submit_yolo.sh`

- Ran YOLOv8 person detection on original video frames
- Dropped any 8-frame window where ≥ 4 frames have no detected person (confidence ≥ 0.75)
- Removes windows where subject is off-screen, occluded, or in scene transitions
- Output: `outputs/yolo_cleaned_30fps/{video_id}_cleaned.jsonl`
- **40,195 videos**, **107 GB**

#### Phase 5 — Adaptive PCHIP Tokenisation
**Script:** `pipeline/phase5_adaptive_pchip.py` | **SLURM:** `slurm/submit_phase5_adaptive.sh`

- For each 8-frame window, for each of 17 joints:
  - Computed trajectory curvature
  - Selected 2, 4, or 8 PCHIP control points based on curvature thresholds
  - Low curvature (nearly static) → 2 CPs, medium → 4 CPs, high (fast motion) → 8 CPs
- Quantized positions to uint8: `N = clip(round((v + 2.0) / 4.0 * 255), 0, 255)`
- Produced self-describing tokens: `<fps_30> <pelvis> <pelvis_t_0> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128> ... </pelvis> ...`
- Output: `outputs/agent_tokens_adaptive/{video_id}_tokens.jsonl`
- **18,847 videos** (only videos where YOLO confirmed person presence), **7.4 GB**
- Token count per chunk: 171 (all 2-CP) to 579 (all 8-CP), typical ~250–300

**Why adaptive PCHIP?** A static pelvis doesn't need 8 data points — 2 suffice. A fast-moving wrist does need 8. This reduces average token count by ~35% vs fixed 8-CP while preserving reconstruction quality where it matters.

**Previous iterations (superseded):**
- `phase5_interpolation_tokenizer.py` — 256 opaque uint8 tokens per chunk (scale + anchor + motion CPs). Abandoned because tokens were not self-describing.
- `phase5b_xyzt_tokenizer.py` — 409 fixed tokens per chunk (all 8 frames × 17 joints × 3 dims). Clear and self-describing but wasteful for static joints.

### 2.3 Merge (Phase 6)
**Script:** `pipeline/phase6_merge_adaptive.py` | **SLURM:** `slurm/submit_merge_adaptive.sh`

- Injected `<agent>` blocks after each `<avc_lm>` block in the training_ready files
- Time alignment: matched agent windows to AVC-LM chunks by frame index (both at 30fps, 8-frame windows)
- Added `chunk_timing` array to each activity — maps every 8-frame chunk to its temporal position:
  ```json
  {
    "chunk_idx": 0,
    "abs_frame": 30,
    "start_sec": 1.0,
    "end_sec": 1.267,
    "has_seed2": true,
    "has_cosmos": true,
    "has_avc_lm": true,
    "has_agent": true
  }
  ```
- Added `timing_meta` with fps and rate info for each modality
- Output: 160 `final_vla_adaptive_rank_*.jsonl` files, **657 GB** total
- **~399K activities** across all videos, **~2.15M agent blocks** injected

### 2.4 Flatten (Phase 7)
**Script:** `pipeline/phase7_flatten.py`

- Converted hierarchical JSON (video → scenes → activities) to flat Megatron-LM JSONL
- **Agent-only filter**: only activities containing `<agent>` blocks are emitted, ensuring every record has action data
- Seed2/cosmos/avc_lm tokens flattened: `<seed2> 3758 2157 </seed2>` → `<seed2_3758> <seed2_2157>`
- Agent blocks passed through unchanged (already self-describing named tokens)
- Output: 160 `flat_final_vla_adaptive_rank_*.jsonl` files

#### Modality dropout (token balancing)

In the raw data, image tokens massively outnumber action tokens. The raw token ratio per activity:

| Modality | Avg tokens/activity | Ratio vs Agent |
|----------|-------------------|----------------|
| AVC-LM | ~125,000 | ~373x |
| Cosmos | ~6,400 | ~19x |
| Seed2 | ~340 | ~1x |
| Agent | ~300 | 1x (baseline) |

To balance modalities for pretraining, **modality dropout** is applied during flattening:

| Modality | Drop rate | Effective keep | Resulting tokens |
|----------|-----------|---------------|-----------------|
| AVC-LM | **99%** | ~1% of chunks | ~1,250 |
| Cosmos | **90%** | ~10% of chunks | ~640 |
| Seed2 | 0% | 100% | ~340 |
| Agent | 0% | 100% | ~300 |

This brings all four modalities into roughly the same order of magnitude (~300–1,250 tokens each), preventing the model from being overwhelmed by image tokens during pretraining.

#### Data augmentation

The flatten also applies text augmentation to improve robustness:

| Augmentation | Rate | Description |
|-------------|------|-------------|
| Synonym replacement | 15% | Content words (>5 chars) replaced with WordNet synonyms |
| Stopword dropout | 5% | Common stopwords randomly removed |
| Sentence permutation | 10% | Speech transcript sentences randomly reordered |
| Speech/token interleaving | — | Speech chunks inserted at random positions among tokens |
| Layout block shuffling | — | Title/Context/Keywords/Tokens blocks randomly reordered |

Each output record contains four layout blocks (randomly shuffled):
```
### Title: <scene title, augmented>
### Context: <global context + activity prompt, augmented>
### Keywords: <scene thematic + mood, augmented>
<interleaved speech chunks and flattened tokens>
```

### 2.5 Vocabulary Extension & Tokenizer
**Script:** `tools/expand_vocab.py`

Extended GPT-NeoX-20b base vocabulary (`vocab/vocab.json`) with all VLA tokens:

| Token type | Count | Example |
|-----------|-------|---------|
| `<seed2_N>` (N: 0–8191) | 8,192 | `<seed2_3758>` |
| `<cosmos_N>` (N: 0–63999) | 64,000 | `<cosmos_58567>` |
| `<avclm_N>` (N: 0–8191) | 8,192 | `<avclm_263>` |
| `<fps_N>` (N: 1–60) | 60 | `<fps_30>` |
| Joint wrappers (17 × 2) | 34 | `<pelvis>`, `</pelvis>` |
| `<joint_x_N>`, `_y_N`, `_z_N` (0–255) | 13,056 | `<pelvis_x_128>` |
| `<joint_t_N>` (0–7) | 136 | `<pelvis_t_0>` |
| Modality wrappers | 8 | `<agent>`, `</agent>`, `<seed2>`, ... |
| Legacy `<agent_N>` (0–255) | 256 | `<agent_128>` |

Output: `vocab/vocab_expanded.json`

#### Tokenizer creation

The vocab JSON is a lookup table only — it does **not** make the HuggingFace tokenizer treat these as atomic tokens. A BPE tokenizer without proper registration will split `<seed2_1137>` into sub-pieces (`<`, `seed`, `2`, `_`, `11`, `37`, `>`).

To fix this, a proper HuggingFace tokenizer was created using `tokenizer.add_tokens(special_tokens=True)`:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tok.add_tokens(new_vla_tokens, special_tokens=True)  # 93,938 tokens
tok.save_pretrained("tokenizer_vla_adaptive")         # vocab size: 144,215
```

This tokenizer is published at [EmpathicRobotics/tokenizer-vla-adaptive](https://huggingface.co/EmpathicRobotics/tokenizer-vla-adaptive) and used for Megatron-LM tokenization.

**Script:** `tools/upload_tokenizer.py` — creates and uploads the tokenizer to HuggingFace.

### 2.6 Megatron-LM Tokenization (Phase 8)
**Script:** `/p/data1/mmlaion/nguyen38/mv-scale/tokenize_vla_adaptive.sbatch`

Tokenizes the flattened JSONL into Megatron-LM binary format (`.bin/.idx` shards) for pretraining:

- **Input:** 160 `flat_final_vla_adaptive_rank_*.jsonl` files (18 GB)
- **Tokenizer:** `EmpathicRobotics/tokenizer-vla-adaptive` (144,215 vocab, all VLA tokens atomic)
- **Compute:** 4 nodes, Ray-distributed, 48 CPUs per worker
- **Output:** `/p/data1/mmlaion/shared/vla/tokenized_output/vla_adaptive/data_shard_*.bin/.idx`

### 2.7 HuggingFace Uploads
**Scripts:** `tools/upload_flattened_hf.py`, `tools/upload_vla_agent_hf.py`, `tools/upload_phase4_hf.py`, `tools/upload_tokenizer.py`

All datasets compressed with gzip (level 5), split 152 train / 8 test (95/5, seed 42).

---

## 3. Published Datasets & Tokenizer

| Resource | What | Records | Size | Format |
|----------|------|---------|------|--------|
| [EmpathicRobotics/FineVideo-Prototype-Tokenized](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Prototype-Tokenized) | Base video tokens (Seed2/Cosmos/AVC-LM) from prototype pipeline | ~40K videos | ~660 GB | Hierarchical JSON |
| [EmpathicRobotics/FineVideo-Phase2-3DPose](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase2-3DPose) | 3D pose NPY from MotionBERT (after Phase 2) | ~40K videos | ~259 GB | NumPy arrays |
| [EmpathicRobotics/FineVideo-Phase4-YOLOPose](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase4-YOLOPose) | YOLO-cleaned 3D poses (after Phase 3+4, raw floats) | millions of windows | ~107 GB | `{video_id, window_id, states: float[8][17][3]}` |
| [EmpathicRobotics/FineVideo-Phase5-AgentTokens](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase5-AgentTokens) | Full hierarchical merged dataset with agent tokens (after Phase 5+6) | ~399K activities | ~657 GB | Hierarchical JSON |
| [EmpathicRobotics/FineVideo-Phase7-Flattened](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase7-Flattened) | Flat Megatron-LM JSONL (after Phase 7, agent-only, with modality dropout + augmentation) | 69,844 | ~19 GB | `{"text": "### Title: ... <seed2_N> ... <fps_30> <pelvis> ..."}` |
| [EmpathicRobotics/tokenizer-vla-adaptive](https://huggingface.co/EmpathicRobotics/tokenizer-vla-adaptive) | HuggingFace tokenizer with 93,938 VLA tokens added via `add_tokens()` | — | 144,215 vocab | HF tokenizer dir |

### Published models

| Model | What | Params | Tokenizer |
|-------|------|--------|-----------|
| [EmpathicRobotics/vla-1.7b-pab-spline-25b-test](https://huggingface.co/EmpathicRobotics/vla-1.7b-pab-spline-25b-test) | First VLA model (broken tokenizer, fixed 256-token agent format) | 1.7B | Broken (sub-piece splitting) |
| [EmpathicRobotics/vla-1.7b-pab-spline-adaptive](https://huggingface.co/EmpathicRobotics/vla-1.7b-pab-spline-adaptive) | Second VLA model (fixed tokenizer, adaptive PCHIP agent tokens) | 1.91B | [tokenizer-vla-adaptive](https://huggingface.co/EmpathicRobotics/tokenizer-vla-adaptive) (144,215 vocab, all atomic) |

### What's in each dataset

**FineVideo-Phase7-Flattened** — Use this for LLM pretraining. Each record is a single activity with all modalities flattened into one text string, with modality dropout (99% AVC-LM, 90% Cosmos) and text augmentation applied. Only activities containing 3D pose agent tokens are included.

**FineVideo-Phase5-AgentTokens** — Use this if you need the full structure. Each record is a full video with scenes, activities, timestamps (`chunk_timing`), speech transcripts, and all modality tokens in their original hierarchical form. No dropout or augmentation — all data preserved. You can extract timestamps, filter by modality, or re-flatten with custom logic.

**FineVideo-Phase4-YOLOPose** — Use this if you need raw 3D joint positions (floats in metres, not tokenised). Each record is one 8-frame window with 17 joints × 3 dims. Root-centred, bone-normalised, smoothed.

**tokenizer-vla-adaptive** — The HuggingFace tokenizer for Megatron-LM tokenization. Base GPT-NeoX-20b extended with 93,938 VLA tokens using `add_tokens(special_tokens=True)`. All tokens like `<seed2_1137>` and `<pelvis_x_128>` are treated as single atomic tokens by the BPE tokenizer.

---

## 4. Timestamps and Time Alignment

All four modalities share the same **30fps frame grid**:

| Token type | Fires at | Covers | Timestamp |
|------------|----------|--------|-----------|
| Seed2 | every 30 frames | 1 frame | `activity_start + k × 1.0s` |
| Cosmos | every 8 frames | 8 frames | `activity_start + k × 8/30s` |
| AVC-LM | every 8 frames | 8 frames | `activity_start + k × 8/30s` |
| Agent | every 8 frames | 8 frames | `activity_start + window_id × 8/30s` |

### How to get the timestamp for any token

In the **FineVideo-Phase5-AgentTokens** dataset, each activity has:

```json
{
  "chunk_timing": [
    {
      "chunk_idx": 0,
      "abs_frame": 0,
      "start_sec": 0.0,
      "end_sec": 0.267,
      "has_seed2": true,
      "has_cosmos": true,
      "has_avc_lm": true,
      "has_agent": false
    },
    {
      "chunk_idx": 1,
      "abs_frame": 8,
      "start_sec": 0.267,
      "end_sec": 0.533,
      "has_seed2": false,
      "has_cosmos": true,
      "has_avc_lm": true,
      "has_agent": true
    }
  ],
  "timing_meta": {
    "video_fps": 30,
    "chunk_frames": 8,
    "seed2_rate": "1fps_keyframe",
    "cosmos_rate": "every_8_frames",
    "avc_lm_rate": "every_8_frames",
    "agent_rate": "every_8_frames_adaptive_pchip"
  }
}
```

In the **flattened** dataset, timestamps can be computed from the token sequence order:
- Each `<seed2>...<cosmos>...<avc_lm>...<agent>...</agent>` group = one 8-frame chunk
- Chunk N covers time `[N × 8/30, (N+1) × 8/30]` seconds from activity start
- The activity start/end times are in the original FineVideo metadata

---

## 5. Token Format Details

### Agent tokens (adaptive PCHIP)

```
<agent>
  <fps_30>
  <pelvis> <pelvis_t_0> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128>
           <pelvis_t_7> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128> </pelvis>
  <r_hip>  <r_hip_t_0>  <r_hip_x_115> <r_hip_y_130> <r_hip_z_126>
           <r_hip_t_1>  <r_hip_x_115> <r_hip_y_130> <r_hip_z_126>
           <r_hip_t_3>  <r_hip_x_115> <r_hip_y_128> <r_hip_z_126>
           <r_hip_t_7>  <r_hip_x_116> <r_hip_y_125> <r_hip_z_124> </r_hip>
  ...17 joints total...
</agent>
```

- **`t` tokens**: frame index 0–7 within the 8-frame window (tells you which frames are control points)
- **`x/y/z` tokens**: quantized position in uint8 [0, 255], mapping to [-2.0m, +2.0m]
- **Dequantize**: `position_metres = token_value / 255.0 * 4.0 - 2.0`
- **Reconstruct all 8 frames**: parse the control points (t, x, y, z) per joint, apply PCHIP interpolation

### Joint names (H36M 17-joint skeleton)

| Index | Name | Index | Name | Index | Name |
|-------|------|-------|------|-------|------|
| 0 | pelvis | 6 | l_ankle | 12 | l_elbow |
| 1 | r_hip | 7 | spine | 13 | l_wrist |
| 2 | r_knee | 8 | thorax | 14 | r_shoulder |
| 3 | r_ankle | 9 | nose | 15 | r_elbow |
| 4 | l_hip | 10 | head_top | 16 | r_wrist |
| 5 | l_knee | 11 | l_shoulder | | |

### Flattened record example

```json
{
  "text": "### Keywords: educational, informative\n### Title: Introduction to Forspoken\n### Context: Join Ircha as she shares her thoughts on two games...\n<seed2_6750> <seed2_680> ... <cosmos_63127> <cosmos_42647> ... When it comes to Forspoken... <avclm_263> <avclm_107> ... <fps_30> <pelvis> <pelvis_t_0> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128> <pelvis_t_7> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128> </pelvis> <r_hip> ... </r_hip> ..."
}
```

Note: layout blocks (Title/Context/Keywords/tokens) are randomly shuffled, and speech chunks are interleaved among tokens at random positions.

### Sample files

See the [`samples/`](samples/) directory for concrete examples of the data before and after flattening:

| File | Description |
|------|-------------|
| [`before_flatten.json`](samples/before_flatten.json) | One hierarchical record from `final_dataset_adaptive` — shows video_id, scene metadata, chunk_timing, timing_meta, and one chunk preview per modality (seed2, cosmos, avc_lm, agent) |
| [`after_flatten.json`](samples/after_flatten.json) | Three flattened records from `megatron_dataset_adaptive` — shows the flat `{"text": "..."}` format with per-modality token breakdown (seed2, cosmos, avclm, agent counts after dropout) |
| [`before_vs_after.txt`](samples/before_vs_after.txt) | Readable side-by-side comparison of hierarchical vs flat format, with token counts and text previews |

---

## 6. Pipeline Summary (numbers)

| Stage | Videos | Output Size | Script |
|-------|--------|-------------|--------|
| FineVideo source | ~40,000 | — | HuggingFace dataset |
| Step A: Video tokens | ~40,000 | 160 files, ~660 GB | `prototype_pipeline/pipeline.py` |
| Phase 1: 2D pose (HRNet) | 40,804 | 145 GB | `pipeline/phase1_hrnet_gpu.py` |
| Phase 2: 3D pose (MotionBERT) | 40,804 | 259 GB | `pipeline/phase2_motionbert_gpu.py` |
| Phase 2.5: 30fps resample | 40,804 | 67 GB | `pipeline/phase2_5_resample_30fps.py` |
| Phase 3: Kinematics | 40,200 | 193 GB | `pipeline/phase3_kinematics_processor.py` |
| Phase 4: YOLO cleaning | 40,195 | 107 GB | `pipeline/phase4_yolo_cleaner.py` |
| Phase 5: Adaptive PCHIP | 18,847 | 7.4 GB | `pipeline/phase5_adaptive_pchip.py` |
| Phase 6: Merge | 160 files | 657 GB | `pipeline/phase6_merge_adaptive.py` |
| Phase 7: Flatten (dropout + augment) | 160 files, 69,844 records | 19.2 GB | `pipeline/phase7_flatten.py` |
| Phase 8: Megatron tokenization | 2 shards, 2.84B tokens | 10.58 GB | `tokenize_vla_adaptive.sbatch` |
| Phase 9: Training | 2,032 iters, 3 epochs | 3.6 GB (HF ckpt) | `oellm-autoexp` |

### Why 18,847 not 40,000?

Not all videos contain visible humans. After YOLO filtering (Phase 4), only videos with sufficient person-detected windows produce agent tokens. The other ~21K videos still have Seed2/Cosmos/AVC-LM tokens — they just don't have `<agent>` blocks.

---

## 7. Data Locations on Jupiter

All data under `$DATA = /e/data1/datasets/playground/mmlaion/shared/nguyen38`:

| What | Path |
|------|------|
| FineVideo source | `/e/scratch/reformo/nguyen38/finevideo_disk` |
| Phase 1 output (2D) | `$DATA/outputs/2d_json/` |
| Phase 2 output (3D) | `$DATA/outputs/3d_npy/` |
| Phase 2.5 output (30fps) | `$DATA/outputs/3d_npy_30fps/` |
| Phase 3 output (kinematics) | `$DATA/outputs/states_jsonl_30fps/` |
| Phase 4 output (YOLO cleaned) | `$DATA/outputs/yolo_cleaned_30fps/` |
| Phase 5 output (adaptive tokens) | `$DATA/outputs/agent_tokens_adaptive/` |
| Step A output (video tokens) | `$DATA/FineVideo-VLA/training_ready_rank_*.jsonl` |
| Phase 6 output (merged) | `$DATA/FineVideo-VLA/final_dataset_adaptive/` |
| Phase 7 output (flattened) | `$DATA/FineVideo-VLA/megatron_dataset_adaptive/` |

---

## 8. Flattened Dataset Quality Metrics

Evaluated on the final `megatron_dataset_adaptive/` output (with modality dropout and augmentation):

| Metric | Value |
|--------|-------|
| Total files | 160 shards |
| Total records | 69,844 |
| Total size | 19.2 GB |
| Avg file size | 120 MB (range: 85.8 – 176.7 MB) |
| Malformed JSON | 0 |
| Records with `### Title:` | 100% |
| Records with `### Context:` | 100% |
| Records with `### Keywords:` | 100% |
| Records with agent (3D pose) | **100%** (agent-only filter) |

### Modality coverage (after dropout)

| Modality | Coverage | Avg tokens/record |
|----------|----------|-------------------|
| seed2 | 100% | ~1,320 |
| cosmos | ~88% | ~3,091 |
| avclm | ~49% | ~7,260 |
| agent | 100% | ~9,712 |

### Token length per record

| Stat | Value |
|------|-------|
| Min | 336 |
| Median | 8,512 |
| Mean | 21,563 |
| Max | 505,180 |

### Agent block validation
- All 17 joints present in every record (pelvis through r_wrist, using `head_top` per H36M convention)
- XYZ values in valid range [0, 255]
- T values in valid range [0, 7]
- Agent tokens per block: 171–555, mean ~327

---

## 9. How to Use the Data

### For LLM pretraining (Megatron-LM)

Use `EmpathicRobotics/FineVideo-Phase7-Flattened`. Each line is `{"text": "..."}` — ready for Megatron-LM tokenization with the expanded tokenizer. Every record contains agent (3D pose) tokens with balanced modality ratios.

```python
from datasets import load_dataset

ds = load_dataset("EmpathicRobotics/FineVideo-Phase7-Flattened", streaming=True)
for sample in ds["train"]:
    text = sample["text"]  # ### Title: ... <seed2_N> ... <fps_30> <pelvis> ...
    break
```

To tokenize with the correct tokenizer:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("EmpathicRobotics/tokenizer-vla-adaptive")
ids = tok.encode(text)  # all VLA tokens are single atomic tokens
```

### For structured analysis (timestamps, filtering)

Use `EmpathicRobotics/FineVideo-Phase5-AgentTokens`. Full hierarchical data with all metadata.

```python
from datasets import load_dataset

ds = load_dataset("EmpathicRobotics/FineVideo-Phase5-AgentTokens", streaming=True)
for sample in ds["train"]:
    video_id = sample["video_id"]
    for scene in sample["scenes"]:
        for activity in scene["activities"]:
            tokens = activity["video_tokens"]
            timing = activity.get("chunk_timing", [])
            speech = activity.get("speech_transcript", "")
            # Each chunk in timing tells you exact start_sec/end_sec
            # and which modalities are present
    break
```

### For raw 3D poses (float coordinates)

Use `EmpathicRobotics/FineVideo-Phase4-YOLOPose`. Raw float arrays, not tokenised.

```python
from datasets import load_dataset

ds = load_dataset("EmpathicRobotics/FineVideo-Phase4-YOLOPose", streaming=True)
for sample in ds["train"]:
    states = sample["states"]     # float[8][17][3] — 8 frames, 17 joints, xyz
    video_id = sample["video_id"]
    window_id = sample["window_id"]
    timestamp = window_id / 30.0  # seconds from video start
    break
```

---

## 10. Tokenizer Fix & Megatron Tokenization (Phase 8)

### 10.1 The tokenizer bug

The first VLA model ([EmpathicRobotics/vla-1.7b-pab-spline-25b-test](https://huggingface.co/EmpathicRobotics/vla-1.7b-pab-spline-25b-test), May 2026) was trained with a broken tokenizer. The expanded vocabulary was created by manually editing `vocab.json`, but this **does not register tokens with the BPE merge rules**. The HuggingFace tokenizer split VLA tokens into sub-pieces:

```
<seed2_1137>  →  ['<', 'seed', '2', '_', '11', '37', '>']   (7 sub-tokens)
<pelvis_x_128>  →  ['<', 'pel', 'vis', '_', 'x', '_', '128', '>']  (8 sub-tokens)
```

Despite this, the model still showed signal — it learned to predict sequences of sub-tokens that looked like VLA tokens. But it was not decoding real tokens.

### 10.2 The fix

A proper HuggingFace tokenizer was created using `tokenizer.add_tokens(special_tokens=True)`:

```python
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")  # 50,277 tokens
tok.add_tokens(new_vla_tokens, special_tokens=True)               # +93,938 tokens
tok.save_pretrained("tokenizer_vla_adaptive")                     # 144,215 total
```

Now each VLA token is a single atomic token:
```
<seed2_1137>    →  [59908]     (1 token)
<pelvis_x_128>  →  [131151]    (1 token)
```

Published at [EmpathicRobotics/tokenizer-vla-adaptive](https://huggingface.co/EmpathicRobotics/tokenizer-vla-adaptive).

### 10.3 Megatron-LM tokenization

The flattened JSONL was tokenized into Megatron `.bin/.idx` binary format using the fixed tokenizer:

- **Script:** `tokenize_vla_adaptive.sbatch` (4 nodes, Ray-distributed, 48 CPUs/worker)
- **Input:** 160 `flat_final_vla_adaptive_rank_*.jsonl` files (18 GB)
- **Output:** 2 shards in `/p/data1/mmlaion/shared/vla/tokenized_output/vla_adaptive/`

| Shard | Tokens | Size |
|-------|--------|------|
| `data_shard_00000.bin` | 2,684,323,146 | 10.00 GB |
| `data_shard_00001.bin` | 156,389,702 | 0.58 GB |
| **Total** | **2,840,712,848 (2.84B)** | **10.58 GB** |

---

## 11. Training (Phase 9)

### 11.1 First model (May 2026, broken tokenizer)

- **Model:** [EmpathicRobotics/vla-1.7b-pab-spline-25b-test](https://huggingface.co/EmpathicRobotics/vla-1.7b-pab-spline-25b-test)
- **Architecture:** OpenSci-Ref 1.7B (24 layers, 2048 hidden, 32 heads)
- **Data:** ~25B tokens from the old `vla_25b` dataset (broken tokenizer, no joint tokens)
- **Result:** Model could replicate seed2/cosmos token sequences but was decoding sub-pieces, not real VLA tokens

### 11.2 Second model (June 2026, fixed tokenizer)

- **Model:** [EmpathicRobotics/vla-1.7b-pab-spline-adaptive](https://huggingface.co/EmpathicRobotics/vla-1.7b-pab-spline-adaptive)
- **Architecture:** OpenSci-Ref 1.7B (24 layers, 2048 hidden, 32 heads, 1.91B params with 144K vocab embeddings)
- **Data:** 2.84B tokens from `vla_adaptive` (fixed tokenizer, with named joint tokens)
- **Training config:** `oellm-autoexp/config/experiments/nguyen38/vla_adaptive.yaml`
- **Schedule:** WSD (200 warmup iters, peak LR 4e-3, 400 linear decay at end), 2,032 iters (~3 epochs), GBS=1024, MBS=4, seq_len=4096
- **Compute:** 64 nodes × 4 GH200 GPUs (256 GPUs), ~287 TFLOP/s/GPU, ~35 min wall time
- **Tokenizer:** `EmpathicRobotics/tokenizer-vla-adaptive` (144,215 vocab, all VLA tokens atomic)
- **Vocab size:** 144,256 (padded to 128 for Megatron)
- **Checkpoints saved:** iter 500, 1000, 1500, 2000, 2032 (all converted to HF format)

#### Loss curve

| Iter | Train Loss | LR | Tokens Seen |
|------|-----------|-----|-------------|
| 50 | 6.158 | 1.0e-3 | 0.21B |
| 100 | 3.927 | 2.0e-3 | 0.42B |
| 200 | 2.982 | 4.0e-3 | 0.84B |
| 500 | 2.070 | 4.0e-3 | 2.10B |
| 1000 | 1.672 | 4.0e-3 | 4.19B |
| 1500 | 1.555 | 4.0e-3 | 6.29B |
| 2000 | 1.476 | 3.2e-4 | 8.39B |
| **2032 (val)** | **1.501** | — | — |
| **2032 (test)** | **1.494** | — | — |

Final validation PPL: **4.49**, test PPL: **4.45**.

### 11.3 Evaluation results (June 21, 2026)

Evaluation script: `tools/eval_vla_sanity.py`

#### Test 1: Token atomicity — PASS

All 23 tested VLA tokens encode as single atomic token IDs. The tokenizer fix is confirmed:

```
<seed2_1137>    → [59908]   (1 token)   ← old model: 7 sub-pieces
<pelvis_x_128>  → [131151]  (1 token)   ← old model: 8 sub-pieces
<fps_30>        → [130992]  (1 token)
<cosmos_58567>  → [125530]  (1 token)
```

#### Test 2: Greedy generation — partial success

| Prompt | Tokens | Result |
|--------|--------|--------|
| Full training-like prompt (Title/Context/Keywords) | 2000 | Generated valid `<seed2_N>` tokens but stayed in seed2 mode, never transitioned to cosmos/avclm/agent |
| Partial agent block (given `<fps_30> <pelvis> <pelvis_t_0> ...`) | 500 | Correctly completed the full 17-joint agent block with valid structure |
| Real seed2 block from training data | 2000 | Continued generating seed2 tokens, no transition to cosmos/agent |

**Agent continuation result (the key success):**

The model correctly generated:
```
<pelvis_t_7> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128> </pelvis>
<r_hip> <r_hip_t_0> <r_hip_x_115> <r_hip_y_127> <r_hip_z_127>
        <r_hip_t_7> <r_hip_x_115> <r_hip_y_127> <r_hip_z_127> </r_hip>
<r_knee> <r_knee_t_0> <r_knee_x_114> <r_knee_y_155> <r_knee_z_133> ...
```

- Correct joint ordering (H36M sequence: pelvis → r_hip → r_knee → ... → r_wrist)
- Valid open/close tag pairs
- xyz values in range [0, 255], t values in [0, 7]
- Correct adaptive CP count (2 CPs for pelvis, appropriate counts for other joints)
- Successfully decoded to 3D pose: shape (8, 17, 3), range [-0.31, 0.89] m

#### Comparison: old model vs new model

| Aspect | Old model (25b-test) | New model (adaptive) |
|--------|---------------------|---------------------|
| Token atomicity | ❌ `<seed2_1137>` → 7 sub-pieces | ✅ `<seed2_1137>` → 1 token |
| Agent token format | Fixed 256 opaque integers | Self-describing `<joint_t_N> <joint_x_N>` |
| Seed2 generation | Generated sub-piece fragments | ✅ Generates valid atomic seed2 tokens |
| Agent completion | Could not complete (wrong format) | ✅ Completes full 17-joint sequence |
| Decode to 3D pose | Not possible (tokens were sub-pieces) | ✅ Decodes to (8, 17, 3) trajectory |
| Modality transitions | N/A (broken tokens) | ❌ Cannot initiate agent blocks from text alone |
| Training data | ~25B tokens (but wasted on sub-pieces) | 2.84B tokens (3 epochs) |

### 11.4 Known limitations

- **Data scarcity:** 2.84B tokens is small for a 1.7B model (Chinchilla optimal: ~20B). The model memorises training patterns but cannot generalise to novel prompts or learn modality transitions (seed2 → cosmos → avclm → agent). As noted in team discussions: "people train a 1.7B on 11T tokens... you are throwing only a few 100B tokens."
- **No autonomous modality transitions:** The model generates seed2 tokens when prompted with text, but never transitions to cosmos/avclm/agent on its own. It requires agent tokens in the prompt to continue in agent mode. This is expected with 3 epochs of training — the model has seen each modality transition pattern only ~3 times per sample.
- **Modality dropout imbalance:** Phase 7 drops 99% of avclm and 90% of cosmos tokens, so the model sees far fewer cosmos/avclm examples relative to seed2 and agent tokens. This likely contributes to the model's inability to learn transitions.
- **No simulation data:** Training data is 100% FineVideo YouTube videos. No Isaac Sim rollouts, RL policies, or MoCap data are included yet.
- **Simplified tokenizer vs spec:** The current adaptive PCHIP tokenizer encodes only xyz positions. The PAB-Spline spec calls for joint angles (q), velocities (qd), phase variable φ, cyclic detection, and static joint compression — all not yet implemented.

---

## 12. Upcoming Work

### Data improvements
- **Rich augmentation pipeline:** Run `process_finevideo.py` + `decode_and_caption.py` (from the FineVideo VLA Pipeline doc) to add perspective framing (robot/human/cinematic), `<think>` planning blocks, Cosmos/Seed2 visual decoding + SmolVLM2 captioning. This would produce 4× more records with much richer language context.
- **More data sources:** Incorporate [SenseNova-SI-8M](https://huggingface.co/datasets/sensenova/SenseNova-SI-8M), [stera-10m](https://huggingface.co/datasets/fpvlabs/stera-10m), [MixtureVitae-Omni](https://huggingface.co/datasets/mixture-vitae/MixtureVitae-Omni) to scale beyond 20B tokens.
- **Isaac Sim integration:** Generate simulation rollouts with the Unitree H1, tokenize with the PAB-Spline tokenizer, and mix into training data.

### Tokenizer improvements
- **Upgrade to PAB-Spline spec:** Add joint angles (q/qd), phase variable φ ∈ [0,1], cyclic gait detection, static joint compression. Current PCHIP xyz-only tokenizer is v1.
- **Qwen3 migration:** Retokenize data with Qwen3-based expanded tokenizer for ecosystem compatibility (native HF support, vLLM, llama.cpp).

### Evaluation
- ~~**Token verification:** Decode `.bin/.idx` shards back to text and verify per-token round-trip.~~ **DONE** — confirmed all VLA tokens are atomic (Section 11.3).
- ~~**Pose reconstruction quality:** Decode agent tokens from model output → PCHIP interpolation → 3D skeleton.~~ **DONE** — model generates decodable agent tokens, decoder script at `tools/decode_agent_tokens.py` (Section 11.3).
- **Per-token-type accuracy:** Teacher-forced next-token accuracy on held-out data, broken down by modality (seed2/cosmos/avclm/agent). Would reveal if the model learned agent patterns better than other modalities.
- **Cross-checkpoint comparison:** Run eval on iter 500/1000/1500/2000/2032 to see learning curves per modality.
- **Standard NLP benchmarks:** Run `oellm-cli` eval (open-sci-0.01, dclm-core-22) to check language ability retention.
- **Video evaluation:** Use CLIP Benchmark video evals (per Jenia's suggestion).

### Deployment
- **Sim-to-real:** Map predicted joint tokens to Unitree H1 control signals via Isaac Sim / ManiSkill.
- **Multi-agent OS:** Safety, Motion, Vision, Exploration agents for real-time trajectory execution with preemption.

---

## 13. Improvement Plan (June 2026)

### Problem Diagnosis

The second model (vla-1.7b-pab-spline-adaptive) validates the architecture and tokenization — it can complete 17-joint agent blocks with correct grammar, joint ordering, and decodable 3D poses. However, it cannot perform modality transitions (seed2 → cosmos → avclm → agent) autonomously. Three root causes:

1. **Data starvation**: 2.84B tokens for 1.91B params (~1.5× Chinchilla ratio). Optimal is ~20× (38B tokens). The model saw each training sample only ~3 times — enough to memorize local patterns but not enough to learn the higher-level sequencing of modality blocks.

2. **No rich language context**: Text is just Title/Context/Keywords. No captions describe what's happening visually at each timestamp. Without language anchors at modality transitions, the model has no signal for "what comes next." Huu: "we need more language through captions. Otherwise the model won't be easily steerable."

3. **Over-aggressive modality dropout**: 99% avclm drop + 90% cosmos drop means most training samples lack the full transition chain. The model rarely sees seed2 → cosmos → avclm → agent in sequence.

### Phase 1: Data Inventory & Pie Chart

**Goal**: Count tokens across ALL available multimodal datasets. Create a pie chart and table showing token counts by modality, number of records, and size in GB.

**Datasets to inventory**:

| Dataset | Source | Token types present |
|---------|--------|---------------------|
| FineVideo-Phase7-Flattened | `EmpathicRobotics/FineVideo-Phase7-Flattened` | seed2, cosmos, avclm, agent, text |
| MixtureVitae-Omni | `mixture-vitae/MixtureVitae-Omni` | image tokens, snac, seed2, text — **no cosmos/avclm/agent** |
| MixtureVitae-Backup (stack_images) | `mixture-vitae-backup/MixtureVitae-Backup` | stack exchange images + text (raw, needs tokenizing) |
| MixtureVitae-Backup (valid_with_seed) | `mixture-vitae-backup/MixtureVitae-Backup` | seed2 tokens + text |
| SenseNova-SI-8M | `sensenova/SenseNova-SI-8M` | 8M image-text pairs (raw) |
| stera-10m | `fpvlabs/stera-10m` | 10M video clips (license concern) |
| OmniAction | `OpenMOSS-Team/OmniAction` | action-labeled video data |

**Output**: Two pie charts (by modality + by dataset) + summary table.

**Script**: `tools/data_inventory.py` — downloads 1 shard samples, counts token types, extrapolates, generates charts.

**Preliminary results** (FineVideo exact, MV-Omni sampled):

| Dataset | Records | Size | seed2 | cosmos | avclm | agent | snac | text | Total |
|---------|---------|------|-------|--------|-------|-------|------|------|-------|
| FineVideo-VLA | 69,844 | 19.2 GB | 89.9M | 210.2M | 474.4M | 637.9M | — | 362.5M | 1.77B |
| MV-Omni | ~181K | 36 GB gz | 5.8M | — | — | — | 106.8M | 55.1M | 0.17B |
| MV-Backup valid_with_seed | large | 1,233 GB gz | est. ~30B | — | — | — | — | est. ~138B | ~168B |

Key findings:
- **valid_with_seed is dominant** (~168B tokens, mostly text+seed2). Even 10% of it would give ~17B tokens.
- **Only FineVideo has agent tokens** (638M). External datasets add language/image diversity but not action data.
- **MV-Omni uses `<seed_N>` not `<seed2_N>`** — needs vocab mapping or rename during data conversion.
- **SNAC tokens** (~107M in MV-Omni) are not in our current vocab — need expansion.

**Can do during JUPITER downtime**: Yes — use HF streaming API on JUSUF.

### Phase 2: Video Captioning for FineVideo

**Goal**: Generate natural language captions for each video segment and interleave them with tokens. This is the highest-impact improvement.

Current format:
```
### Context: Person chops vegetables
<seed2_6750> <seed2_680> ... <cosmos_N> ... <avclm_N> ... <agent> <fps_30> <pelvis> ...
```

With captions:
```
### Context: Person chops vegetables
A woman in a blue apron stands at a kitchen counter. She picks up a knife with her right hand.
<seed2_6750> <seed2_680> ...
She brings the knife down in a smooth chopping motion on a red bell pepper.
<cosmos_N> ... <avclm_N> ... <agent> <fps_30> <pelvis> ...
```

**How**:
- Use timestamps from `chunk_timing` (already in Phase 5/6 output) to locate keyframes
- Extract keyframe images at each seed2 timestamp (1 FPS)
- Run a vision-language captioner (SmolVLM2, Qwen2.5-VL, or Moondream2) on each keyframe
- Interleave captions into the token sequence at matching timestamps
- This is the "rich augmentation pipeline" from the FineVideo VLA Pipeline spec (`process_finevideo.py` + `decode_and_caption.py`)

**Impact**: Gives the model language anchors at modality transitions. With perspective framing (robot/human/cinematic views), produces 4× more records with richer language context.

**Blocked by**: Needs GPU for captioning model. Code can be written during downtime.

### Phase 3: Integrate External Datasets

**Important**: MV-Omni, stack exchange, SenseNova have text + image tokens but **NOT** cosmos/avclm/agent. They help the model maintain language understanding and learn seed2/snac patterns, but only FineVideo has the full multimodal chain.

**Strategy**: Mix external data as a "language+image backbone":
- **FineVideo VLA** (~30% of training mix): teaches the full multimodal chain with agent tokens
- **MV-Omni + stack exchange + etc.** (~70% of mix): teaches general language and image understanding

This way the model sees each FineVideo sample ~10× while getting diverse text+image training.

**Vocab impact**: Need to add snac tokens to the tokenizer for MV-Omni. Current vocab has seed2/cosmos/avclm/agent but not snac.

### Phase 4: Adjust Modality Dropout

**Current vs proposed dropout rates**:

| Modality | Current drop | Proposed drop | Rationale |
|----------|-------------|---------------|-----------|
| AVC-LM | 99% | 80–90% | Keep 10–20% so model sees avclm regularly |
| Cosmos | 90% | 50–70% | Keep 30–50% so cosmos appears in most records |
| Seed2 | 0% | 0% | Already balanced |
| Agent | 0% | 0% | Already balanced |

**Trade-off**: More tokens per record → fewer records fit in context (seq_len=4096). May need to increase seq_len to 8192.

**Impact**: Model sees real modality transitions in training. Cheapest fix available.

### Phase 5: Re-training Strategy

| Version | Data | Est. tokens | Key change |
|---------|------|-------------|------------|
| v0.2 (quick) | FineVideo VLA + adjusted dropout + captions | ~5–10B | Dropout fix + captions, same 1.7B model |
| v0.3 (scaled) | FineVideo VLA + MV-Omni + stack exchange | ~20–40B | Mixed dataset, possibly larger model |
| v1.0 (full) | All sources + Isaac Sim + cyclic PAB-Spline | ~100B+ | Full spec implementation |

### Priority Table

| Priority | Task | During downtime? | Impact on model |
|----------|------|------------------|-----------------|
| 1 | Data inventory + pie chart | Yes | Huu's explicit request, guides all other decisions |
| 2 | Write captioning pipeline code | Yes (code only) | Prep for highest-impact improvement |
| 3 | Adjust dropout + re-flatten | Partially (code + test) | Cheapest fix for modality transitions |
| 4 | Download + explore MV-Omni | Yes (streaming) | Understand data landscape |
| 5 | Vocab expansion for snac tokens | Yes (code) | Needed for MV-Omni integration |
| 6 | Re-training v0.2 | No (needs JUPITER) | First real model improvement |
| 7 | Full captioning run on GPUs | No (needs JUPITER) | Major data quality boost |

---

## 14. Repository

**GitHub:** [TieuDaoChanNhan/3D-Human-Pose-VLA](https://github.com/TieuDaoChanNhan/3D-Human-Pose-VLA)

All pipeline scripts, SLURM jobs, upload tools, vocab, and documentation are in this repo. See `README.md` for setup instructions and detailed usage.
