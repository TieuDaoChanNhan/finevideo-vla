# PAB-Spline VLA — Project Progress

**Author:** Van Khue Nguyen  
**Last updated:** July 2, 2026  
**Cluster:** JUPITER (JSC), `booster` partition, GH200 nodes  
**Goal:** Build a multimodal Vision-Language-Action model that can watch video, hear speech, and generate robot motion tokens.

---

## The Big Picture — What Are We Actually Building?

The ultimate target: a single model that receives multimodal input (video frames + speech/text commands) and outputs action tokens that can be decoded into robot joint trajectories. The analogy Huu used: *"hear a verbal command like 'walk forward', and have a robot actually walk forward based on learned pose patterns."*

The longer-term vision is a model that, given an image of a beaker and a chemical formula, could reason through the task ("Make salt water") and translate that into arm/hand movements — **without being explicitly trained on that exact task**. This requires genuinely cross-modal binding: vision ↔ language ↔ action.

We are building this by pretraining a 1.7B LLM on an interleaved token stream:

```
USER: <activity description> [Speech: ...]  ASSISTANT:
  <seed2_N> ...          # Semantic keyframe tokens  (1fps, vocab 8192)
  <cosmos_N> ...         # Spatial video tokens      (every 8 frames, vocab 64000)
  <avclm_N> ...          # H.264 BPE video tokens    (every 8 frames, vocab 8192)
  <fps_30> <pelvis> ...  # 3D human pose tokens      (every 8 frames, 17 joints)
  <snac_N> ...           # Audio tokens — SNAC listen format (~10 tokens per 8-frame chunk)
```

The model learns to "read" and "continue" this interleaved sequence. In inference, you prompt it with video tokens + a text command, and it predicts the next agent tokens = the motion.

**Why this approach?** No prior VLA model has tried to unify video tokenization (Seed2/Cosmos), speech (SNAC), and continuous motion (PCHIP spline) into a single LLM autoregressive context. We are at the research frontier — nobody here has done this before.

---

## Timeline Overview

| Period | Key milestone |
|--------|--------------|
| Jun 2025 | Project started. FineVideo dataset chosen (~40K YouTube videos). |
| Jul–Sep 2025 | Branch A: Video token extraction pipeline (Seed2, Cosmos, AVC-LM). 160 GPU run. |
| Sep–Nov 2025 | Branch B phase 1–3: HRNet 2D pose, MotionBERT 3D lifting, kinematics. |
| Nov–Dec 2025 | Phase 4: YOLO cleaning. Phase 5 first iteration (opaque 256-token format). |
| Jan–Feb 2026 | Phase 5 rewrite → Adaptive PCHIP (self-describing named joint tokens). |
| Mar 2026 | Phase 6 merge, Phase 7 flatten. First Megatron tokenization. |
| Apr 2026 | **First model** trained (vla-1.7b-pab-spline-25b-test). Broken tokenizer discovered. |
| May 2026 | Tokenizer fix: `add_tokens(special_tokens=True)`. Full re-tokenization. |
| Jun 2026 | **Second model** trained (vla-1.7b-pab-spline-adaptive). Evaluation. Data inventory. |

---

## What Is Done — Detailed

### Phase A: Video Token Extraction

**Script:** `pipeline_video/pipeline.py` | **Compute:** 40 nodes × 4 GPU

Processed all ~40K FineVideo videos. Each activity segment tokenized into:
- **Seed2**: 1fps semantic keyframe tokens (8192 vocab)
- **Cosmos**: every-8-frame spatial tokens (64000 vocab)
- **AVC-LM**: every-8-frame H.264 BPE tokens (8192 vocab)

Output: 160 `training_ready_rank_*.jsonl` files. Each file contains hierarchical JSON (video → scenes → activities → tokens + speech transcript + metadata).

---

### Phase 1: 2D Pose Detection

**Script:** `pipeline_pose/phase1_hrnet_gpu.py`

- HRNet-W48 + Faster R-CNN person detector on all 40K videos
- Output: 2D joint coordinates (17 joints, COCO format) per frame
- **40,804 videos**, 145 GB

---

### Phase 2: 3D Pose Lifting

**Script:** `pipeline_pose/phase2_motionbert_gpu.py`

- MotionBERT lifts 2D → 3D (pretrained on Human3.6M)
- **40,804 videos**, 259 GB

---

### Phase 2.5: 30fps Resampling

**Script:** `pipeline_pose/phase2_5_resample_30fps.py`

- Linear interpolation from native video fps → uniform 30fps
- Required so all modalities share the same time grid
- 67 GB

---

### Phase 3: Kinematics Processing

**Script:** `pipeline_pose/phase3_kinematics_processor.py`

- Butterworth temporal smoothing
- Bone-length normalization to canonical H36M skeleton
- Pelvis root-centering
- Anti-teleportation filter (drops sudden-jump windows)
- Windowed into 8-frame chunks → shape `(windows, 8, 153)` where 153 = 17 joints × 3 dims × 3 kinematics (pos/vel/acc)
- **40,200 videos** (604 dropped as too short), 193 GB

---

### Phase 4: YOLO Person-Presence Filtering

**Script:** `pipeline_pose/phase4_yolo_cleaner.py`

- YOLOv8 person detection per frame
- Drops any 8-frame window where ≥4 frames have no detected person (confidence ≥ 0.75)
- **40,195 videos**, 107 GB

**⚠ Pose Data Quality Finding (Jul 2, 2026):**

Side-by-side skeleton visualization (`tools/visualize_skeleton_sidebyside.py`) + direct inspection of `yolo_cleaned` data revealed significant quality issues:

| Issue | Detail |
|-------|--------|
| **Joint sparsity** | Average 4–7 finite joints per frame out of 17 (24–41% skeleton) |
| **Arms absent** | j11–j16 (both arms: shoulder/elbow/wrist) = NaN in nearly all frames — MotionBERT cannot reliably lift arm joints from YouTube videos due to occlusion/side views |
| **Zero-fill artifact** | j10 (head_top) often stores (0,0,0) when undetected, identical to pelvis position — counted as finite but is wrong/misleading |
| **Coordinate scale OK** | ankle at ~−0.638m below pelvis is anatomically plausible; metric scale is correct |

**Impact on training:** Pose tokens are predominantly lower body (hip/knee/ankle) + torso. The arms — most important for manipulation tasks — are almost never captured. The model learns rough walking/sitting body motion but not fine hand/arm motion. This is a fundamental limitation of monocular video pose lifting from YouTube.

**Does NOT affect FineVideo as pretraining signal** — even noisy lower-body pose is better than none for learning video-pose correlation. But for downstream manipulation fine-tuning, better pose data (simulation, MoCap, or depth cameras) will be needed.

---

### Phase 5: Adaptive PCHIP Tokenization

**Script:** `pipeline_pose/phase5_adaptive_pchip.py`

For each 8-frame window, for each of 17 joints:
1. Compute trajectory curvature
2. Choose 2, 4, or 8 control points: low curvature (static) → 2 CPs; medium → 4 CPs; fast motion → 8 CPs
3. Quantize positions to uint8: `N = clip(round((v + 2.0) / 4.0 * 255), 0, 255)` mapping [-2m, +2m]
4. Emit self-describing tokens: `<pelvis> <pelvis_t_0> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128> ... </pelvis>`

**Why adaptive?** A static pelvis doesn't need 8 data points — 2 suffice. A fast-moving wrist needs 8. Reduces average token count by ~35% vs fixed 8-CP.

**Previous iterations (abandoned):**
- `phase5_interpolation_tokenizer.py` — 256 opaque uint8 tokens. Abandoned: tokens were not self-describing, model couldn't learn joint semantics.
- `phase5b_xyzt_tokenizer.py` — 409 fixed tokens (all 8 frames × 17 joints × 3 dims). Self-describing but wasteful.

Output: **18,847 videos** (only where YOLO confirmed human presence), 7.4 GB.  
Token range: 171 (all 2-CP, very static pose) to 579 (all 8-CP, fast motion), typical ~250–300 per window.

---

### Phase 6: Merge

**Script:** `pipeline_pose/phase6_merge_adaptive.py`

- Injected `<agent>...</agent>` blocks after each `<avc_lm>` block in the training_ready files
- Frame-aligned by matching agent window_ids to AVC-LM chunk indices (both at 30fps, 8-frame windows)
- Added `chunk_timing` array to each activity (precise timestamps for every chunk, which modalities are present)
- ~399K activities, **~2.15M agent blocks** injected
- Output: 160 `final_vla_adaptive_rank_*.jsonl`, **657 GB**

**Phase 6 v2 — SNAC injection support (Jun 28, 2026):**
- Added `--snac-tokens-dir` argument to also inject SNAC audio tokens per chunk
- New `inject_chunk_tokens()` function handles both agent + SNAC in one pass over `video_tokens`
- Token order per 8-frame chunk: `<cosmos>...</cosmos> <avc_lm>...</avc_lm> [<agent>...</agent>] [<snac>...</snac>]`
- `chunk_timing` now includes `has_snac` flag per chunk
- `timing_meta` now includes `snac_rate: "37.5_tokens_per_sec_listen_format"`
- Backward compatible: running without `--snac-tokens-dir` behaves identically to v1
- **Requires `snac_finevideo.py` to run first** → `{video_id}_snac.jsonl` files in snac output dir

---

### Phase 7: Flatten + Augment

**Script:** `pipeline_pose/phase7_flatten.py`

Converts hierarchical JSON → flat Megatron-LM JSONL. Key decisions:

**Agent-only filter:** Only activities with `<agent>` blocks are emitted (every training record has action data).

**Modality dropout (token balancing) — v1 (old, already trained on this):**
| Modality | Raw ratio vs agent | Drop rate | Resulting ratio |
|----------|-------------------|-----------|----------------|
| AVC-LM | ~373× | 99% | ~4× |
| Cosmos | ~19× | 90% | ~2× |
| Seed2 | ~1× | 0% | 1× |
| Agent | baseline | 0% | 1× |

**Modality dropout — v2 (Jun 27, 2026 update, pending re-flatten):**
| Modality | Drop rate | Reason |
|----------|-----------|--------|
| AVC-LM | **100%** | Removed until ablations confirm benefit (per Huu) |
| Cosmos | **50%** | Keep ~6/12 blocks per activity for modality transition learning |
| Seed2 | 0% | Keep all — primary visual signal |
| Agent | 0% | Keep all |

**Text augmentation:** 15% synonym replacement, 5% stopword dropout, 10% sentence permutation, random speech/token interleaving, random layout block shuffling.

Output v1: 160 files, **69,844 records**, 19.2 GB → `megatron_dataset_adaptive/`  
Output v2: → `megatron_dataset_v2/` (cosmos 50% drop, avclm 100% drop — re-flattened Jun 27, 2026)

**Phase 7 v3 — SNAC + updated filter (COMPLETE Jul 2, 2026):**
- Added `<snac>...</snac>` block extraction in `process_tokens_to_individual_tags` (pass-through, like agent)
- Added `--drop_snac` argument (default 0.0 = keep all SNAC tokens)
- **Changed record filter:** was `<agent> required`; now emits if `<agent>` OR `<snac>` present
  - Full-chain records: seed2 + cosmos + agent + snac — **69,811 records (18.8%)**
  - Partial-chain records: seed2 + cosmos + snac — **302,044 records (81.2%)**
  - Bad records (neither): **0**
- Output: `megatron_dataset_v3/` — 160 files, **371,888 records**, **72 GB**
- Sample: `samples/after_flatten_v3.json` | Upload script: `tools/upload_flattened_hf.py` (updated for v3)

**✅ Phase 7 v4 — Per-chunk temporal ordering (COMPLETE Jul 2, 2026):**

Phase 7 fully rewritten (`pipeline_pose/phase7_flatten.py`). State machine walks Phase 6 output in document order, emitting per chunk: `[seed2?][cosmos?][agent?][snac?]`. Speech moved to dedicated `### Speech:` header.

**v4 stats:** 160/160 files, 371,888 records, **5.217B tokens** (seed2 6.4% / cosmos 74.4% / agent 12.2% / snac 7.0%). Runtime: 36 min / 32 workers.

**Bugs fixed:**
- Temporal misalignment (v3: all agent at end → 69% of records had 0% agent in first 4096 tokens. v4: per-chunk → all records have agent in first 4096 tokens)
- Speech injection into agent grammar (v3: speech words scattered into joint sequences. v4: speech in header only)

Output: `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/megatron_dataset_v4/` (160 files)

**Token rates per 8-frame chunk (verified Jul 2, 2026):**

| Modality | Tokens/chunk | Per 30s (after v3 dropout) |
|----------|-------------|---------------------------|
| Seed2 | 32 fixed (1 block per 3.75 chunks) | 30 × 32 = **960** |
| Cosmos | 200 fixed (every chunk) | ~56 × 200 = **11,200** |
| Agent | 171–579 (~280 typical) | up to 112 × 280 = **31,360** |
| SNAC | 9 or 12 (avg 10, alternating) | 112 × 10 = **1,120** |
| AVC-LM | 885–5,055 | **0** (dropped) |

**DATA PATHS (IMPORTANT — updated Jun 27, 2026):**  
JUPITER `/e/data1` is sometimes down (cluster maintenance). All critical data copied to `/p/`:
```
/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/
  ├── final_dataset_adaptive/final_vla_adaptive_rank_*.jsonl  ← INPUT for Phase 7
  ├── megatron_dataset_adaptive/flat_*.jsonl                  ← v1 flat output
  └── megatron_dataset_v2/flat_*.jsonl                       ← v2 flat output (pending)
```
Phase 7 script and SLURM now default to `/p/` paths.

---

### Tokenizer

**Script:** `tools/expand_vocab.py`, `tools/upload_tokenizer.py`

Extended GPT-NeoX-20b (50,277 tokens) with 93,938 VLA tokens using `tokenizer.add_tokens(special_tokens=True)`.

**The critical bug in the first model:** Editing `vocab.json` directly does NOT register BPE merge rules. The tokenizer split `<seed2_1137>` → 7 sub-pieces. Despite this, the first model showed signal (learned to predict sub-piece sequences) but was not decoding real tokens.

**The fix:** `add_tokens(special_tokens=True)` bypasses BPE merging, treating every VLA token as atomic.

Published: `EmpathicRobotics/tokenizer-vla-adaptive` (144,215 vocab, padded to 144,256 for Megatron).

---

### Phase 8: Megatron-LM Tokenization

Tokenized 160 JSONL files → 2 binary shards:

| Shard | Tokens | Size |
|-------|--------|------|
| `data_shard_00000.bin` | 2,684,323,146 | 10.00 GB |
| `data_shard_00001.bin` | 156,389,702 | 0.58 GB |
| **Total** | **2,840,712,848 (2.84B)** | **10.58 GB** |

---

### Phase 9: Training — Model 2 (June 2026)

**Model:** `EmpathicRobotics/vla-1.7b-pab-spline-adaptive`  
**Architecture:** OpenSci-Ref 1.7B (24 layers, 2048 hidden, 32 heads → **1.91B params** with 144K vocab embeddings)  
**Config:** `oellm-autoexp/config/experiments/nguyen38/vla_adaptive.yaml`  
**Compute:** 64 nodes × 4 GH200 = 256 GPUs, ~35 min wall time

Training schedule:
| Iter | Loss | LR | Tokens seen |
|------|------|----|------------|
| 200 | 2.982 | 4e-3 | 0.84B |
| 500 | 2.070 | 4e-3 | 2.10B |
| 1000 | 1.672 | 4e-3 | 4.19B |
| 2000 | 1.476 | 3.2e-4 | 8.39B |
| **2032 (val)** | **1.501** | — | — |

Val PPL: **4.49** | Test PPL: **4.45** | ~3 epochs over 2.84B tokens

---

### Data Inventory (June 26, 2026 — Complete)

**Script:** `tools/data_inventory.py` | **Checkpoint:** `tools/inventory_checkpoint_v2.json`

Scanned all 242 files across 4 dataset families:

| Dataset | seed2 | cosmos | avclm | agent | snac | text | **TOTAL** |
|---------|-------|--------|-------|-------|------|------|-----------|
| FineVideo-VLA (160 files) | 89.9M | 210.2M | 474.4M | 564.9M | — | 11.4M | **1.35B** |
| MV-Backup valid_with_seed (64 HF shards) | 5.6M | — | — | — | — | — | **5.6M** |
| MV-Backup stack_images3_gzip (12 archives) | 313K | — | — | — | — | — | **313K** |
| MV-Omni valid_snac (6 gzip files) | — | — | — | — | 4.92B | 1.99B | **6.93B** |
| **TOTAL** | **95.8M** | **210.2M** | **474.4M** | **564.9M** | **4.92B** | **2.00B** | **8.29B** |

**Key findings:**
- `valid_with_seed` (1.1 TB downloaded!) yields only 5.6M seed2 tokens — **negligible, not worth the storage cost**. Shards 0–30 contain only raw `.png`/`.ogg` with zero tokenized content. Only shards 31–63 have `_seed2.jsonl` inside inner archives.
- MV-Omni is the only substantial external source at 6.93B tokens. BUT `<snac_N>` and `<seed_N>` tokens are **not in the current tokenizer vocab** — blocked until vocab expansion.
- **Only FineVideo has agent (3D pose) tokens.** No external dataset contributes pose data.
- **Training-ready today: 1.35B tokens** (FineVideo only, with current vocab).

---

## Current State — What Works, What Doesn't

### Works
- Pipeline end-to-end: raw video → 3D pose → tokens → Megatron bin → training → deployable HF checkpoint
- All VLA tokens are atomic (tokenizer fix confirmed)
- Model correctly completes 17-joint agent blocks: right joint ordering, valid xyz/t values, decodable to 3D pose via PCHIP
- 3D pose decoder verified: model output → (8, 17, 3) trajectory in correct physical range

### Does NOT work yet
- **Autonomous modality transitions:** When prompted with only text, the model stays in seed2 mode and never transitions to cosmos/avclm/agent. It requires agent tokens in the prompt to continue in agent mode.
- **Root cause 1 — Data starvation:** 2.84B tokens for 1.91B params = ~1.5× Chinchilla ratio. Optimal is ~20×. Each training sample seen only ~3 times — enough for local pattern memorization, not high-level sequencing.
- **Root cause 2 — No visual language anchors:** Text is only Title/Context/Keywords. No captions describe what's happening at each timestamp. The model has no language signal to know "after these seed2 tokens, cosmos tokens come next."
- **Root cause 3 — Over-aggressive dropout:** 99% AVC-LM + 90% Cosmos dropout means most records lack the full transition chain. Model rarely sees seed2 → cosmos → avclm → agent in sequence.

---

## What's Next — Prioritized Roadmap

### Immediate priorities (code during any available time, no GPU needed)

**Priority 1 — Vocab expansion for SNAC tokens** ← PARTIALLY DONE
- ~~Convert MV-Omni `<seed_N>` → `<seed2_N>`~~ **DONE** (Jun 27, 2026)
  - Script: `data_prep/convert_mvomni_seed.py`
  - Output: `/p/data1/mmlaion/shared/vla/mv_omni_converted/mv_omni_snac_*.jsonl.gz`
  - 1,593,301 records | 19,249,664 seed tokens converted | 30 GB
  - `<seed_N>` tokens fully eliminated — zero remaining in output
- **REMAINING:** Add `<snac_0>` ... `<snac_4095>` (~4096 tokens) to tokenizer via `add_tokens(special_tokens=True)`
  - New vocab: ~148,311 tokens (no need for `<seed_N>` — already converted to `<seed2_N>`)
  - Unlocks MV-Omni's **6.93B tokens** for training
  - Effort: ~1 day

**Priority 2 — Adjust modality dropout in Phase 7** ← ~~DONE~~ (Jun 27, 2026)
- AVC-LM: 99% → **100% drop** (removed entirely)
- Cosmos: 90% → **50% drop** (keeps ~6/12 chunks per activity)
- Output: `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/megatron_dataset_v2/`
- Uploaded to `EmpathicRobotics/FineVideo-Phase7-Flattened` (v2 commit)
- **Next step:** Megatron re-tokenize `megatron_dataset_v2/` → new `.bin/.idx` shards → re-train v0.2

**Priority 3 — Ego-centric perspective for FineVideo**
- Read Phase 4 yolo_cleaned pose data
- Apply rotation matrix: place camera at `head_top` joint position, orient along thorax forward direction
- Generate additional agent token sequences from ego-centric view
- Same underlying motion data, double the data diversity (first-person + third-person)
- Effort: ~1 week code + 1 SLURM run

**Priority 4 — Write captioning pipeline code**
- Use `chunk_timing` timestamps to extract keyframes from FineVideo videos
- Pass each keyframe through SmolVLM2 or Qwen2.5-VL
- Interleave generated captions into the token sequence
- Expected impact: ×4 records with language anchors at every modality transition → fixes root cause 2
- Effort: 1–2 weeks code (GPU run on JUPITER is separate)

### Medium-term (needs dedicated GPU time on JUPITER)

**Priority 5 — Collect agent + cosmos + snac from Cosmos3-DROID**
- `nvidia/Cosmos3-DROID` on HuggingFace: robot arm manipulation videos with Cosmos video tokens
- Run YOLO + Phase 1–5 equivalent to extract agent tokens (robot arms/hands)
- Add SNAC tokens if audio track exists
- First robot-domain data — critical for generalization beyond human motion
- Hold off on AVC-LM until ablations confirm it helps (per Huu's guidance)

**Priority 6 — Vocab expansion (tokenizer build)** ← **COMPLETE (Jul 1, 2026)**

Script: `tools/build_tokenizers.py`. Hai output:
- `tokenizer_vla_adaptive_v2`: 144,215 (base) + 12,290 SNAC = **156,505 vocab**, tất cả atomic ✓
- `tokenizer_vla_qwen3`: ~151,669 (Qwen3) + 106,228 VLA = **257,897 vocab**, tất cả atomic ✓

**Priority 7 — SNAC tokenization for FineVideo** ← **COMPLETE (Jul 1, 2026)**

Job `snac_cpu_14077331`, 32 array tasks on `batch` partition (CPU), submitted from `jwlogin08`.

**Results:**
| Metric | Value |
|--------|-------|
| Tasks completed | **32/32** (100%) |
| Activities processed (ok) | **371,855** |
| fail_audio (no audio track) | 530 (~0.1%) |
| fail_snac | **0** |
| Total SNAC tokens | **363,029,331 (~363M)** |
| Output files | **40,779** `{video_id}_snac.jsonl` |
| Output size | **6.5 GB** |
| Output location | `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/snac_tokens/` |

**Output format** — one file per video, one JSONL line per activity:
```json
{
  "video_id": "abc123",
  "activity_id": "scene_1_act_1",
  "start_sec": 1.0,
  "end_sec": 8.9,
  "has_agent": true,
  "snac_by_chunk": {
    "0": ["<snac_132247>", "<snac_132788>", "<snac_147076>", ...],
    "1": [...],
    ...
  }
}
```

**Chunk splitting mechanism:**
- Encode full activity audio once (1 call to SNAC) → flat list of tokens
- SNAC output = sequence of base frames, each = exactly 3 tokens (L0 + L1_even + L1_odd triplet)
- `n_base = len(flat_tokens) // 3` — truncate to complete base frames (atomic unit, cannot split triplet)
- Proportional split: `start_base[k] = round(k * n_base / n_chunks)`, `end_base[k] = round((k+1) * n_base / n_chunks)`
- Each chunk gets `end_base - start_base` base frames × 3 = **9 or 12 tokens** (alternating due to 3.33 base frames/chunk)
- Last chunk is NOT shorter — `round(n_chunks × n_base / n_chunks) = n_base` exactly
- Temporal alignment error: ±1 base frame = ±80ms at each chunk boundary (acceptable for pretraining)

**Next steps unblocked by this completion:**
1. Vocab expansion — add 12,288 `<snac_N>` tokens to tokenizer
2. Re-run Phase 6 v2 with `--snac-tokens-dir`
3. Re-run Phase 7 v3 → `megatron_dataset_v3/`
4. Megatron re-tokenize → train v0.3

**CLUSTER ARCHITECTURE NOTE (discovered Jun 28, 2026):**
- `jwlogin08.juwels` = JUWELS Cluster login node (x86_64)
- `juwels-booster.fz-juelich.de` = JUWELS Booster login nodes (separate system, ppc64le compute)
- `laionize` account with GPU access (`booster` partition) is only usable from the Booster login nodes
- From JUWELS Cluster login, `laionize` only has CPU partitions: `batch`, `devel`, `large`
- **To submit GPU job: SSH to `juwels-booster.fz-juelich.de` first**

**Task list already built (Jun 28, 2026):**
```
/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/snac_task_list.json
→ 40,798 videos, 372,385 activities, all with chunk_timing
```

**Submission commands (when on juwels-booster login node):**
```bash
cd /p/data1/mmlaion/nguyen38/3d-human-pose

# GPU mode: 16 workers on booster partition, ~8-12h
bash slurm/submit_snac_finevideo.sh

# CPU fallback (from jwlogin, slower ~24h, no SSH needed):
bash slurm/submit_snac_finevideo.sh --cpu
```

**Run sequence after SNAC tokenization completes:**
```bash
# Step 1 — DONE: build task list
# snac_task_list.json already at TASK_CACHE path

# Step 2 — RUNNING (Jun 30): CPU batch job, 32 workers, ~20-24h
# Output: .../FineVideo-VLA/snac_tokens/{video_id}_snac.jsonl (~40K files)

# Step 3: Vocab expansion — add 12,288 <snac_N> tokens to tokenizer
# TODO: update tools/expand_vocab.py to include snac range [128266..148745]

# Step 4: Re-run Phase 6 with SNAC injection
python pipeline_pose/phase6_merge_adaptive.py \
  --input-glob "/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/training_ready_rank_*.jsonl" \
  --agent-tokens-dir /p/data1/mmlaion/shared/nguyen38/data/outputs/agent_tokens_adaptive \
  --snac-tokens-dir  /p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/snac_tokens \
  --output-dir       /p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/final_dataset_adaptive_v2 \
  --output-prefix    final_vla_adaptive_v2

# Step 5: Re-run Phase 7 → megatron_dataset_v3/
python pipeline_pose/phase7_flatten.py \
  --input-glob ".../final_dataset_adaptive_v2/final_vla_adaptive_v2_rank_*.jsonl" \
  --output-dir ".../megatron_dataset_v3" \
  --drop_cosmos 0.5 --drop_avc 1.0 --drop_snac 0.0 --workers 16

# Step 6: Megatron tokenize → .bin/.idx → train v0.3
```

**Priority 7 — Investigate leo seed2 + euro_pat**
- Check what's on the `leo` cluster: seed2 + euro_pat datasets mentioned by Huu
- Quantify token counts before committing storage/compute

**Priority 8 — First re-training run (v0.2)**
- After items 1, 2, 4 are done: estimated **10–20B tokens** available
- Continue training from current checkpoint (2032 iter) with new data + adjusted dropout
- Expected result: model begins to learn modality transitions autonomously

### Long-term (3–6 months)

**Priority 9 — More text data**
- Mix in standard LLM text data (to create language binding and prevent catastrophic forgetting)
- Target: text tokens at ~10–15% of total training mix

**Priority 10 — Qwen3 migration**
- Retokenize entire dataset with Qwen3-based expanded tokenizer
- Requires full re-run of Phase 8 (Megatron tokenization) and training from scratch
- Benefit: native HF ecosystem support, vLLM, llama.cpp compatibility
- Huu's config: cherry-picked from commit `7dcf8a5`

**Priority 11 — PAB-Spline spec upgrade**
- Current tokenizer: PCHIP xyz-only (positions)
- Spec calls for: joint angles (q/qd), phase variable φ ∈ [0,1], cyclic gait detection, static joint compression
- Blocked by: need to run kinematics pipeline again with angle computation

**Priority 12 — Isaac Sim integration**
- Generate Unitree H1 rollouts in Isaac Sim / ManiSkill
- Tokenize simulation data with PAB-Spline tokenizer
- Sim-to-real gap: map joint tokens → H1 control signals

---

## Data Landscape — Where We Are and What We Need

### Current training-ready data: 1.35B tokens (FineVideo only)
This is too small. For a 1.7B model, Chinchilla-optimal is ~34B tokens. We're at ~4% of that.

### Unlockable with vocab expansion only (no new collection): +6.93B tokens
MV-Omni valid_snac is sitting there, tokenized, but blocked by missing `<snac_N>` / `<seed_N>` vocab entries. Adding these two token families = 1–2 days of work = unlock 6.93B tokens = reach ~8.3B total. This is the highest-leverage action available right now.

### Unlockable with GPU runs: +5–10B tokens (captioning, ego-centric, Cosmos3-DROID)
The captioning pipeline alone multiplies FineVideo by ~4× (69,844 records → ~280K records) with richer language context. Ego-centric adds a second perspective for free.

### Target: 20–40B tokens for v0.2 training
With vocab expansion + MV-Omni + captioning + Cosmos3-DROID + SNAC-FineVideo, reaching 20–40B tokens is realistic within 2–3 months of focused work.

---

## Honest Assessment — Are We On The Right Track?

**Yes, the architecture is sound.** The second model proved the core hypothesis: a 1.7B LLM can learn the grammar of multimodal token sequences — joint ordering, valid xyz ranges, modality-specific token distributions — purely from next-token prediction on flat interleaved sequences.

**The bottleneck is data, not architecture.** The model's failure to autonomously transition between modalities is fully explained by data starvation and missing language anchors. These are solvable engineering problems, not fundamental flaws.

**The direction is genuinely novel.** No published work unifies Seed2 + Cosmos + SNAC + PCHIP pose tokens in a single autoregressive LLM context. The closest prior work (RT-2, OpenVLA, π0) uses much simpler action representations and doesn't attempt continuous 3D body pose. We're building something nobody else has built.

**The risks:**
1. **Scale gap:** Even at 20B tokens, we're far below frontier LLMs. Our model may generalize poorly to novel prompts. Mitigation: mix in standard text data to maintain language ability.
2. **No robot deployment yet:** Current pose data is from YouTube humans, not actual robot joints. Isaac Sim integration is still future work. The model won't directly control a real robot without sim-to-real adaptation.
3. **SNAC/audio quality:** Orpheus SNAC2 is "good enough" per Huu's assessment, but retokenizing with Moss Audio Tokenizer V2 (mentioned in chat, 2.1B decoder) could improve audio quality significantly. Deferred for now.
4. **Qwen3 migration overhead:** If we retokenize for Qwen3, existing `.bin/.idx` shards become obsolete. Should be done once, not multiple times — wait until the data landscape is more stable.

**What success looks like at each stage:**
- **v0.2 (2–3 months):** Model autonomously transitions from text prompt → seed2 → cosmos → agent tokens without needing agent tokens in the prompt.
- **v0.3 (4–6 months):** Model responds to spoken commands (SNAC) by generating valid agent motion tokens. "Walk forward" → valid pelvis/hip/knee trajectory.
- **v1.0 (6–12 months):** Model observes visual scene + receives instruction, generates motion that respects scene geometry. The chemical beaker test.

---

## Key Decisions Log

| Decision | Why | Date |
|----------|-----|------|
| Chose Adaptive PCHIP over fixed 409-token format | Self-describing, ~35% fewer tokens for static joints | Feb 2026 |
| Tokenizer fix via `add_tokens()` not vocab.json edit | BPE requires merge rules, not just vocab entries | May 2026 |
| 99% AVC-LM dropout in Phase 7 | AVC-LM was 373× more tokens than agent — would dominate context | Mar 2026 |
| valid_with_seed NOT worth using | 1.1 TB download for 5.6M tokens (< 0.5% of FineVideo) | Jun 2026 |
| Hold AVC-LM in new datasets until ablations | No evidence yet that it helps vs adds noise | Jun 2026 |
| Ego-centric perspective as free data multiplier | Same underlying motion, different reference frame, doubles diversity | Jun 2026 |
| Qwen3 migration deferred | Too early — data landscape still changing | Jun 2026 |
| MV-Omni: convert seed→seed2 instead of adding new vocab | Avoids unnecessary vocab expansion; seed_N and seed2_N are identical semantics | Jun 2026 |
| SNAC injection in Phase 6, not Phase 7 | Phase 6 already does per-chunk injection; Phase 7 is stateless flatten. Keeping injection in Phase 6 means Phase 7 needs no external lookups. | Jun 2026 |
| SNAC chunk alignment: encode full activity once, split by count | Encoding per-chunk (0.267s segments) would lose audio context + slow due to many small calls. Encode once, split evenly preserves context and is accurate (SNAC rate is constant). | Jun 2026 |
| SNAC for ALL activities, not just agent ones | Only 14% of activities have agent tokens. Other 86% still have seed2+cosmos — adding SNAC teaches seed2→cosmos→snac transitions. Filtering to agent-only wastes most of the GPU run. | Jun 2026 |

---

## Published Artifacts

| Artifact | Location | Status |
|----------|----------|--------|
| Tokenizer v1 (144,215 vocab, GPT-NeoX) | `EmpathicRobotics/tokenizer-vla-adaptive` | Live |
| **Tokenizer v2 (156,505 vocab, GPT-NeoX + SNAC)** | `EmpathicRobotics/tokenizer-vla-adaptive-v2` | **Live (Jul 1, 2026)** |
| **Tokenizer Qwen3 (257,897 vocab)** | `EmpathicRobotics/tokenizer-vla-qwen3` | **Live (Jul 1, 2026)** |
| FineVideo-Phase7-Flattened v4 (371,888 records, 5.217B tokens) | `EmpathicRobotics/FineVideo-Phase7-Flattened` | **Pending upload** |
| FineVideo-Phase5-AgentTokens (~399K activities) | `EmpathicRobotics/FineVideo-Phase5-AgentTokens` | Live |
| FineVideo-Phase4-YOLOPose (millions of windows) | `EmpathicRobotics/FineVideo-Phase4-YOLOPose` | Live |
| VLA Model v1 (broken tokenizer) | `EmpathicRobotics/vla-1.7b-pab-spline-25b-test` | Live (deprecated) |
| VLA Model v2 (fixed tokenizer) | `EmpathicRobotics/vla-1.7b-pab-spline-adaptive` | Live |
| Megatron .bin/.idx shards (2.84B tokens) | `/p/data1/mmlaion/shared/vla/tokenized_output/vla_adaptive/` | Local |
| Data inventory checkpoint | `tools/inventory_checkpoint_v2.json` | Local |

---

## Environments & How to Run (JUWELS login node)

**env_tools** — use for: phase7_flatten, data_inventory, HF uploads, eval, any non-GPU script  
Location: `/p/data1/mmlaion/nguyen38/env_tools`  
Python: 3.12.3 | Has: torch, transformers, wn, datasets, scipy, huggingface-hub, rich, tqdm, ...

> **Note (Jun 27, 2026):** env_tools was created on JUSUF but we run on JUWELS. Python symlink and pyvenv.cfg had wrong paths. Fixed via `load_env_tools.sh`.

```bash
# Activate env_tools (source it, don't bash it — needs to modify your shell):
source /p/data1/mmlaion/nguyen38/3d-human-pose/load_env_tools.sh
# → auto-fixes symlinks on first run, then activates

# Then run whatever you need:
python pipeline_pose/phase7_flatten.py --workers 16 --skip-existing
```

**env_pose** (miniforge3 conda) — use for: phases 1–6 (HRNet, MotionBERT, YOLO, kinematics)  
Location: `/p/data1/mmlaion/nguyen38/3d-human-pose/env_pose`  
Activate: `source /p/data1/mmlaion/nguyen38/3d-human-pose/miniforge3/etc/profile.d/conda.sh && conda activate /p/data1/mmlaion/nguyen38/3d-human-pose/env_pose`

**Data paths on `/p/` (use these when JUPITER `/e/` is down):**
```
/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/
  ├── final_dataset_adaptive/final_vla_adaptive_rank_*.jsonl  ← Phase 7 input
  ├── megatron_dataset_adaptive/flat_*.jsonl                  ← v1 flatten output
  └── megatron_dataset_v2/flat_*.jsonl                       ← v2 flatten output (pending)

/p/data1/mmlaion/shared/vla/
  ├── mv_omni_converted/mv_omni_snac_*.jsonl.gz              ← MV-Omni seed→seed2 done
  ├── tokenizer_vla_adaptive/                                 ← local tokenizer copy
  └── tokenized_output/vla_adaptive/data_shard_*.bin/.idx    ← Megatron shards (2.84B tokens)
```

---

## Immediate Action Items (Next 2 Weeks)

### Đã hoàn thành (Jun–Jul 2026)
- [x] **SNAC CPU job** — **COMPLETE (Jul 1, 2026)**. Job `snac_cpu_14077331`, 32/32 tasks. 371,855 activities, 363M tokens, 6.5 GB → `/p/.../snac_tokens/`
- [x] **Dataset overlap check** — **COMPLETE (Jun 30, 2026)**. Kết quả: 27,359 video chồng nhau (86.9% of valid_with_seed ∈ omni_valid). Xem section "Dataset Overlap Analysis" bên dưới.
- [x] **Vocab expansion (tokenizer build)** — **COMPLETE (Jul 1, 2026)**. Script: `tools/build_tokenizers.py`. Tạo 2 tokenizer:
  - `tokenizer_vla_adaptive_v2` (GPT-NeoX-20b + SNAC): **156,505 vocab** → `/p/data1/mmlaion/shared/vla/tokenizer_vla_adaptive_v2/`
  - `tokenizer_vla_qwen3` (Qwen3 + tất cả VLA tokens): **257,897 vocab** → `/p/data1/mmlaion/shared/vla/tokenizer_vla_qwen3/`
  - Spot-check 12 tokens đại diện: tất cả **atomic** (1 token/ID), không có sub-piece splitting
  - SNAC token range: L0 [128266..132361], L1A [132362..136457], L1B [144650..148745]

### Pre-training Discussion Items (Jul 2, 2026 — from Huu chat)

> **⚠ Huu explicitly said: "Before you train let's talk." — do NOT start training until these 3 items are resolved.**

**[DISCUSS-1] Language data mix — what to add before training?**
- Current plan (FineVideo v4 + MV-Omni = 12B tokens) has almost no instruction/language data
- Huu: "mix in a few billion tokens mixture so we can steer the robot better"
- Huu: "We should look for some SFT dataset for our various target (pick up the Apple, Drive left, etc.)"
- Candidates on leo (`/mnt/sdb/mixture-vitae-working/`): `clappa_text_only`, `coco` (synthetic permissive), `misc_instr/hpprc-r1-distill-qwen-pseudo-qa.jsonl` (Japanese instruction)
- Also wants multilingual instruction datasets with reasoning/thinking
- **Action needed:** Identify token counts of available language datasets → decide mix ratio

**[DISCUSS-2] Compression analysis of Adaptive PCHIP — RESULTS READY**
- Huu: "Did you do an analysis by how much compression we got? If there is no or low compression then we know it's wrong."
- **DONE:** `tools/analyze_pchip_compression.py` — 18,847 files, 1,743,189 windows. Results:
  - **50.9% token saving** vs fixed 8-CP (284.1 avg vs 579)
  - CP tiers: 55.2% 2-CP / 25.6% 4-CP / 19.2% 8-CP
  - Most dynamic: r_knee (33.5% 8-CP), r_wrist (29.4%). Most static: pelvis (100% 2-CP)
  - Pelvis confirmed at origin: 500/500 samples within ±0.1m ✓
  - Coordinate system: absolute xyz after root-centering is correct
- **NEW — Pose data quality concern (Jul 2, 2026):**
  - Only **4–7 joints finite per frame** (out of 17) — 24–41% skeleton coverage
  - **Arms (j11–j16) nearly always NaN** — MotionBERT cannot reliably lift arm joints from YouTube (occlusion/side views)
  - **head_top (j10) zero-fill artifact** — stores (0,0,0) when undetected, same as pelvis, counted as finite but wrong
  - Impact: model learns lower-body + torso pose only. Fine for pretraining pose presence; NOT sufficient for arm/hand manipulation learning
- **Action needed:** Report numbers + pose quality concern to Huu

**BEAST comparison (Jul 3, 2026) — context for reporting to Huu:**

Huu asked: *"What does the BEAST paper say about their compression? That will give us a sanity check."*

BEAST = "B-spline Encoded Action Sequence Tokenizer" (KIT, NeurIPS 2025, arXiv 2506.06072). Uses B-splines with fixed N control points fit by ridge regression. Claims **4–8× compression** vs binning (e.g., ACT 100-step chunk → 15 tokens = 6.67×).

**Why our 50.9% looks lower:** different baselines.

| | Baseline | Result |
|---|---|---|
| **BEAST** | Binning (1 token/timestep/DoF) | 4–8× fewer tokens |
| **Ours** | Fixed 8-CP (already compressed) | ~2× fewer tokens |

Vs raw binning: our 284 tokens / (8×17×3=408 raw values) = **~1.5×** — much less than BEAST. Root cause: 34% of our tokens are overhead (wrappers + t tokens) to make the format self-describing for the LLM. BEAST has zero overhead (decoder structure is hardcoded). Self-describing is a deliberate design choice for LLM joint-semantic learning.

**Huu's 1-CP suggestion (Jul 3, 2026):** *"Why do we have 2-CP as minimum? Can we have 1-CP? Like — relative, no movement."*

Why 2-CP was minimum originally:
1. PCHIP requires ≥2 points (it's an interpolating polynomial — 1 point = nothing to interpolate)
2. "Low curvature" ≠ "no movement": joint may drift linearly within the window below tau_low

How 1-CP would work: if `quantize(frame_0) == quantize(frame_7)` for all 3 dims → emit only `<joint_x_N> <joint_y_N> <joint_z_N>` (no t token, 3 tokens vs current 8 tokens)

Estimated gain: ~4–5 qualifying joints/window × 5 tokens saved ≈ 20–47 tokens/window → **additional ~8–16% compression**. Requires grammar change + re-run of Phase 5 and all downstream phases.

**[DISCUSS-3] Eval setup**
- Huu: "We should start eval just to see how things perform with baseline"
- Need to define eval tasks BEFORE training, not after
- Candidates: agent token decode quality (MPJPE on 3D pose), modality transition accuracy, instruction-following on robot commands
- **Action needed:** Define eval protocol and implement baseline metrics

---

### Việc tiếp theo (unblocked bởi vocab expansion)
- [x] **Phase 6 v2 dry run** — **COMPLETE (Jul 1, 2026)**. Chạy thử 1 file (254 videos, ~5 phút). Kết quả:
  - SNAC inject: **259,503/259,505** avc blocks (~100%)
  - Agent inject: **12,705** blocks (đúng — 46% video có Phase 5 output)
  - Format verified: `</avc_lm> <agent>...</agent> <snac> <snac_N>... </snac>` ✓
  - `chunk_timing` có đủ các flag `has_seed2/cosmos/avc_lm/agent/has_snac` ✓
  - SLURM script mới: `slurm/submit_merge_adaptive_v2.sh` (account `laionize`, partition `batch`, 32 workers, 2h)
  - Ước tính toàn bộ 160 file với 32 workers: **~25–40 phút**
- [x] **Re-run Phase 6 v2** — **COMPLETE**. Job `14082096`, 32/32 workers. 40,804 videos | 398,775 activities | SNAC 100% | Agent 5.5% | 0 errors → `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/final_dataset_adaptive_v2/` (160 files)
- [x] **Re-run Phase 7 v3** — **COMPLETE (Jul 2, 2026)**. 160/160 files, 371,888 records, 72 GB → `megatron_dataset_v3/`
  - Full-chain (agent+snac): 69,811 (18.8%) | Snac-only: 302,044 (81.2%) | Bad records: 0
  - Token counts: seed2 332.6M | cosmos 3.88B | snac 363M | agent windows 2,148,474 | avclm 0 ✓
  - Sample: `samples/after_flatten_v3.json` | Upload script updated: `tools/upload_flattened_hf.py`
- [x] **Phase 7 v4 — temporal alignment fix** — **COMPLETE (Jul 2, 2026)**. Per-chunk ordering fixed, speech in headers, 5.217B tokens → `megatron_dataset_v4/`. See stats above.
- [ ] **Upload Phase 7 v4 to HF** → `EmpathicRobotics/FineVideo-Phase7-Flattened`:
  ```bash
  export HF_TOKEN='hf_...'
  python tools/upload_flattened_hf.py
  ```
  Source: `megatron_dataset_v4/` | Upload dir: `hf_upload_flattened_v4/` | Dataset card: `tools/vla_flattened_dataset_card.md` (updated for v4)
- [ ] **Megatron re-tokenize** `megatron_dataset_v4/` with `tokenizer-vla-adaptive-v2` (156,505 vocab) → new `.bin/.idx` → train v0.3
- [x] **Upload tokenizers** — **COMPLETE (Jul 1, 2026)**. `EmpathicRobotics/tokenizer-vla-adaptive-v2` (156,505) + `EmpathicRobotics/tokenizer-vla-qwen3` (257,897), cả hai Live với model card đầy đủ

### Kết luận overlap check (Jun 30, 2026 — XONG)
- [x] Quyết định **KHÔNG dùng `valid_with_seed`** — 86.9% đã có trong omni_valid, 13.1% còn lại (4,141 video) chỉ có seed2 token và không đáng tốn thêm storage/compute
- [x] omni_valid **không overlap với FineVideo-VLA** (khác nguồn hoàn toàn — MixtureVitae vs YouTube FineVideo)

### Coding (không cần GPU)
- [ ] Start writing ego-centric perspective converter (Phase 3 → rotate to head camera)
- [ ] Start writing captioning pipeline (SmolVLM2/Qwen2.5-VL trên FineVideo keyframes)
- [ ] Investigate leo seed2 + euro_pat token counts
- [ ] Plan Cosmos3-DROID pipeline (download strategy, SLURM script)

---

## Dataset Overlap Analysis (Jun 30, 2026)

### Background

Từ cuộc họp Jun 28: Huu chỉ ra rằng `omni_valid` có khả năng được subsample từ `valid_with_seed`, dẫn đến **double-counting** nếu dùng cả hai. Kết luận ban đầu: chỉ dùng 3 dataset:
1. **FineVideo-VLA** (local)
2. **omni_valid** (MixtureVitae-Omni)
3. **stack_images3_gzip** (MixtureVitae-Backup)

Còn `valid_with_seed` cần check xem overlap với `omni_valid` bao nhiêu % trước khi quyết định có dùng hay không.

### Data đã download

| Dataset | Location | Format |
|---------|----------|--------|
| `valid_with_seed` | `/p/data1/mmlaion/nguyen38/inventory_cache/hf_shards/` | 64 outer `.tar.gz`, mỗi cái chứa file `.ogg`, `.png`, `_seed2.jsonl` |
| `omni_valid` | `/p/data1/mmlaion/nguyen38/inventory_cache/hf_snac/` | 6 `valid_snac_N.jsonl.gz` |
| `ontocord/VALID` | `multimodal/head.txt` | Chỉ có head sample (5 records), chưa download full |

### Script overlap check

**Script:** `tools/check_dataset_overlap.py`

**Logic:**
- `valid_with_seed`: Extract YouTube video ID (11 chars đầu) từ tên file trong các tar.gz
- `omni_valid`: Extract `params.id` từ metadata của mỗi JSONL record
- So sánh hai set, tính % overlap

**Command để chạy:**
```bash
cd /p/data1/mmlaion/nguyen38/3d-human-pose
python3 tools/check_dataset_overlap.py
```

Không cần env đặc biệt — chỉ dùng stdlib Python (tarfile, gzip, json, re).

**Output:** In kết quả ra màn hình + lưu vào `tools/dataset_overlap_results.json`

### Kết quả (HOÀN THÀNH — Jun 30, 2026)

Script `tools/check_dataset_overlap.py` đã chạy xong. Kết quả lưu tại `tools/dataset_overlap_results.json`:

| Metric | Số liệu |
|--------|---------|
| `valid_with_seed` unique video IDs | **31,500** |
| `omni_valid` unique video IDs | **238,539** |
| Overlap (cả hai) | **27,359** (86.9% của seed / 11.5% của omni) |
| Chỉ có trong `valid_with_seed` | **4,141** |
| Chỉ có trong `omni_valid` | **211,180** |

**Kết luận: KHÔNG dùng `valid_with_seed`.** omni_valid đã cover 86.9% video của nó. 4,141 video còn lại chỉ có seed2 token và không đủ giá trị (tổng < 700K token) để bù cho 1.1 TB storage.

Log: `tools/overlap_run.log` | Kết quả JSON: `tools/dataset_overlap_results.json`

**Lưu ý quan trọng:** Script phải mở **2 tầng tar** (outer tar → inner tar → files) vì shards 0–30 chỉ chứa inner tar.gz bên trong, không có loose files. Script v1 bị lỗi (0 IDs) do bỏ qua inner tar — đã fix trong v2.

### Cấu trúc dataset (đã verify)

**omni_valid record format:**
```json
{
  "text": "<listen><snac_N>...<snac_N></listen>\n<see><seed_N>...</see>\n...",
  "metadata": "[{\"source\": \"grass-yt-cc-by.{YT_ID}|...\", \"params\": \"{\\\"id\\\": \\\"{YT_ID}\\\", ...}\"}]"
}
```
→ YouTube ID ở `metadata[0].params.id` (11 ký tự)

**valid_with_seed shard format:**
- Outer tar chứa: `shard_NNNNN.tar.gz` (inner) + loose files (`*.ogg`, `*.png`, `*_seed2.jsonl`)
- File name format: `{YT_ID_11chars}_{clip_num}[_{crop}][_seed2].{ext}`
- YouTube ID = 11 ký tự đầu của filename

**ontocord/VALID format** (head.txt — 3 lines per record):
```jsonl
{"file_name": "-mbDQC0y0PY_6.ogg", "media_type": "audio", "text": "...", "snac_token": [...]}
{"file_name": "-mbDQC0y0PY_6.png", "media_type": "image", "text": "..."}
{"emotion": "...", "query": "...", "answer": "...", "shard_idx": "shard_0"}
```

### Câu hỏi đã trả lời (Jun 30, 2026)

1. **Bao nhiêu % của `omni_valid` đến từ `valid_with_seed`?**
   → 11.5% (27,359/238,539). omni_valid chủ yếu là data riêng, KHÔNG phải subsample từ valid_with_seed như Huu dự đoán — thực ra ngược lại: valid_with_seed là subset của omni_valid.

2. **`valid_with_seed` có video không có trong `omni_valid` không?**
   → Có: 4,141 video (13.1% của valid_with_seed). Nhưng những video này chỉ có seed2 token, không có SNAC, và tổng token ~700K — không đủ giá trị để dùng riêng.

3. **`ontocord/VALID` có tương đương với `valid_with_seed` không?**
   → Chưa check (chỉ có head sample 20 records → 0 video ID). Không cần điều tra thêm vì đã quyết định không dùng valid_with_seed.

### Kết luận cuối cùng

**Chỉ dùng 2 nguồn external:**
- **omni_valid (MV-Omni)** — 238,539 video, 6.93B token (SNAC + text + seed2). Cần vocab expansion `<snac_N>`.
- **stack_images3_gzip** — 313K token seed2. Quá nhỏ nhưng không tốn gì thêm nếu đã có sẵn.

**Bỏ valid_with_seed.** 1.1 TB đã download có thể xóa để giải phóng storage.
