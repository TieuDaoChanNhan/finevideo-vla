# FineVideo-VLA: Full Project Report

**Author:** Van Khue Nguyen  
**Date:** June 2025 – June 2026  
**Cluster:** JUPITER (JSC), `booster` partition, GH200 nodes

---

## Repo Reorg Note (Jul 9, 2026)

`tools/` was split into subfolders (`upload/`, `tokenizer/`, `inventory/`, `eval/`, `visualize/`, `analysis/`, `extract/`) and ambiguously-named dirs were renamed (`multimodal/` → `investigations/mixturevitae_multimodal/`, `data_prep/` → `investigations/mv_omni_seed_conversion/`, `test/` → `manual_checks/`; `dev/` archived). **Script paths referenced below in older sections reflect the pre-reorg flat `tools/` structure** — e.g. `tools/data_inventory.py` is now `tools/inventory/data_inventory.py`. See the updated root `README.md` for the current layout.

---

## Pre-training Blockers (Jul 2, 2026, updated Jul 8, 2026)

Three items must be resolved before the next training run (per Huu's directive):

1. **Language data mix** — add ~few billion tokens of instruction/caption data alongside FineVideo v4 + MV-Omni. Candidates: clappa, synthetic COCO, robot SFT datasets, multilingual instruction. **Still open** — need to count tokens of candidates and decide mix ratio.
2. **PCHIP compression analysis** — quantify token saving of adaptive vs fixed 8-CP; confirm coordinate system (absolute xyz vs delta-to-pelvis). **Done** — 50.9% saving vs fixed 8-CP confirmed (Section 2.2, Phase 5). 1-CP follow-up investigated and **deferred** (see Phase 5 section below).
3. **Eval setup** — define baseline eval protocol (MPJPE, modality transition, instruction-following) before training. **Still open.**

**Additional blocker as of Jul 8, 2026: JSC cluster outage.** JUPITER down since ~Jul 6, 2026. JUWELS booster + JURECA have partial GPU availability. ETA per Huu: officially 1 week, realistically ~2 weeks. This blocks Megatron re-tokenization at scale and any training run regardless of data readiness.

---

## 1. Goal

Build a multimodal Vision-Language-Action pretraining dataset from ~40K YouTube videos (HuggingFace [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo)). The final output is a Megatron-LM-ready flat JSONL dataset where each record interleaves five token modalities:

- **Seed2** — semantic keyframe tokens (1 FPS, vocab 8192)
- **Cosmos** — spatial video tokens (every 8 frames, vocab 64000)
- **AVC-LM** — H.264 BPE tokens (every 8 frames, vocab 8192)
- **Agent** — 3D human pose tokens (every 8 frames, adaptive PCHIP, 17 joints)
- **SNAC** — audio tokens in listen format (~10 tokens per 8-frame chunk, 12,288 vocab) ← *tokenization complete (Jul 1, 2026)*

---

## 2. What Was Done

### 2.1 Branch A: Video Token Extraction (prototype pipeline)

**Script:** `pipeline_video/pipeline.py`  
**Compute:** 40 SLURM nodes × 4 GPUs = 160 GPUs  

Processed all ~40K FineVideo videos:
- Extracted frames at 30fps
- Tokenised each activity segment with Seed2 (1fps keyframes), Cosmos (8-frame spatial), and AVC-LM (8-frame H.264 BPE)
- Output: 160 `training_ready_rank_*.jsonl` files with hierarchical JSON (video → scenes → activities → tokens)

Each activity contains: `text_prompt`, `speech_transcript`, `video_tokens` (with `<seed2>`, `<cosmos>`, `<avc_lm>` blocks).

### 2.2 Branch B: 3D Human Pose Pipeline

#### Phase 1 — 2D Pose Detection (HRNet + Faster R-CNN)
**Script:** `pipeline_pose/phase1_hrnet_gpu.py` | **SLURM:** `slurm/submit_hrnet.sh`

- Ran HRNet with Faster R-CNN person detection on all videos
- Output: `outputs/2d_json/{video_id}_2d.json` — 2D joint coordinates per frame
- **40,804 videos** processed, **145 GB**

#### Phase 2 — 3D Pose Lifting (MotionBERT)
**Script:** `pipeline_pose/phase2_motionbert_gpu.py` | **SLURM:** `slurm/submit_motionbert.sh`

- Lifted 2D poses to 3D using MotionBERT (pretrained on Human3.6M)
- Processed at native video fps
- Output: `outputs/3d_npy/{video_id}.npy` — 3D joint arrays
- **40,804 videos**, **259 GB**

#### Phase 2.5 — 30fps Resampling
**Script:** `pipeline_pose/phase2_5_resample_30fps.py` | **SLURM:** `slurm/submit_phase2_5.sh`

- Resampled all 3D poses from native video fps to uniform 30fps via linear interpolation
- Required so pose tokens align to the same time grid as Seed2/Cosmos/AVC-LM (all at 30fps)
- Output: `outputs/3d_npy_30fps/{video_id}.npy`
- **40,804 videos**, **67 GB**

#### Phase 3 — Kinematics Processing
**Script:** `pipeline_pose/phase3_kinematics_processor.py` | **SLURM:** `slurm/submit_kinematics.sh`

- Applied temporal smoothing (Butterworth filter)
- Bone length normalisation to canonical Human3.6M skeleton
- Root centering (pelvis at origin)
- Anti-teleportation filter (removes sudden jumps)
- Windowed into 8-frame chunks with position/velocity/acceleration
- Output: `outputs/states_jsonl_30fps/{video_id}_states.jsonl` — shape `(windows, 8, 153)`
- **40,200 videos** (604 dropped due to too-short sequences), **193 GB**

#### Phase 4 — YOLO Person-Presence Cleaning
**Script:** `pipeline_pose/phase4_yolo_cleaner.py` | **SLURM:** `slurm/submit_yolo.sh`

- Ran YOLOv8 person detection on original video frames
- Dropped any 8-frame window where ≥ 4 frames have no detected person (confidence ≥ 0.75)
- Removes windows where subject is off-screen, occluded, or in scene transitions
- Output: `outputs/yolo_cleaned_30fps/{video_id}_cleaned.jsonl`
- **40,195 videos**, **107 GB**

**Data quality analysis (Jul 2, 2026):**

Direct inspection of `yolo_cleaned` data via `tools/visualize_skeleton_sidebyside.py` + per-window statistics revealed:

| Issue | Finding |
|-------|---------|
| Joint sparsity | **4–7 finite joints per frame** out of 17 (24–41% skeleton coverage) |
| Arms absent | j11–j16 (both arms: shoulder/elbow/wrist) = **NaN in nearly 100% of frames** — MotionBERT cannot reliably lift arm joints from YouTube footage due to occlusion and side-view ambiguity |
| Zero-fill artifact | j10 (head_top) stores **(0,0,0) when undetected** — coincides with pelvis origin, is counted as finite by `~np.isnan()` but is anatomically wrong |
| Coordinate scale | Ankle at −0.638 m below pelvis is anatomically plausible; metric scale is correct |

**Root cause:** Monocular 2D→3D lifting (MotionBERT) has fundamental depth ambiguity, especially for distal joints (hands, feet) under occlusion. "In the wild" YouTube videos are a much harder setting than controlled Human3.6M studio recordings (MotionBERT's pretraining domain).

**Training impact:** Pose tokens are predominantly lower body (hip/knee/ankle) + torso spine. This is sufficient as a weak pretraining signal for learning video-pose correlation — even noisy partial skeletons teach the model that certain visual motion patterns co-occur with particular joint configurations. However, for downstream robot manipulation fine-tuning (arm/hand control), higher-quality pose data (simulation, MoCap, depth cameras, or 4D-Humans-style fitting) will be needed.

#### Phase 5 — Adaptive PCHIP Tokenisation
**Script:** `pipeline_pose/phase5_adaptive_pchip.py` | **SLURM:** `slurm/submit_phase5_adaptive.sh`

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

**Why 2-CP is the minimum (not 1-CP):** PCHIP is an *interpolating* polynomial — it needs ≥2 points to construct a curve. With 1 point there is nothing to interpolate. More importantly, "low curvature" ≠ "no movement": a joint below tau_low may still drift linearly (e.g., 15mm) from frame 0 to frame 7. 2-CP captures that drift accurately via linear interpolation between the two endpoints; 1-CP would wrongly assume a constant value.

**NaN handling (important for understanding coverage):** `process_file()` skips any window where `np.isnan(states).any()` — i.e., if *any* of the 17 joints has NaN in *any* frame, the entire window is discarded. This is why REPORT Section 8 says "All 17 joints present in every record" — it is true, but only because windows with missing joints are filtered out entirely. The 18,847 videos represent the subset of FineVideo where at least some 8-frame windows had all 17 joints finite simultaneously. Arm joints (j11–j16) are NaN in nearly all YouTube frames, so agent tokens in practice encode lower-body + torso motion only.

**Compression analysis results (Jul 2, 2026):** `tools/analyze_pchip_compression.py` — 18,847 files, 1,743,189 windows:
- **50.9% token saving** vs fixed 8-CP (284.1 avg vs 579 max)
- CP tiers: 55.2% 2-CP / 25.6% 4-CP / 19.2% 8-CP
- Most dynamic: r_knee (33.5% 8-CP), r_wrist (29.4%). Most static: pelvis (100% 2-CP)

**Comparison with BEAST (Jul 3, 2026):**

BEAST ("B-spline Encoded Action Sequence Tokenizer", KIT, NeurIPS 2025, arXiv 2506.06072) uses B-splines with a *fixed* N control points fit via ridge regression, and claims **4–8× compression** vs binning-based tokenization (e.g., 100-step action chunk → 15 control points = 6.67×).

The 50.9% figure above is not directly comparable to BEAST's 4–8× because the baselines differ:

| | Baseline | Compression |
|---|---|---|
| **BEAST** | Binning: 1 token/timestep/DoF | 4–8× (75–87% fewer tokens) |
| **Ours** | Fixed 8-CP PCHIP (already compressed) | ~2× (51% fewer tokens) |

If compared against raw binning (1 token/frame/dim/joint):
- Raw: 8 frames × 17 joints × 3 dims = **408 scalar values**
- Our adaptive avg: 284 tokens (including 35 wrapper tokens + ~62 t tokens + ~187 xyz tokens)
- **Compression vs raw binning: ~1.5× — far below BEAST's 4–8×**

The gap is explained by overhead: our format is **self-describing** (joint name embedded in token string). This gives the LLM semantic grounding ("pelvis" = body center, "r_wrist" = end of arm) but costs 34% of every token budget as overhead (wrappers + t tokens). BEAST has zero overhead because its decoder has the structure hardcoded.

| Aspect | Ours (Adaptive PCHIP) | BEAST |
|--------|----------------------|-------|
| Spline type | PCHIP (exact interpolation) | B-spline (ridge regression) |
| CP count | Adaptive: 2/4/8 per joint | Fixed N for all joints |
| Fitting | Curvature heuristic | Optimal least-squares |
| Tokens/static joint | 10 (2 wrappers + 2 t + 6 xyz) | N×3 (e.g. N=3 → 9, no overhead) |
| Overhead | ~97/284 = **34%** | **0%** |
| Format | Self-describing (joint name in token) | Position-indexed (hardcoded structure) |
| Variable length | Yes (complicates LLM learning) | No (fixed → parallel decode) |
| Compression vs raw | ~1.5× | 4–8× |

**1-CP proposal (Huu, Jul 3, 2026):** For joints where both endpoints quantize to identical values — `quantize(frame_0) == quantize(frame_7)` for all 3 dims — 2-CP is redundant. A single xyz triple with no t token suffices:

```
# Current 2-CP (8 tokens/joint):
<pelvis_t_0> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128>
<pelvis_t_7> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128>

# Proposed 1-CP (3 tokens/joint, implied constant):
<pelvis_x_128> <pelvis_y_128> <pelvis_z_128>
```

Estimated impact: ~55% of joint-windows at tier 2-CP; if ~half qualify as truly static → ~4–5 joints/window × 5 tokens saved ≈ **20–47 tokens/window** → additional ~8–16% compression on top of current 50.9%. Grammar change required (decoder distinguishes 1-CP by absence of `<joint_t_N>` after open tag). Would break backward compatibility with existing Phase 5 output — requires re-run of Phase 5 and all downstream phases.

**Validated estimate (Jul 3–4, 2026):** `tools/analyze_cp_tradeoff.py` on 50 videos / 1,940 windows confirmed the targeted (static-only) 1-CP approach is safe — it only collapses a joint to 1-CP when `quantize(frame_0) == quantize(frame_7)`, meaning no additional reconstruction error is introduced (by construction, any drift is below the quantization step). Measured: 53.6% of tier-2 (already low-curvature) joint-windows qualify, ~4.1 qualifying joints/window, saving ~20 tokens/window (284 → 264 avg), for **+7.1% additional compression**. This is distinct from naively forcing 1-CP on *every* joint (the raw N=1 row in the sweep table above), which gives a much worse 24.3mm MAE because it also collapses genuinely moving joints — that global approach was never proposed for production use.

**Final decision (Jul 8, 2026): deferred.** Confirmed with Huu via Discord — keep the current adaptive 2/4/8-CP format in production. A full-dataset validation run (18,847 videos) was attempted but interrupted by the JUWELS cluster outage and not resumed. The +7.1% gain does not justify a full Phase 5→6→7 re-run at this time; revisit only if later evidence shows it's necessary. For reporting purposes (e.g. a paper), "compression reduces token count by more than 50% vs fixed 8-CP" is the number the team is comfortable citing as-is.

**Previous iterations (superseded):**
- `phase5_interpolation_tokenizer.py` — 256 opaque uint8 tokens per chunk (scale + anchor + motion CPs). Abandoned because tokens were not self-describing.
- `phase5b_xyzt_tokenizer.py` — 409 fixed tokens per chunk (all 8 frames × 17 joints × 3 dims). Clear and self-describing but wasteful for static joints.

### 2.3 Merge (Phase 6)
**Script:** `pipeline_pose/phase6_merge_adaptive.py` | **SLURM:** `slurm/submit_merge_adaptive.sh`

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
    "has_agent": true,
    "has_snac": false
  }
  ```
- Added `timing_meta` with fps and rate info for each modality
- Output: 160 `final_vla_adaptive_rank_*.jsonl` files, **657 GB** total
- **~399K activities** across all videos, **~2.15M agent blocks** injected

**Phase 6 v2 (Jun 28, 2026) — SNAC injection support:**

Added `--snac-tokens-dir` argument. When provided, Phase 6 reads per-activity SNAC output from `snac_finevideo.py` and injects `<snac>...</snac>` blocks alongside agent tokens in a single pass over `video_tokens`. Token order per 8-frame chunk becomes:

```
<cosmos>...</cosmos> <avc_lm>...</avc_lm> [<agent>...</agent>] [<snac>...</snac>]
```

SNAC rate alignment: SNAC listen format produces 37.5 tokens/sec. Each 8-frame chunk at 30fps = 0.267s → ~9–10 SNAC tokens per chunk (3.33 base frames × 3 tokens/frame). `snac_finevideo.py` encodes the full activity audio once (preserving temporal audio context) then splits tokens evenly across chunks, snapping to 3-token boundaries (1 SNAC base frame = 3 tokens).

Running without `--snac-tokens-dir` is backward compatible with v1 behavior.

**Phase 6 v2 dry run (Jul 1, 2026) — VERIFIED:**

Dry run on `training_ready_rank_0.jsonl` (254 videos, ~5 min):

| Metric | Result |
|--------|--------|
| avc_lm blocks found | 259,505 |
| SNAC blocks injected | **259,503** (~100%) |
| Agent blocks injected | **12,705** (46% of videos have Phase 5 output) |
| Agent misses | 246,800 (expected — most videos have no Phase 5 output) |
| Output format | `</avc_lm> <agent>...</agent> <snac> <snac_N>... </snac>` ✓ |
| chunk_timing flags | `has_seed2/cosmos/avc_lm/agent/has_snac` all correct ✓ |

**Full run results (Jul 1, 2026) — COMPLETE:** Job `14082096`, 32/32 workers, 0 errors.

| Metric | Result |
|--------|--------|
| Videos | 40,804 |
| Activities | 398,775 |
| avc_lm blocks | 38,825,249 |
| SNAC injected | 38,824,718 (**100.0%**) |
| Agent injected | 2,148,474 (5.5% — expected, only ~18K videos have Phase 5 output) |
| Output | `FineVideo-VLA/final_dataset_adaptive_v2/` — 160 files |

**⚠ Correction (Jul 12, 2026): the "chunk_timing flags all correct ✓" claim above (Jul 1 dry run) was wrong for `has_seed2`/`has_cosmos`.**

`build_chunk_timing()` computed `has_seed2`/`has_cosmos` as `i < len(seed2_matches)` — comparing the chunk loop index against the *total* count of `<seed2>`/`<cosmos>` tags found anywhere in the activity's `video_tokens`, not a real per-chunk positional check. Since seed2 fires at 1fps while avc_lm/cosmos chunks occur at 3.75/sec, `len(seed2_matches)` is always much smaller than the chunk count — so `has_seed2` came out `True` for an artificial prefix of chunks and `False` for the rest of the activity, a single fake ON→OFF transition per activity rather than reflecting real per-chunk presence. Verified empirically: 2,558/2,558 sampled activities showed exactly one ON→OFF flip (never OFF→ON), at wildly inconsistent timestamps (0.27s–638s) depending only on activity length. `has_cosmos` shared the same buggy formula but the bug never manifested, because cosmos fires at the same per-chunk rate as avc_lm, so `len(cosmos_matches) ≈ avc_count` and the comparison was true almost everywhere anyway.

**Fix (Jul 12, 2026):** attribute each `<seed2>`/`<cosmos>` tag to a chunk by its *string position* — a tag belongs to chunk `i` if it falls between the end of chunk `(i-1)`'s `<avc_lm>` block and the end of chunk `i`'s, matching the real temporal write order from `pipeline_video/pipeline.py` (seed2 checked once per frame, before that frame is added to the cosmos/avc_lm buffer). `has_cosmos`/`has_avc_lm` were simplified to hardcoded `True` (verified always correct — 0 flips in 34,732 sampled activities from the fixed re-run, spanning all 40,804 videos).

**Impact assessment — no re-tokenization needed:** `chunk_timing` is metadata only. `phase7_flatten.py` never reads it (confirmed by code search across the repo — the only consumers were `phase6_merge_adaptive.py` itself, `snac_finevideo.py` (which only uses `chunk_idx`/`start_sec`/`end_sec`, never `has_seed2`/`has_cosmos`), and the new captioning prototype scripts written this session). A byte-for-byte diff of `video_tokens` between v2 and the fixed re-run (v3) showed 0 differences on a sample file — the actual token content injected into training data is untouched. **All existing trained models, Megatron `.bin/.idx` files, and `FineVideo-Phase7-Flattened` uploads remain valid** — this bug only matters for new work that reads `chunk_timing` directly, i.e. the captioning pipeline (§2.5c below).

**Re-run (Jul 12, 2026):** SLURM job `14102737` (`slurm/submit_merge_adaptive_v3.sh`), 32/32 tasks COMPLETED, 0 errors → `final_dataset_adaptive_v3/` (160 files; `final_dataset_adaptive_v2/` kept for comparison/rollback). Aggregate stats from worker logs matched v2 exactly: 40,804 videos, 398,775 activities, 2,148,474 agent blocks injected, 38,824,718 SNAC tokens injected — confirming content parity. `final_dataset_adaptive_v3/` is now the standard input for anything that reads `chunk_timing`.

### 2.4 Flatten (Phase 7)
**Script:** `pipeline_pose/phase7_flatten.py`

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

To balance modalities for pretraining, **modality dropout** is applied during flattening.

**v1 dropout** (used for first two training runs — `megatron_dataset_adaptive/`):

| Modality | Drop rate | Effective keep | Resulting tokens |
|----------|-----------|---------------|-----------------|
| AVC-LM | 99% | ~1% of chunks | ~1,250 |
| Cosmos | 90% | ~10% of chunks | ~640 |
| Seed2 | 0% | 100% | ~340 |
| Agent | 0% | 100% | ~300 |

**v2 dropout** (Jun 27, 2026 — `megatron_dataset_v2/`):

| Modality | Drop rate | Reason |
|----------|-----------|--------|
| AVC-LM | **100%** | Removed until ablations confirm benefit (per Huu) |
| Cosmos | **50%** | Keep ~6/12 chunks per activity for seed2→cosmos→agent transition learning |
| Seed2 | 0% | Keep all — primary visual signal |
| Agent | 0% | Keep all |

Output: `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/megatron_dataset_v2/`

**v3 — SNAC support (COMPLETE Jul 2, 2026):**

| Modality | Drop rate | Notes |
|----------|-----------|-------|
| AVC-LM | 100% | Unchanged |
| Cosmos | 50% | Unchanged |
| Seed2 | 0% | Unchanged |
| Agent | 0% | Unchanged |
| **SNAC** | **0%** | New — pass-through, `<snac_N>` tokens extracted from `<snac>...</snac>` blocks |

**Changed record filter:** v1/v2 required `<agent>` in record. v3 emits if `<agent>` OR `<snac>` present:
- **Full-chain:** seed2 + cosmos + agent + snac — 69,811 records (18.8%)
- **Partial-chain:** seed2 + cosmos + snac — 302,044 records (81.2%)
- Pure seed2+cosmos activities still skipped
- **0 bad records** (verified full scan)

**v3 output stats (verified Jul 2, 2026):**

| Metric | Value |
|--------|-------|
| Files | 160/160 |
| Total records | **371,888** |
| Malformed JSON | 0 |
| Full-chain (agent+snac) | 69,811 (18.8%) |
| Snac-only | 302,044 (81.2%) |
| Bad records (no agent, no snac) | **0** |
| seed2 tokens | 332.6M |
| cosmos tokens | 3.88B |
| snac tokens | 363M |
| agent windows | 2,148,474 |
| avclm tokens | 0 ✓ |
| Total size | 72 GB |

Sample: `samples/after_flatten_v3.json` | Output: `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/megatron_dataset_v3/`

#### ⚠ Known design issue in v3: temporal misalignment (FIXED in v4)

`process_tokens_to_individual_tags()` in v3 extracted all `<agent>` and `<snac>` blocks first, then **appended all agent tokens at the end, followed by all snac tokens**. At seq_len=4096 (measured on 2,269 full-chain records): only **31%** had agent tokens in the first 4096 positions — in most training steps the model saw video OR pose, rarely both.

Additionally, `interleave_speech_and_tokens()` scattered speech words into the middle of agent joint sequences, breaking the `<pelvis_x_N>` grammar in ~42.9% of full-chain records.

**Both bugs are fixed in v4** (see below).

**v4 — Per-chunk temporal ordering (COMPLETE Jul 2, 2026):**

Phase 7 fully rewritten with a state machine that walks Phase 6 output in document order. Output per chunk: `[seed2?][cosmos?][agent?][snac?]`. Speech moved to dedicated `### Speech:` header, never mixed into token sequence.

**v4 output stats (verified Jul 2, 2026):**

| Metric | Value |
|--------|-------|
| Files | 160/160 (0 skipped) |
| Total records | **371,888** |
| Runtime | 36 min / 32 workers |

| Modality | Tokens | % | Avg/record |
|----------|--------|---|------------|
| seed2 | 332,592,448 | 6.4% | 894 |
| cosmos | 3,882,981,800 | 74.4% | 10,440 |
| agent | 637,924,374 | 12.2% | 1,715 |
| snac | 363,029,331 | 7.0% | 976 |
| **TOTAL** | **5,216,527,953** | — | **14,027** |

At seq_len=4096, each context window now contains ~8–10 fully aligned `[cosmos?][agent?][snac?]` tuples (~490 tokens/chunk). The model sees video and pose simultaneously in every training step.

Script: `pipeline_pose/phase7_flatten.py` | SLURM: `slurm/submit_phase7_v4.sh`
Output: `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/megatron_dataset_v4/`
Upload script: `tools/upload_flattened_hf.py` | HuggingFace: `EmpathicRobotics/FineVideo-Phase7-Flattened`

#### Token rates per 8-frame chunk (30fps grid)

All modalities are aligned to 30fps. One chunk = 8 frames = 8/30s ≈ 0.267s.

| Modality | Tokens/chunk (raw) | Notes |
|----------|-------------------|-------|
| Seed2 | **32** (fixed) | Only at keyframe chunks (1 per 30 frames = every 3.75 chunks). Most chunks have no seed2. |
| Cosmos | **200** (fixed) | Every chunk. DV8x16x16 spatial encoding. |
| AVC-LM | **885–5,055** (variable) | Every chunk. H.264 BPE, varies with motion. Dropped 100% in v3. |
| Agent | **171–579** (~280 typical) | Only chunks with detected person. Adaptive PCHIP. |
| SNAC | **9 or 12** (alternating, avg 10) | Every chunk. 37.5 tok/s × 0.267s = 10; snaps to 3-token triplets. |

In 30 seconds (≈ 112 chunks at 50% cosmos dropout):

| Modality | Tokens/30s (v3 dropout) |
|----------|------------------------|
| Seed2 | 30 × 32 = **960** |
| Cosmos | ~56 × 200 = **11,200** |
| Agent | up to 112 × 280 = **31,360** (when person present) |
| SNAC | 112 × 10 = **1,120** |
| AVC-LM | **0** (dropped) |

#### Data augmentation

The flatten also applies text augmentation to improve robustness:

| Augmentation | Rate | Description |
|-------------|------|-------------|
| Synonym replacement | 15% | Content words (>5 chars) replaced with WordNet synonyms |
| Stopword dropout | 5% | Common stopwords randomly removed |
| Sentence permutation | 10% | Speech transcript sentences randomly reordered |
| Speech/token interleaving | — | Speech chunks inserted at random positions among tokens |
| Layout block shuffling | — | Title/Context/Keywords/(Speech) blocks randomly reordered |

Each output record (v4) has text headers followed by the token sequence (speech no longer interspersed in tokens):
```
### Title: <scene title, augmented>
### Context: <global context + activity prompt, augmented>
### Keywords: <scene thematic + mood, augmented>
[### Speech: <transcript, augmented>]   ← only if speech present
<flat token sequence in per-chunk temporal order>
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

### 2.5b SNAC Audio Tokenization (snac_finevideo.py) — *In Progress*
**Script:** `pipeline_pose/snac_finevideo.py` | **SLURM:** `slurm/submit_snac_finevideo.sh`

SNAC (Scalable Neural Audio Codec) tokenises the audio track of each FineVideo video using SNAC_24kHz in **listen format** (3 tokens per base frame):

```
SNAC base frame i  →  <snac_{ codes[0][i] + 128266 }>      (Level 0, 12.5 Hz)
                       <snac_{ codes[1][2i] + 132362 }>     (Level 1 even, 25 Hz)
                       <snac_{ codes[1][2i+1] + 144650 }>   (Level 1 odd, 25 Hz)
```

Listen format ignores Level 2 (50 Hz fine detail), giving **37.5 tokens/sec** vs 87.5 for the full "speak" format. Compatible with MixtureVitae-Omni offsets.

**Chunk alignment:** SNAC rate (37.5 tok/s) and video chunk rate (3.75 chunks/s) have irrational ratio. Solution:
1. Encode full activity audio in one call (preserves temporal audio context across chunk boundaries)
2. Divide flat token list evenly across chunks, snapping to 3-token boundaries
3. Each 8-frame chunk receives ~9–10 SNAC tokens (3.33 base frames × 3)

Output: `{OUTPUT_DIR}/{video_id}_snac.jsonl` — one line per activity:
```json
{
  "video_id": "abc123",
  "activity_id": "scene_1_act_2",
  "has_agent": true,
  "snac_by_chunk": {
    "0": ["<snac_130055>", "<snac_133001>", "<snac_144980>", ...],
    "1": ["<snac_129900>", "<snac_132800>", "<snac_145200>", ...],
    ...
  }
}
```

**Coverage:** ALL activities, not just agent ones. 86% of activities have no agent tokens but still have valid seed2+cosmos — adding SNAC to these creates seed2+cosmos+snac training records that teach audio↔video binding.

**Vocab cost:** 3 × 4096 = 12,288 new `<snac_N>` token strings (N ∈ {[128266,132361], [132362,136457], [144650,148745]}). Added to tokenizer via `add_tokens(special_tokens=True)` → new vocab **156,505** (see Section 2.5b Tokenizer v2).

**Environment setup (done Jun 28, 2026):**
- `snac 1.2.1` installed into both `env_tools` (x86, login node) and `my_env_clean` (ppc64le, booster)
- SNAC model weights pre-downloaded: `/p/scratch/laionize/nguyen38/hf_cache/hub/models--hubertsiuzdak--snac_24khz`
- `HF_HUB_OFFLINE=1` set in SLURM script — compute nodes have no internet
- Task list built: `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/snac_task_list.json` (40,798 videos, 372,385 activities)

**SLURM cluster note (discovered Jun 28, 2026):**
- `jwlogin08.juwels` (JUWELS Cluster) cannot submit to `booster` partition with `laionize` account
- Must SSH to `juwels-booster.fz-juelich.de` (separate Slurm cluster) to access GPU nodes
- CPU fallback available: `bash slurm/submit_snac_finevideo.sh --cpu` (uses `batch` partition, x86, ~24h)

**Run procedure (GPU, from juwels-booster.fz-juelich.de):**
```bash
# Task list already built — skip step 1
# Step 2: submit array job (16 GPU workers, ~8-12 hours)
cd /p/data1/mmlaion/nguyen38/3d-human-pose
bash slurm/submit_snac_finevideo.sh
```

**STATUS: COMPLETE (Jul 1, 2026)** — Job `snac_cpu_14077331`, 32/32 tasks done. 371,855 activities, 363M tokens, 6.5 GB → `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/snac_tokens/`

### 2.5b Tokenizer v2 — SNAC support (Jul 1, 2026)
**Script:** `tools/build_tokenizers.py`

Built two tokenizers with SNAC support. All VLA tokens verified atomic (single token ID per string):

| Tokenizer | Base | + VLA tokens | Final vocab | Path |
|-----------|------|--------------|-------------|------|
| `tokenizer_vla_adaptive_v2` | GPT-NeoX-20b (50,277) | 93,938 existing + 12,290 SNAC | **156,505** | `/p/data1/mmlaion/shared/vla/tokenizer_vla_adaptive_v2/` |
| `tokenizer_vla_qwen3` | Qwen3 (~151,669) | 106,228 VLA (all types) | **257,897** | `/p/data1/mmlaion/shared/vla/tokenizer_vla_qwen3/` |

SNAC token ranges (12,288 tokens total, plus `<snac>` and `</snac>` wrappers = 12,290):
- Level 0: `<snac_128266>` … `<snac_132361>` (4096 tokens)
- Level 1 even: `<snac_132362>` … `<snac_136457>` (4096 tokens)
- Level 1 odd: `<snac_144650>` … `<snac_148745>` (4096 tokens)

Usage:
```bash
source /p/data1/mmlaion/nguyen38/3d-human-pose/activate_env_tools.sh
python tools/build_tokenizers.py --mode current   # GPT-NeoX v2 only
python tools/build_tokenizers.py --mode qwen3     # Qwen3 only
python tools/build_tokenizers.py --mode all       # both
```

### 2.5c Captioning Pipeline (design finalized Jul 12, 2026 — not yet coded at full scale)

**Goal:** add a natural-language caption at key points in the token sequence, so the model has a language anchor for when/why the modality mix is about to change — root cause #2 for the model's inability to self-transition between modalities during inference.

**Prototype scripts:** `tools/analysis/caption_prototype.py` (core building blocks: `extract_frame`, `select_anchor_points`, model loaders/callers for Qwen2.5-VL / Florence-2 / SmolVLM2), plus one-off batch/visual-QA scripts in the same directory (`caption_prototype_batch.py`, `caption_prototype_visual.py`, `caption_prototype_visual_batch.py`, `caption_florence2_visual_batch.py`, `caption_model_compare.py`, `caption_final_compare.py`).

**Anchor point selection — `select_anchor_points(chunk_timing, min_gap_sec=5.0)`:**

The original plan was to caption at every point where any of the 5 `chunk_timing` flags (`has_seed2/cosmos/avc_lm/agent/snac`) changes. Measured on real data this doesn't work: `has_cosmos`/`has_avc_lm` never vary within an activity (they're encoded at the same fixed 8-frame cadence as the chunk grid itself), and `has_seed2` — even after the §2.3 bugfix — still flips ~54x/activity purely because seed2 fires at a fixed 1fps rate; that's a technical cadence, not a content change. The only flag that reflects a genuine visual event is `has_agent` (a person detected/not detected by YOLO in Phase 4). Final design: caption the activity's first chunk (opening context) plus every chunk where `has_agent` flips, with a `min_gap_sec=5.0` debounce — because `has_agent` itself flickers frame-to-frame in busy/high-motion scenes (sports, martial arts) due to noisy YOLO detection (a known pre-existing data-quality issue, not a bug requiring a Phase 6 fix). The debounce only affects which points get captioned; it doesn't touch stored `chunk_timing` data.

**Known limitation (as originally measured, since addressed — see §2.5d below):** this design gave ~1.86 captions/activity on average (measured at a 2s debounce gap; slightly lower at 5s) — far short of the "×4 records" impact originally targeted. 82.8% of activities got exactly 1 caption (the opening frame only, no agent event ever occurs in that activity).

**Model selection — Qwen2.5-VL-3B-Instruct chosen after testing 3 candidates:**

| Model | Result |
|---|---|
| **Qwen2.5-VL-3B-Instruct** (chosen) | No hallucinations across all tests (including a 96-caption batch across 10 videos). Natively supported in `transformers` — no compatibility risk. Prompt: `"Describe what the person is doing in one short sentence."` CPU speed: ~11-14s/caption. |
| Florence-2-base | `<DETAILED_CAPTION>` task mode hallucinates (e.g. "he appears to be a psycholinguist" for a bearded man with glasses — reproducible, not sampling noise, since generation used deterministic beam search with no `temperature`/`do_sample`). Switching to `<CAPTION>` task mode eliminated the hallucination, cut generation time to ~1.5-3s/caption (3.5x faster than Qwen), and fixed truncation (raised `max_new_tokens` 48→64). Requires a separate venv (`env_caption_test/`, `transformers==4.49.0`, torchvision reinstalled from the CPU wheel index) because its `trust_remote_code=True` custom modeling code breaks under newer `transformers` (`AttributeError: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'`). |
| SmolVLM2-2.2B-Instruct | 2x *slower* than Qwen2.5-VL on CPU in this environment (27.7s vs 14.0s/caption average) — contradicts its "fast, edge-oriented" reputation, likely due to an unoptimized CPU code path in this transformers version. Also hallucinated once (invented "holding a book and reading it" for a plain white intro-slate frame with no book). Rejected. |

Rationale for choosing Qwen2.5-VL-3B despite losing the CPU speed benchmark to Florence-2: quality/no-hallucination and long-term library-compatibility risk were weighted higher than raw CPU speed. (Note: an earlier draft of this section assumed the full run "must happen on GPU regardless of model choice" — superseded by Van Khue's Jul 12 decision to run A2 on CPU, and by the real cost measurement in §2.5d below.)

**Full production pipeline:**

```
final_dataset_adaptive_v3/ (chunk_timing-fixed, see §2.3)
    → [A1] Task list generation (CPU): scan chunk_timing for every activity,
            compute anchor points via select_anchor_points(), write a
            {video_id: [{activity_id, chunk_idx, start_sec, has_agent}, ...]}
            task list (same pattern as snac_task_list.json)
            STATUS: DONE, see §2.5d.
    → [A2] SLURM array job: each worker loads Qwen2.5-VL-3B once, opens
            videos_staging/{video_id}.mp4, extracts a frame per anchor point,
            captions it → outputs/captions/{video_id}_captions.jsonl
            STATUS: coded + smoke-tested, full run not started, see §2.5d.
    → [B1] Extend phase6_merge_adaptive.py with a --captions-dir flag
            (same pattern as --snac-tokens-dir), injecting
            <caption>...</caption> immediately BEFORE the <cosmos> block of
            the anchor chunk (chunk-boundary insertion only — never mid-block,
            avoiding a repeat of the v3→v4 speech-interleaving bug in §2.4)
            → final_dataset_adaptive_v4/
            STATUS: not started.
    → [B2] phase7_flatten.py (unchanged) → megatron_dataset_v5/ → tokenize → train
            STATUS: not started.
```

### 2.5d A1/A2 implementation, validation, and cost measurement (Jul 12, 2026, later same day)

**`select_anchor_points()` extended with a periodic-supplement step** to fix the §2.5c density limitation: after the agent-transition step, if fewer than `target_count=4` points were found, evenly-spaced supplemental points are added across the activity duration, snapped to the nearest real chunk, debounced (`min_gap_sec`) against already-kept points. New signature: `select_anchor_points(chunk_timing, min_gap_sec=2.0, target_count=4)` in `tools/analysis/caption_prototype.py`.

**Bug found and fixed:** the supplement's `duration` was computed from the chunk's **absolute** `end_sec` instead of relative to the activity's own start_sec — activities starting late in a long video got target timestamps computed far outside their actual span, making the supplement silently a no-op for them. Fixed by subtracting `activity_start` first. Verified on 2,563 real activities: % reaching `target_count=4` rose from 10.4% → 54.8%.

**A1 (`tools/analysis/generate_caption_tasks.py`) run on the full dataset** (160/160 shards; 13 via a since-cancelled SLURM array job `14103227`, 147 directly on the login node with `--skip-existing`). Result: 40,798 videos, 372,385 activities, **912,998 task points**, avg **2.45 captions/activity**. Validated end-to-end: 100% schema/type/video_path-exists checks pass; 0 duplicate `chunk_idx` per activity; 0 debounce violations; a 5-shard cross-check (11,576 activities, 28,156 points) recomputing from source `chunk_timing` matched the saved output 100%, with 0 missing/orphan activities.

**Re-scoped the "×4" target:** re-reading §13 (below) clarifies the original ×4 figure was the combined effect of captioning *and* perspective framing multiplying total training record count — not "4 captions per activity." Measured avg 2.45/activity is 61.3% of the narrow (captions-per-activity) reading; root cause is structural, not a bug: ~59% of activities are under 15s, which cannot geometrically fit 4 points ≥5s apart. Decision: keep `target_count=4, min_gap_sec=5.0` — lowering the gap to force the number up would add near-duplicate captions on short static clips (compute cost with no real language-signal gain), not the correct lever for the ×4 goal.

**A2 (`pipeline_pose/caption_finevideo.py`) coded**, following the same worker-split/resume pattern as `snac_finevideo.py` (model loaded once per worker; videos striped `all_vids[task_id::num_tasks]`; one output file per video for safe resume). Smoke test (video `A1UVeD9UB1I`, t=248.0s) produced a sensible caption ("The person is arranging jewelry on a box.") matching the source `text_prompt` ("Woman opens a gift box.").

**Infra bug found and fixed:** no PyTorch thread limit was set, so two concurrent local test runs on an 80-core shared login node oversubscribed each other, producing 57.6s/caption (~4x the true rate). Added `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`torch.set_num_threads()` pinned to `SLURM_CPUS_PER_TASK` (default 4) to prevent the eventual 32 SLURM workers from oversubscribing each other the same way.

**Clean CPU throughput measured (4 threads, no contention, 3 repeats): ~13.8s/caption** (12.9/15.2/13.4s) — consistent with the §2.5c CPU estimate. **Full-run cost: 912,998 tasks × 13.8s ≈ 3,500 CPU-hours.** With 32 workers (matching the SNAC job's scale) → ~109h/worker (~4.6 days), needing ~5 resubmits at `--time=24:00:00`; safe to resubmit via per-video skip-existing. Submit script `slurm/submit_caption_finevideo.sh` written but **not yet submitted** — open decision for next session is CPU (ready now, ~4.6 days) vs GPU (2×4090 machine, unmeasured; a decisive GPU win likely requires implementing batched inference in `caption_frame()`, which does not exist yet — currently one image per forward pass).

Captions are plain English sentences — regular BPE tokenization, no vocab expansion required.

**Infra note (Jul 12, 2026):** the real captioning run (step A2) will use CPU (many cores) rather than the available 2×GPU (RTX 4090) test machine — Van Khue's call, no GPU access needed yet for this step.

**Side findings from this session:**
- FineVideo source videos are already staged locally at `/p/data1/mmlaion/shared/nguyen38/data/videos_staging/` (43,751 mp4s, named `{video_id}.mp4` — note the directory name has an "s"; a similarly-named but empty `video_staging/` also exists, don't confuse them). No JUPITER dependency or HF streaming needed for frame extraction.
- Read and evaluated `HumanoidBench` (arXiv 2403.10506) as a candidate eval benchmark — **not a fit** for the current model. It's a closed-loop RL/control benchmark (MuJoCo simulation, Unitree H1 + two Shadow Hands, 61-dim joint-position action space at 50Hz) whereas this project's agent tokens are xyz world positions for 17 H36M human joints with no joint angles or hand/finger data. Only relevant to the already-deferred Priority 12 (Isaac Sim / H1 sim-to-real), not the near-term eval-protocol discussion (DISCUSS-3).
- Home directory quota is much smaller than `/p/data1` project storage — set `HF_HOME=/p/data1/mmlaion/nguyen38/hf_cache` before downloading large HF models to avoid `OSError: [Errno 122] Disk quota exceeded`.
- `huggingface_hub`'s Xet download backend can fail transiently (`RuntimeError: ... Background writer channel closed`) — set `HF_HUB_DISABLE_XET=1` to fall back to plain HTTP downloads.

### 2.5e A2 full run: CPU decided, submitted, and confirmed working (Jul 13, 2026)

**Decision:** CPU chosen (Van Khue) over the unmeasured GPU/batching path — `slurm/submit_caption_finevideo.sh` was ready to go, so no reason to wait on the GPU-batching investigation described in §2.5d's open question.

**First submit (job `14104070`) failed 32/32 tasks at model load.** `slurm/submit_caption_finevideo.sh` pointed `HF_CACHE` at `/p/scratch/laionize/nguyen38/hf_cache`, which only contains `bert-base-uncased` and `snac_24khz` — not the Qwen2.5-VL-3B-Instruct weights used by A2. Since compute nodes run with `HF_HUB_OFFLINE=1` (no internet), `Qwen2_5_VLForConditionalGeneration.from_pretrained()` raised `OSError: We couldn't connect to 'https://huggingface.co' ... Qwen/Qwen2.5-VL-3B-Instruct is not the path to a directory containing a file named config.json` in every one of the 32 array tasks. The correct cache — the one actually used for the §2.5d smoke test — is `/p/data1/mmlaion/nguyen38/hf_cache` (confirmed present: `models--Qwen--Qwen2.5-VL-3B-Instruct`, 7.1GB), consistent with the general env note already in §2.5d ("set `HF_HOME=/p/data1/mmlaion/nguyen38/hf_cache`... to avoid Disk quota exceeded" — the same path serves double duty as the correct model cache location).

**Fix + resubmit:** changed `HF_CACHE` in `slurm/submit_caption_finevideo.sh` to `/p/data1/mmlaion/nguyen38/hf_cache`, resubmitted as job `14104104`. Confirmed working: all 32/32 array tasks reached running state, each worker's model load completed cleanly in ~44-45s with no offline errors, and per-video output files began appearing within ~5 minutes. Spot-checked one output file (`-0-6Som0MGY_captions.jsonl`, 10 lines) — well-formed JSON matching the documented schema, captions qualitatively specific and accurate to the source video content (e.g. *"The person is pouring sulfuric acid into an energy drink can."*, *"The person is using a blue dropper to apply coconut oil onto a surface."*).

**End-of-session status:** job `14104104` running with all 32 workers active, ETA ~4.6 days per the §2.5d cost estimate (912,998 tasks × 13.8s / 32 workers), will need ~5 resubmits at the `--time=24:00:00` limit — safe via per-video skip-existing. Next actions: monitor/resubmit job `14104104` until `outputs/captions/` covers all 40,798 videos, then start B1 (`--captions-dir` flag on `phase6_merge_adaptive.py`) — B1 does not strictly require 100% A2 completion, only enough coverage to prototype against.

**Auto-chaining set up so no manual resubmission is needed:** `slurm/submit_caption_finevideo.sh` extended to accept an optional `$1` job id and pass `--dependency=afterany:$1` to `sbatch` (prints the new job id on stdout for easy chaining). Submitted a 5-job chain after `14104104`: `14104104 → 14104155 → 14104156 → 14104157 → 14104158 → 14104159`, each starting only once the previous one's full array exits (success, failure, or timeout) — 6 jobs × 24h = 144h (~6 days) of coverage, comfortably above the ~4.6-day estimate. Confirmed via `squeue --start`: all 5 queued jobs show `(Dependency)` as their wait reason.

**Spot-check of live output (333+ files, ~340 sampled captions) — quality holds up at scale, one hallucination class found:** captions are specific and match source `text_prompt` content well (e.g. a Freemasons' Hall museum video correctly captioned *"standing in front of an open door with a sign that says 'We're Open Free Admission Library Museum Shop Tours'"*, matching the source activity "Entering Freemasons' Hall"). **One clear hallucination found:** video `-Gq3DJyhJ3I` (a soccer/Frankie de Jong video) got the caption *"The person is performing a complex mathematical operation involving fractions, exponents, and square roots"* at t=0.0s — completely unrelated to the actual source content ("Soccer players playing a match, a goal is scored"). Extracted the real frame at t=0.0s with `cv2` to check: it's a **near-black fade-in frame** (very low mean pixel intensity), and the model hallucinated content instead of saying "not visible" (which it does correctly elsewhere, e.g. *"The person is not visible in the image"* on a similar dark/empty frame in another video). **Assessed as a known, low-severity Qwen2.5-VL limitation** (consistent with the ~1-in-30-96 hallucination rate measured during model selection in §2.5c), not a pipeline bug — does not block the full run. **Possible future mitigation for B1** (not implemented, low priority): compute mean pixel intensity of each extracted frame in `caption_finevideo.py` and skip/flag captioning on near-black frames before they're injected into training data.

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
| [EmpathicRobotics/FineVideo-Phase7-Flattened](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase7-Flattened) | **v4 (live Jul 7, 2026)** — flat Megatron-LM JSONL, per-chunk temporal ordering, agent OR snac required, modality dropout + augmentation | 371,888 | 5.217B tokens | `{"text": "### Title: ... ### Speech: ... <seed2_N> <cosmos_N> <fps_30> <pelvis> ... <snac_N> ..."}` |
| [EmpathicRobotics/tokenizer-vla-adaptive](https://huggingface.co/EmpathicRobotics/tokenizer-vla-adaptive) | GPT-NeoX-20b + 93,938 VLA tokens (v1, no SNAC) | — | 144,215 vocab | HF tokenizer dir |
| [EmpathicRobotics/tokenizer-vla-adaptive-v2](https://huggingface.co/EmpathicRobotics/tokenizer-vla-adaptive-v2) | GPT-NeoX-20b + VLA + 12,290 SNAC tokens (v2) | — | **156,505 vocab** | HF tokenizer dir |
| [EmpathicRobotics/tokenizer-vla-qwen3](https://huggingface.co/EmpathicRobotics/tokenizer-vla-qwen3) | Qwen3 base + all 106,228 VLA tokens incl. SNAC | — | **257,897 vocab** | HF tokenizer dir |

### Published models

| Model | What | Params | Tokenizer |
|-------|------|--------|-----------|
| [EmpathicRobotics/vla-1.7b-pab-spline-25b-test](https://huggingface.co/EmpathicRobotics/vla-1.7b-pab-spline-25b-test) | First VLA model (broken tokenizer, fixed 256-token agent format) | 1.7B | Broken (sub-piece splitting) |
| [EmpathicRobotics/vla-1.7b-pab-spline-adaptive](https://huggingface.co/EmpathicRobotics/vla-1.7b-pab-spline-adaptive) | Second VLA model (fixed tokenizer, adaptive PCHIP agent tokens) | 1.91B | [tokenizer-vla-adaptive](https://huggingface.co/EmpathicRobotics/tokenizer-vla-adaptive) (144,215 vocab, all atomic) |

### What's in each dataset

**FineVideo-Phase7-Flattened** — Use this for LLM pretraining. **As of v4 (Jul 7, 2026)**, each record is a single activity with all modalities flattened in per-chunk temporal order (`[seed2?][cosmos?][agent?][snac?]` per 8-frame chunk), speech moved to a dedicated `### Speech:` header, modality dropout (100% AVC-LM, 50% Cosmos) and text augmentation applied. Records require `<agent>` OR `<snac>` (not agent-only like earlier versions) — 18.8% are full-chain (seed2+cosmos+agent+snac), 81.2% are partial-chain (seed2+cosmos+snac only).

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
| Step A: Video tokens | ~40,000 | 160 files, ~660 GB | `pipeline_video/pipeline.py` |
| Phase 1: 2D pose (HRNet) | 40,804 | 145 GB | `pipeline_pose/phase1_hrnet_gpu.py` |
| Phase 2: 3D pose (MotionBERT) | 40,804 | 259 GB | `pipeline_pose/phase2_motionbert_gpu.py` |
| Phase 2.5: 30fps resample | 40,804 | 67 GB | `pipeline_pose/phase2_5_resample_30fps.py` |
| Phase 3: Kinematics | 40,200 | 193 GB | `pipeline_pose/phase3_kinematics_processor.py` |
| Phase 4: YOLO cleaning | 40,195 | 107 GB | `pipeline_pose/phase4_yolo_cleaner.py` |
| Phase 5: Adaptive PCHIP | 18,847 | 7.4 GB | `pipeline_pose/phase5_adaptive_pchip.py` |
| Phase 6: Merge | 160 files | 657 GB | `pipeline_pose/phase6_merge_adaptive.py` |
| Phase 7: Flatten (dropout + augment) | 160 files, 69,844 records | 19.2 GB | `pipeline_pose/phase7_flatten.py` |
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
- **Pose data quality — arm joints absent (Jul 2, 2026):** Only 4–7 of 17 joints are finite per frame; arm joints (j11–j16) are NaN in nearly all windows. The model learns lower-body and torso motion but has no arm/hand signal. head_top (j10) has a zero-fill artifact (stored as (0,0,0) when undetected, same as pelvis). This is a limitation of monocular MotionBERT lifting on unconstrained YouTube video.

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

**Goal**: Count tokens across ALL available multimodal datasets. Create a chart and table showing token counts by modality, number of records, and size in GB.

**Status**: **COMPLETE** (June 26, 2026). Script: `tools/data_inventory.py`. Final checkpoint: `tools/inventory_checkpoint_v2.json`. Chart: `tools/data_inventory_charts.png`.

**Datasets surveyed**:

| Dataset | Source | Content | Status |
|---------|--------|---------|--------|
| FineVideo-Phase7-Flattened | local `/p/data1/mmlaion/shared/vla/vla_adaptive/` | Tokenized JSONL, 160 files | **Done** |
| MixtureVitae-Backup (valid_with_seed) | HF `mixture-vitae-backup/MixtureVitae-Backup` | 64 HF shards (~1.1 TB total downloaded to `/p/data1/mmlaion/nguyen38/inventory_cache/hf_shards/`) | **Done** |
| MixtureVitae-Backup (stack_images3_gzip) | local `/p/data1/mmlaion/nguyen38/inventory_cache/stack_images3/` | 12 tar.gz archives | **Done** |
| MixtureVitae-Omni (valid_snac) | HF `mixture-vitae/MixtureVitae-Omni` | 6 gzip JSONL files, cached at `/p/data1/mmlaion/nguyen38/inventory_cache/hf_snac/` | **Done** |
| SenseNova-SI-8M | `sensenova/SenseNova-SI-8M` | 8M image-text pairs (raw) | Needs tokenization |
| stera-10m | `fpvlabs/stera-10m` | 10M video clips | Restrictive license |
| OmniAction | `OpenMOSS-Team/OmniAction` | Action-labeled video | CC-BY-NC-4.0 |

**Final results** (all 242 files scanned; `seed` and `seed2` merged as seed2):

| Dataset | seed2 | cosmos | avclm | agent | snac | text | **TOTAL** |
|---------|-------|--------|-------|-------|------|------|-----------|
| FineVideo-VLA (160 files, 69,844 records) | 89.9M | 210.2M | 474.4M | 564.9M | — | 11.4M | **1.35B** |
| MV-Backup valid_with_seed (64 HF shards) | 5.6M | — | — | — | — | — | **5.6M** |
| MV-Backup stack_images3_gzip (12 archives) | 313K | — | — | — | — | — | **313K** |
| MV-Omni valid_snac (6 gzip files) | 19.2M | — | — | — | 4.92B | 1.99B | **6.93B** |
| **TOTAL** | **115M** | **210.2M** | **474.4M** | **564.9M** | **4.92B** | **2.00B** | **8.29B** |

**Training-ready today** (tokens already in vocab): **1.35B** (FineVideo only). MV-Omni's 6.93B requires vocab expansion for `<snac_N>` and `<seed_N>`.

**Key findings**:

- **valid_with_seed yields only 5.6M seed2 tokens across all 64 shards (~1.1 TB downloaded).** Each outer shard contains ~9 inner tar.gz archives. Shards 00000–00030 (31 shards) have inner archives with only `.png`/`.ogg` files — zero tokenized content. Shards 00031–00063 (33 shards) do contain `_seed2.jsonl` files inside their inner archives, averaging ~170K seed2 tokens each. The total 5.6M is negligible compared to FineVideo's 89.9M seed2 and not worth the 1.1 TB storage cost for training.
- **stack_images3_gzip yields only 313K seed2 tokens** (12 StackExchange archives). Also negligible.
- **MV-Omni is the only substantial external source** at 6.93B tokens (4.92B SNAC + 1.99B text + 19.2M seed). However, `<snac_N>` and `<seed_N>` tokens are not in the current tokenizer vocab — both need to be added via `tokenizer.add_tokens()` before MV-Omni can be used in training.
- **Only FineVideo has agent tokens.** No external dataset adds 3D human pose data.
- **The captioning pipeline (Phase 2) is the highest-impact path forward** — it multiplies FineVideo's value 4× without requiring new data or vocab changes.

**Impact on improvement plan**: The data landscape is now clear. External seed2 sources (valid_with_seed, stack) are negligible. The two actionable paths are: (a) vocab expansion + MV-Omni integration for +6.93B tokens, and (b) rich captioning pipeline on FineVideo for ~4–5B tokens from existing data. Both are needed to reach the ~20B token target for a well-trained 1.7B model.

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

**Known tokenized sources** (inventory complete as of June 26, 2026):

| Source | Tokens | In vocab? | Notes |
|--------|--------|-----------|-------|
| FineVideo-VLA | **1.35B** | ✅ Yes | All 4 modalities, 100% agent coverage. Training-ready. |
| MV-Omni valid_snac | **6.93B** | ❌ No | `<seed_N>` (19.2M) + SNAC (4.92B) + text (1.99B). Needs vocab expansion. |
| MV-Backup valid_with_seed (64 HF shards, 1.1 TB) | **5.6M** | ✅ seed2 | Negligible — not worth the storage/compute cost to mix in. |
| MV-Backup stack_images3_gzip (12 archives) | **313K** | ✅ seed2 | Negligible. |
| SenseNova-SI-8M | 0 | — | Raw image-text pairs, needs Seed2 tokenization. |

**Dataset overlap analysis (Jun 30, 2026 — COMPLETE):**

`tools/check_dataset_overlap.py` compared `valid_with_seed` (64 HF shards) vs `omni_valid` (6 gzip files) by YouTube video ID:

| Metric | Count |
|--------|-------|
| `valid_with_seed` unique video IDs | 31,500 |
| `omni_valid` unique video IDs | 238,539 |
| Overlap | **27,359** (86.9% of seed / 11.5% of omni) |
| Only in `valid_with_seed` | 4,141 |
| Only in `omni_valid` | 211,180 |

**Key finding:** `omni_valid` already covers 86.9% of `valid_with_seed`'s videos. The remaining 4,141 unique-to-seed videos only have seed2 tokens (~700K tokens total) — not worth the 1.1 TB of storage. **Decision: do not use `valid_with_seed`.** The 1.1 TB already downloaded can be freed.

**Short-term**: Add `<snac_N>` tokens to the tokenizer via `tokenizer.add_tokens()` to unlock MV-Omni's 6.93B tokens. valid_with_seed is confirmed not worth mixing in (covered by omni_valid and negligible unique content).

**Medium-term** (requires GPU runs): Run Seed2 tokenization on SenseNova-SI-8M images to add another large seed2 source.

**Vocab impact**: Adding SNAC (~4096 tokens) and `<seed_N>` (~8192 tokens) expands vocab by ~12K tokens. Existing embeddings are preserved; new embeddings initialize randomly and are fine-tuned during continued training.

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

| Priority | Task | During downtime? | Impact on model | Status |
|----------|------|------------------|-----------------|--------|
| 1 | Data inventory + chart | Yes | Guides all other decisions | ✅ DONE |
| 2 | Write captioning pipeline code | Yes (code only) | Prep for highest-impact improvement | Pending |
| 3 | Vocab expansion for `<snac_N>` + `<seed_N>` | Yes (code) | Unlocks MV-Omni's 6.93B tokens | Pending |
| 4 | Adjust dropout + re-flatten | Partially (code + test) | Cheapest fix for modality transitions | Pending |
| 5 | Re-training v0.2 | No (needs JUPITER) | First real model improvement | Pending |
| 6 | Full captioning run on GPUs | No (needs JUPITER) | Major data quality boost | Pending |

---

## 14. Repository

**GitHub:** [TieuDaoChanNhan/finevideo-vla](https://github.com/TieuDaoChanNhan/finevideo-vla)

All pipeline scripts, SLURM jobs, upload tools, vocab, and documentation are in this repo. See `README.md` for setup instructions and detailed usage.

---

## 15. Status Update — July 8, 2026

### Infrastructure

JSC cluster outage since ~Jul 6, 2026: JUPITER fully down, JUWELS booster + JURECA have partial GPU availability. ETA per Huu: officially 1 week, realistically ~2 weeks. Blocks large-scale Megatron re-tokenization, training v0.3, full-dataset 1-CP validation, and the Cosmos3-DROID GPU pipeline.

Cluster account mapping (confirmed Jul 7, 2026):
```
JUSUF:   ccstdl
JUPITER: reformo
JUWELS:  laionize
```

### Team direction: multi-project data sharing

Huu is pooling data across three parallel efforts: this repo's omni-VLA work, joergfranke's architecture comparison project (baselines: qwen3, lfm2.5, olmo3, nemotron on ~2T-token comparison runs once JUPITER is back), and blanchon.jl's diffusion-based world-action-model (video generation + action, targeting a "fast WAM"/"Cosmos Policy"-style architecture). `FineVideo-Phase7-Flattened` is now used as shared input across projects. Longer-term idea discussed: bridge the discrete-token LLM (this project) and the diffusion world-action-model via a llava-like cross-attention connector. Team also agreed synthetic/simulation data should be capped at **≤30% of total training mix**, citing literature guidance.

### New data source candidates (from Jul 7, 2026 team discussion, not yet scoped)

| Source | What | Notes |
|---|---|---|
| `abc.bot` (Amazon) | 400h robot recordings in simulation, includes physics state (MjData) | Most promising — permissive, has an eval environment, consistent embodiment |
| `allenai/MolmoAct2-BimanualYAM-Dataset` | 2 TB, bimanual YAM arm robot data | Check license + embodiment compatibility |
| `MiG-NJU/OmniVideo-100K` | Video dataset | Not yet scoped |
| `mlfoundations/MINT-1T-HTML` | Large text/HTML dataset | Not yet scoped — likely relevant to the language/instruction mix, not video |
| `genrobot2025/Gen-EgoData` | Egocentric robot data | Not yet scoped |
| `finevla.xlang.ai` | Possible VLA dataset | HF link not found — may be unreleased |

### 1-CP: final decision — deferred

See Section 2.2 (Phase 5) for the full analysis. Bottom line: the targeted, static-only 1-CP variant is technically safe (no added reconstruction error) and gives +7.1% additional compression on top of the existing 50.9% savings, but the team (confirmed with Huu on Discord, Jul 8) decided to **defer** implementation — the gain doesn't justify a full Phase 5→6→7 re-run right now, especially with the cluster down. Revisit only if later evidence shows it's necessary.

### Pending investigation tasks (assigned, not yet done)

1. ✅ ~~`mixture-vitae-backup/MixtureVitae-Backup` — `multimodal` branch on HF (Huu asked Jul 5).~~ **Investigated Jul 9, 2026** — see Section 16 below. Awaiting Huu's go/no-go before any integration.
2. "finevideo reformulation" at `leo:/mnt/sdb/mixture-vitae-working/finevideo` — unclear scope, check for overlap with this project's own pipeline before using.
3. MV-Omni mix ratio — naively combining all of MV-Omni (6.93B tokens, 0 agent tokens) with FineVideo v4 would dilute the agent token share from 12.2% to ~5.2% of the combined corpus. Needs a dropout or oversampling strategy before mixing, since agent (3D pose) tokens are the project's core differentiator.

### Current priority ranking (data-first, cluster-down-aware)

| Tier | Task | Needs cluster? | Impact |
|---|---|---|---|
| P0 | Investigate MixtureVitae-Backup/multimodal, clarify finevideo reformulation, decide MV-Omni mix ratio, define eval protocol, decide language-data mix ratio | No | Unblocks planning decisions |
| P1 | Write captioning pipeline code; write ego-centric perspective converter; mix MV-Omni into Megatron format | No / CPU only | Highest — captioning fixes root cause 2 (modality transitions); MV-Omni is +6.93B tokens at near-zero cost |
| P2 | Scope abc.bot, MolmoAct2-BimanualYAM, OmniVideo-100K, MINT-1T-HTML, Gen-EgoData; investigate leo seed2 + euro_pat | No | New sources, size TBD |
| P3 | Cosmos3-DROID pipeline run; full captioning run; Megatron re-tokenize combined corpus; train v0.3 | GPU (JUPITER) | Blocked until cluster back + data ready |
| P4 (deferred) | 1-CP, Moss-Audio V2, Qwen3 migration, PAB-Spline angle spec, Isaac Sim | — | Explicitly held off per team decisions |

---

## 16. MixtureVitae-Backup Multimodal Investigation (Jul 9, 2026)

### Background

Investigated the P0 item from Huu (asked Jul 5): `mixture-vitae-backup/MixtureVitae-Backup/data/multimodal` on HF — a folder never scanned before, distinct from the `valid_with_seed`/`stack_images3_gzip` sections already covered by `tools/data_inventory.py`. Run locally on a Windows dev machine without JUWELS access this session — CPU only, so the approach was sample-based streaming rather than a full 103GB download.

### Method

Two new scripts, reusing `PATTERNS`/`count_tokens`/`_hf_token`/`hf_url`/checkpoint machinery from `tools/data_inventory.py` (imported, not duplicated):

- **`tools/peek_multimodal.py`** — structural probe. Streams just the first few records/members per file (no full download, no local temp file) to discover format and flag VLA-tag-token presence before writing full parsing logic.
- **`tools/count_multimodal_tokens.py`** — true HTTP streaming (never writes the compressed file to disk). Caps each file at `--sample-mb` compressed MB (default 75). Counts VLA-tag tokens via the same regex as `data_inventory.py`, plus any raw integer token arrays (any `*_token`/`*_tokens` field holding a list of ints — generalized, not hardcoded to `snac_token`). Extrapolates sampled counts to full file size. Resumable via an atomic JSON checkpoint (`tools/multimodal_inventory_checkpoint.json`).

**Implementation bug found and fixed:** `valid_data_snac.jsonl.gz`, `train_data_snac.jsonl.gz`, and `emo.jsonl.gz` are not true JSONL (one compact object per line) — they're a pretty-printed JSON array where a single record can span many physical lines. Naive newline-splitting silently produced zero parsed records (every line failed `json.loads`, caught by a broad except and skipped). Fixed by switching to a streaming text buffer combined with `json.JSONDecoder().raw_decode()`, which pulls complete top-level JSON values from the buffer regardless of embedded newlines.

Local environment: plain Python venv (`tools/env_multimodal_inventory/`, gitignored), `pip install requests tqdm` — no conda, no torch/datasets/pandas needed, matching the existing `tools/setup_env_inventory.sh` convention for this class of script. HF token support added (`tools/.hf_token`, gitignored, read by `_hf_token()`) though this particular repo turned out to be public — no auth was required for any of the runs.

### Results (75MB compressed sample per file, extrapolated to full file size)

**No tagged VLA tokens found anywhere** — `<seed2_N>`, `<cosmos_N>`, `<avclm_N>`, `<snac_N>` all zero across all 15 files, confirmed at the 75MB-sample scale (not just the initial 5-record peek).

**2 files carry real SNAC audio tokens, as raw integer arrays** (`snac_token: [128266, ...]`), not `<snac_N>` tag strings:

| File | Size | Sample records | Extrapolated raw SNAC codes |
|---|---|---|---|
| `train_data_snac.jsonl.gz` | 11.1 GB | 131,850 | **~3.11B** |
| `valid_data_snac.jsonl.gz` | 579 MB | 129,996 | **~162M** |
| **Total** | | | **~3.27B raw SNAC codes** |

Comparable in scale to the 4.92B SNAC tokens already found in MixtureVitae-Omni's `valid_snac` (Section 13 data inventory) — a real, previously-uncounted audio-token resource.

**13 remaining files — plain text/caption corpora** (word-count, extrapolated):

| File | Extrapolated text tokens | Content |
|---|---|---|
| high_stack.tar.gz | 4.11B | StackExchange QA |
| valid_text_only.tar.gz | 3.31B | mixed text |
| stack_maga.tar.gz | 1.65B | StackExchange |
| emo.jsonl.gz | 1.04B | audio-transcript + image-caption pairs |
| train_data_snac.jsonl.gz (`text` field) | 865.5M | transcript alongside the SNAC tokens above |
| magalith-10m-florence2.jsonl.gz | 864.4M | image captions |
| synth_llava2.tar.gz | 162.9M | LLaVA-style image captions |
| clappa.tar.gz | 138.4M | video captions (Section 13 DISCUSS-1 candidate) |
| synth_llava.tar.gz | 93.7M | LLaVA-style image captions |
| low_nemo_maga.tar.gz | 73.7M | text |
| valid_data_snac.jsonl.gz (`text` field) | 44.1M | transcript alongside the SNAC tokens above |
| youtube.tar.gz | 38.6M | video storyline/description |
| coco.tar.gz | 10.0M | image captions — **exact** (fully consumed within the sample) |
| europarl.tar.gz | ~0.1M | low confidence — see caveats |

### Caveats

1. **`finevideo_transcripts.jsonl.gz` undercounted (reports 0).** Its real field is `transcripts`, not `text` — the counter only checks `text` (matching the existing `data_inventory.py` convention). Needs a dedicated pass, and — since it's literally FineVideo YouTube transcripts — a video-ID overlap check against this project's own pipeline (same class of risk as the `valid_with_seed` double-counting issue already resolved once, Section 13).
2. **`europarl.tar.gz`'s estimate is close to meaningless.** The first sampled member was a single ~986MB record, so the 75MB sample budget only completed 1 full record. Needs a much larger sample or a dedicated full scan.
3. **Several archives mix huge text members with binary `.wds` shards** (youtube, synth_llava/synth_llava2, stack_maga, high_stack, valid_text_only) — 75MB only reached a handful of members out of many, so extrapolation assumes uniform density across the archive, which may not hold. Lower confidence than files sampled with hundreds of small members (coco, low_nemo_maga).
4. **Raw `snac_token` integer arrays are not in this project's tokenizer's `<snac_N>` string format.** Would need a conversion step (offset/tag scheme) similar to the MV-Omni `seed→seed2` conversion already done (Section 13) before these ~3.27B codes could enter the Megatron pipeline.

### Status

Findings posted to Huu on Discord (Jul 9, 2026, 3:51pm): *"this dataset is mostly text, only train_data_snac.jsonl.gz and valid_data_snac.jsonl.gz have snac tokens ... u want to add it?"* — **awaiting his reply.** No integration or full download has started pending his decision.

## 17. Permissive Dataset Survey — 6 Candidates Investigated, MINT-1T-HTML Download Started (Jul 13, 2026)

While the A2 captioning full run (§2.5e) is in progress, investigated the 6 remaining unscoped candidates from the Jul 7 team chat (§15) to prepare for downloading whatever is actionable. Research done via HuggingFace API/WebFetch, no download attempted for the rejected/deferred candidates.

### Results per candidate

| Candidate | Verdict | Notes |
|---|---|---|
| `mira-wm.com` | **Not relevant, dropped** | Confirmed via search: this is "MIRA," a Rocket League gameplay world model (General Intuition + Kyutai + Epic Games) — video + keyboard actions + game state from bot-vs-bot matches (~10,000 match-hours). No robot/human pose or action data of any kind. Real dataset is `kyutai/rocket-science` on HF, unrelated to this project's needs. |
| `finevla.xlang.ai` | **Deferred — data not yet public** | The actual FineVLA-Data training set (47,159 human-verified trajectories aggregated from 10 robot datasets, 220,606 action steps) is **not released**. Checked the GitHub repo (`xlang-ai/FineVLA`) directly: README says "Coming soon" for policy checkpoints; the only downloadable artifact is `xlangai/RoboFine-bench`, a 500-video **evaluation benchmark**, not training data. Nothing to download until upstream releases it. |
| `nvidia/Cosmos3-DROID` | **Confirmed real, downloadable — architecture decision needed before use** | Raw DROID (real bimanual/single-arm robot teleop data) repackaged to LeRobotDataset v3.0. 71,907 episodes (57,639 success + 14,268 failure), ~22.4M frames @15fps, 707GB, 3 camera streams (2 exterior + 1 wrist) + joint/cartesian/gripper state+action, license OpenMDW 1.1 (commercial-OK). **Not downloaded yet** — this is robot joint-space action data, a fundamentally different representation from this project's xyz human-pose PCHIP tokens (§2.2). Using it requires designing a new robot-action tokenization scheme first (comparable scope to the already-deferred "PAB-Spline joint angles" work), not just a download+count-tokens step. Flagged for Huu to decide scope before investing download time. |
| `MiG-NJU/OmniVideo-100K` | **Deferred — dilution risk** | Video-text QA dataset (multiple-choice + open-ended questions about video content), Apache 2.0, ~100K samples (`mcq_30k`/`oe_70k` subsets). No pose/action signal — would only add more seed2/cosmos/text tokens, same class of risk already flagged for MV-Omni (agent-token ratio dilution, 12.2%→5.2% when MV-Omni was mixed in, §13 Jul 8 update). Not downloaded pending a decision on whether the dilution tradeoff is worth it. |
| `mlfoundations/MINT-1T-HTML` | **Downloaded — see below** | Interleaved text+image web dataset (CommonCrawl HTML, 2017-2024), CC-BY-4.0. Directly addresses the still-open DISCUSS-1 language-data-mix gap (§1, §13) — FineVideo v4's 5.217B tokens are ~100% modality-specific (cosmos 74.4%/agent 12.2%/snac 7.0%/seed2 6.4%), essentially zero plain natural-language text, so this is the first candidate that adds real language grounding at meaningful scale. |
| `genrobot2025/Gen-EgoData` | **Deferred — small scale, format cost** | Egocentric human video + ego-SLAM pose + actions for domestic tasks (kitchen/bedroom/living-room/study; folding clothes, organizing, etc.), CC-BY-SA-4.0 (note: share-alike, has downstream licensing implications unlike the other CC-BY/Apache candidates). Only 500 samples / 4.23 hours total — closest structural match to this project's own pipeline (egocentric perspective, pose+action) but too small to move the token-count needle; value would be qualitative (viewpoint diversity) not quantitative. Data stored as `.mcap` (ROS-like), requires the `genrobot-ai/das-datakit` toolkit to load — not started.

### Key architectural distinction surfaced this session

Datasets split into two classes with very different integration cost:
1. **Raw video** (OmniVideo-100K, and FineVideo itself) — this project's own HRNet→MotionBERT→PCHIP pipeline can process it end-to-end under the project's own joint/coordinate conventions. Integration cost ≈ compute time only.
2. **Pre-posed/pre-actioned data** (Cosmos3-DROID's robot joint-space, Gen-EgoData's ego-SLAM `.mcap`) — each source has its own skeleton/joint convention, coordinate frame, and action representation. Integrating these is a **retargeting problem** (design + validate a mapping into this project's token format, possibly a wholly new modality for robot-embodiment actions), not a data-ingestion problem. Recommendation: don't invest download time in class-2 sources until there's an explicit decision on whether/how to add a robot-action modality distinct from the current human-pose agent tokens.

### MINT-1T-HTML download — in progress

**Size correction:** the dataset card advertises "1 trillion text tokens / 3.4B images / 5.91TB" for the full MINT-1T project (which also includes PDF and ArXiv splits), but the `mlfoundations/MINT-1T-HTML` repo used here is the **HTML-only config** (`data_v1_1`). Measured directly via the HF tree API (paginated, all 6,159 files): **2.89TB actual size**, not 5.91TB.

**Schema (inspected from a downloaded shard, `pyarrow.parquet`):**
```
images:           list<string>   -- image URLs (NOT embedded bytes)
texts:            list<string>   -- actual interleaved text content
metadata:         string (JSON)  -- per-image source info (document_url, unformatted_src, ...)
url:               string        -- source webpage URL
image_hashes:      list<string>
images_metadata:   list<string>
cc_dump:           string        -- source CommonCrawl dump id (e.g. "CC-MAIN-2017-22")
```
**Important finding: the `images` column is a list of image URLs, not raw image bytes.** The `texts` column is directly usable (real text, tokenizable now with this project's tokenizer). Getting actual pixels — needed if the plan is to run these through the Seed2 tokenizer to add `<seed2_N>` tokens alongside the text — would require a **separate crawl step per URL**, with an unmeasured but likely significant dead-link rate given the source pages are from 2011-era blogs crawled in CommonCrawl 2017. Not attempted yet; text-only ingestion is the safe near-term plan.

**Download launched:** `tools/extract/download_mint1t_html.py` (new script) uses `huggingface_hub.snapshot_download` with `allow_patterns=["data_v1_1/*.parquet"]`, 16 parallel workers, and an outer retry loop (safe to interrupt/resume — skips files already complete). Running in a detached `tmux` session (`mint1t`), logging to `logs/download_mint1t_html.log`, target `/p/data1/mmlaion/shared/vla/mint1t_html/` (per-project convention for shared downloaded data, 390TB free at `/p/data1` so no capacity concern for 2.89TB). Copied the existing cached HF token into the custom `HF_HOME` (`/p/data1/mmlaion/nguyen38/hf_cache`) to get authenticated rate limits (~50MB/s single-thread measured vs ~31MB/s unauthenticated).

**Progress at end of session (07:13, Jul 13):** 249/6,159 files, 204GB/2.89TB (~7%), running ~43 minutes, steady rate ~4.7GB/min → **ETA ~10 hours from start** (~9.3h remaining). No errors. Text-token counting against this project's own tokenizer (same method as the §16 MixtureVitae investigation) is the natural next step once a meaningful fraction has landed, to convert the raw 742B-token (their tokenizer) figure into a real budget-relevant number.

**Next steps (not started):** (1) let the download finish or grow further, (2) sample-tokenize a subset of `texts` with the project's own GPT-NeoX+VLA vocab tokenizer to get a real token count, (3) decide with Huu how large a slice of the 2.89TB is actually needed for DISCUSS-1 (likely far less than the full corpus, given the "few billion tokens" target), (4) decide separately whether the image-URL-crawl-for-seed2 idea is worth pursuing given expected link rot.

---

## 18. Caption+Speech Interleaving Pipeline — Implementation Started, Two Real Bugs Found (Jul 14, 2026)

While A2 continues running (job chain `14104155`→`14104156-159`, caption count 11,501→13,783 between Jul 13 and Jul 14 checks), started implementing the approved plan to interleave `<caption>` and `<speech>` tags into the flattened token sequence at modality-transition points — the fix for root cause #2 of the "model can't self-initiate modality transitions" finding (no language anchor explaining what's happening at each timestamp).

### 8-task breakdown and status

1. **Video→shard manifest** (`tools/analysis/build_video_shard_manifest.py`) — DONE. Maps all 43,751 `video_id`s to their `HuggingFaceFV/finevideo` parquet shard index.
2. **Speech extraction script** (`tools/analysis/extract_speech_segments.py`) — coded, two real bugs found and fixed (detailed below).
3. *(not a new script — the already-running A2 SLURM job itself)*.
4. **Caption dict adapter** (`tools/analysis/build_caption_dict.py`) — coded, logic-tested against real A2 output; **not yet run at full scale** (`captions_dict/` doesn't exist on disk yet).
5. **Tokenizer rebuild** — added 4 wrapper tokens (`<caption>`, `</caption>`, `<speech>`, `</speech>`) to `tools/tokenizer/build_tokenizers.py` and `tools/tokenizer/expand_vocab.py`. `tokenizer_vla_adaptive_v2` rebuild confirmed complete: vocab 156,509 (144,215 base+SNAC + 4 new), all 4 new tokens verified atomic, spot-checked all pre-existing token categories (seed2/cosmos/avclm/pelvis/SNAC/agent/fps) still atomic too. `tokenizer_vla_qwen3` rebuild was in progress at time of writing.
6. **`phase6_merge_adaptive.py`** — NOT YET EDITED. Pre-implementation check of the cosmos/avc_lm 1:1 pairing invariant (needed before trusting the planned index-based splice logic) found 1 mismatch in a 2,753-activity sample: video `bg9y_imduwQ`, activity `scene_8_act_1`, 183 `<avc_lm>` vs 184 `<cosmos>` — traced to a dangling trailing incomplete chunk at the very end of the activity's frame range (already excluded from `avc_count`/`chunk_timing` elsewhere in the pipeline, so pre-existing and invisible, not a new bug). Planned indexing (loop bounded by `len(avc_matches)`, index into `cosmos_matches` by the same index) should be safe against this specific trailing-only failure mode, but a broader 5-shard check to confirm no *interspersed* (non-trailing) mismatches exist was interrupted and not yet re-run.
7. **`phase7_flatten.py`** — not started.
8. **End-to-end dry run** — not started.

### Correction: no new Whisper compute needed

Earlier informal framing called this "the Whisper pipeline," but `extract_speech_segments.py` does not run any ASR model. FineVideo already ships a pre-computed per-video transcript (`timecoded_text_to_speech`, sourced from YouTube-Commons ASR) in its HF Hub parquet files. The script's job is purely to re-fetch that field and re-align it onto the 8-frame `chunk_timing` grid already stored per-activity — a mapping/data-wrangling task, not new model inference.

### Two real bugs found while producing sample output

**Bug 1 (initially misdiagnosed as an HF-fetch issue):** a quick manual test (`--video-ids iWv3M3cSBs8,vd6hr_AtYtQ`, 2 videos) on the shared JUWELS login node drove process RSS to 90+ GB within ~9 minutes with no sign of leveling off; killed to protect the shared node (754GB total, other users present). First hypothesis was that `pq.read_table(path, columns=["json"], filesystem=fs)` reading via `HfFileSystem`'s remote streaming was buffering inefficiently — switched to `hf_hub_download()` (download to local `$HF_HOME` cache, then read the local file with plain `pyarrow`). That change is real and worth keeping (avoids repeat-download cost across videos sharing a shard, and is a generally safer I/O pattern), but re-testing showed **the same unbounded growth curve, proving this wasn't the actual cause.**

**Real cause:** `load_activities_needing_speech()` — called before any `--video-ids` filtering — defaults to the unrestricted `INPUT_GLOB_DEFAULT`, i.e. **all 160 files of `final_dataset_adaptive_v3/`, 663GB total** (confirmed via `du -sh`), and for every video with `chunk_timing` it retained the **entire activity dict**, including the `video_tokens` field (the full per-activity token string — potentially hundreds of KB each, given cosmos alone is 74.4% of the corpus's 5.217B tokens). With no early filtering, memory grew roughly proportional to how many of the 160 files had been scanned so far, well past what any 2-video test needed.

**Fix (both applied):**
1. `load_activities_needing_speech()` now only retains the 3 fields actually used downstream (`activity_id`, `chunk_timing`, `time_range_sec`), not the full activity dict.
2. The `--video-ids` allowlist (when given) is now applied *during* the per-line scan, not as a post-hoc dict filter after everything was loaded.

**Verified fix:** re-ran the identical 2-video test — RSS stayed under 500MB for the full run (vs. 90+ GB unbounded before). Note the production full-scale path (32-way SLURM array, each worker already only touches its `SLURM_ARRAY_TASK_COUNT`-sliced ~5 files via `input_paths[start:end]`) was always structurally less exposed to bug 1 than the login-node quick-test path (which had no such slicing), but the field-trimming fix (bug 1's real fix) reduces every worker's memory footprint regardless of slicing.

### Tokenizer upload — pending user action

`tools/upload/upload_tokenizers_v2.py`'s baked-in model cards were updated to describe the 4 new tokens (adaptive_v2: 156,505→156,509 vocab; qwen3: 257,897→257,901 vocab), including a new changelog note in each README. Not run — needs the user's own `HF_TOKEN` exported first. Once `tokenizer_vla_qwen3`'s rebuild is confirmed complete:
```bash
export HF_TOKEN=...   # user's own token
python tools/upload/upload_tokenizers_v2.py --mode all
```
`tokenizer_vla_qwen3` rebuild was moved into a tmux session (`qwen3_rebuild`) partway through, at the user's request, for direct visibility (`tmux attach -t qwen3_rebuild`) — the original `nohup` run was killed and restarted from scratch in tmux (lost ~37 min of prior progress, judged an acceptable tradeoff for future visibility on long-running jobs). Still running as of this entry; base Qwen3 vocab confirmed loaded at 151,669.

### Task #2 full-scale launch — internet-access constraint + a real disk-quota bug (Jul 14, 2026, later same session)

**Why login node, not SLURM:** `extract_speech_segments.py` calls `hf_hub_download()` to fetch per-shard parquet from the HF Hub — this needs internet access, and JUWELS compute nodes have none (`HF_HUB_OFFLINE=1` required there, per the standing cluster constraint). Since the workload itself is I/O-bound (network fetch + JSON parsing), not GPU/CPU-heavy, running it directly on the shared login node was judged the practical choice over a two-phase predownload-then-SLURM-offline design.

**`tools/analysis/run_speech_extraction_login.sh` written:** launches 8 parallel `extract_speech_segments.py` workers (splitting the 160 `final_dataset_adaptive_v3` shard files via the same `SLURM_ARRAY_TASK_ID`/`SLURM_ARRAY_TASK_COUNT` env-var convention the script already supports, just set manually instead of by SLURM), each under `nice -n 15` + `ionice -c3` to stay polite on the shared node (other users present, `uptime` showed 10 active sessions at launch time). Resume is free: per-video output files + `--skip-existing` means a kill-and-rerun picks up only unfinished videos.

**First launch: all 8 workers crashed within ~20 seconds** — `RuntimeError: ... File reconstruction error: IO Error: Disk quota exceeded (os error 122)`. Root cause: the runner script never set `HF_HOME`, so `hf_hub_download()` defaulted to `~/.cache/huggingface` — the small-quota home directory (already a documented gotcha elsewhere in this project's history, e.g. the Jul 12 session note "always set `HF_HOME=/p/data1/mmlaion/nguyen38/hf_cache`... to avoid Disk quota exceeded" — simply forgot to apply it to this new script). 8 parallel workers downloading ~450-500MB parquet shards each blew through the remaining home quota in under a minute (home usage measured at 7.7GB right before the crash, 9.6GB right after).

**Fix:** added `export HF_HOME=/p/data1/mmlaion/nguyen38/hf_cache` (+ `HF_HUB_DISABLE_XET=1` for the known Xet-backend flakiness) to `run_speech_extraction_login.sh`. Cleaned up the 1.9GB of partial `HuggingFaceFV/finevideo` parquet cache the crashed run left behind in `~/.cache/huggingface` (courtesy cleanup, home dir shared with other project caches). Relaunched — confirmed healthy: 8/8 workers alive, ~100% of 1 core each (8/80 total login-node cores, ~10%), RSS 300-400MB/worker (consistent with the earlier memory-bug fix holding), and `hf_cache` under `/p/data1` growing correctly (16GB and rising) while `~/.cache/huggingface` stayed flat. Running in tmux session `speech_full`, per-worker logs at `logs/speech_extraction_login/worker_*.log`, output target `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/speech_segments/`.
