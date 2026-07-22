# FineVideo-VLA: Full Project Report

**Author:** Van Khue Nguyen  
**Date:** June 2025 – June 2026  
**Cluster:** JUPITER (JSC), `booster` partition, GH200 nodes

---

## ⚠️ Project Scope Update (Jul 20, 2026)

Everything in this report describes the **FineVideo/OmniVideo-100K video + 3D-pose branch** — the
most-built-out part of the project and the origin of the "VLA for humanoid robot" framing. As of
Jul 20, 2026, Huu (project lead) clarified directly (Discord) that the actual project scope is
broader: an **omni-modal** model binding *any* modality pair — image, video, sound, action, IMU,
etc. — not specifically humanoid-robot data. Quote: *"omni means all modes... as long as we
balance the dataset and create cross modal bindings."* The acceptance bar for a new data source is
**license permissiveness + balanced modality mix + demonstrable cross-modal binding**, not "does
it involve video/pose/action."

Two sources pulled in under this broader scope, neither of which fits the video+pose branch at
all:
- **`synth_llava`/`synth_llava2`** (`mixture-vitae-backup/MixtureVitae-Backup/data/multimodal`,
  Huu's own dataset) — ~604K synthetic (AI-generated) image+caption pairs, 256×256 PNG, WebDataset
  format (151 shards). No video, no action, no audio. Intended to be tokenized as `<seed2_N>` —
  the only existing tokenizer in this project that accepts a standalone image (`cosmos` needs an
  8-frame temporal window, `avc_lm` encodes H.264 *video* bitstream motion, `agent` needs a lifted
  3D pose sequence). Samples in `samples/synth_llava_sample/`.
- **`laion/emotional-roleplay-finetuning-dataset`** — 67,491 fully-synthetic TTS speech clips
  (~184h, MOSS-TTS-generated, German-dominant + en/es/fr) pairing `text` + a DramaBox-style
  `voice_description` caption with generated audio (mono MP3 24kHz). Intended for `snac` and/or
  "moss" tokens (per Huu's instruction: "concatenate the text and interleave with snac and/or moss
  tokens"). Downloaded to `/p/data1/mmlaion/shared/vla/laion_emotional_roleplay/`.

**Open concern (not resolved, raised by Van Khue Jul 20):** the project's eval protocol is still
listed as "Still open" below (Pre-training Blockers, item 3), and there is not yet a single fixed
central research question that each new data-source addition is being justified against — data
sources are currently being added per ad-hoc Discord instruction rather than against a documented
acceptance test. This is a real risk to paper/scientific-rigor feasibility if scope keeps expanding
without eval discipline catching up; flagged to the team, not yet acted on.

See `../CLAUDE.md`'s Project Overview and `datasets.md` for the fuller external-dataset inventory,
and `PROGRESS_VI.md`'s 2026-07-20 entries for the full discussion.

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

**⚠ Bug found and fixed (Jul 20, 2026): `apply_2d_mask()` frame-index mismatch + missed occlusion masks.**
While extending this pipeline to a new dataset (OmniVideo-100K, see
`data_prep/omnivideo_100k/`), two real bugs were found in
`apply_2d_mask()`, used by every Phase 3 run to date:

1. **fps-mismatch:** the function compared `pose3d` (already resampled to
   the 30fps grid by Phase 2.5) against the 2D confidence JSON (still on
   the video's *native* fps timeline) frame-for-frame, via
   `num_frames = min(len(pose3d), len(pose2d))`. For any video whose
   native fps != 30, this both (a) left every resampled frame past the
   native frame count completely unmasked, and (b) masked the frames it
   did cover against the wrong real-world timestamp — same class of bug
   found independently in Phase 4 (below). Verified on real production
   data (25fps video): 600/3,600 resampled frames were never masked at
   all. **35% of FineVideo's 43,751 videos** have native fps deviating
   ≥5% from 30 and are affected to varying degrees.
2. A second, OmniVideo-100K-specific bug (the "missing joint" check
   required confidence == 0.0 too, which only holds for this script's
   original binarized-confidence convention) does not affect FineVideo,
   whose upstream Phase 1 driver already binarizes confidence.

**Fix:** `pipeline_pose/phase3_kinematics_processor.py` was patched in place
(endpoint-aligned `native_idx <-> resampled_idx` linspace mapping, same
technique as the Phase 4 fix; the "missing joint" check now looks at x,y
only). **Full-dataset rerun in progress** (SLURM job `978074`, submitted
Jul 20, 2026, pending on a cluster maintenance window) — this invalidates
`outputs/states_jsonl_30fps/` as previously described above; the pre-fix
output was moved (not deleted) to
`outputs/states_jsonl_30fps_buggy_2026-07-20/` for rollback. Everything
downstream (Phase 4 cleaning, agent tokens, both trained models) was built
on the pre-fix output — see the Phase 4 note below for the same caveat.

#### Phase 4 — YOLO Person-Presence Cleaning
**Script:** `pipeline_pose/phase4_yolo_cleaner.py` | **SLURM:** `slurm/submit_yolo.sh`

- Ran YOLOv8 person detection on original video frames
- Dropped any 8-frame window where ≥ 4 frames have no detected person (confidence ≥ 0.75)
- Removes windows where subject is off-screen, occluded, or in scene transitions
- Output: `outputs/yolo_cleaned_30fps/{video_id}_cleaned.jsonl`
- **40,195 videos**, **107 GB**

**⚠ Bug found and fixed (Jul 20, 2026): native-fps/resampled-fps frame-index mismatch.**
Found while writing an equivalent driver for OmniVideo-100K: this script
decodes frames sequentially from the *native-fps* source video
(`videos_staging/`) and used the raw decode-order index directly as the
`frame_cache` key — but `window_id` in `states_jsonl_30fps` is indexed on
the *resampled-30fps* grid Phase 2.5 produced. For any video whose native
fps != 30, those are different timelines. Verified on real production
output (25fps video `-2MKTg-LNio`): native frame count is 12,758 but its
`states_jsonl_30fps` runs up to `window_id+8=15,304` — so every window
past 12,758 was silently dropped (losing the last ~1/6 of the video), and
every window that *was* kept read YOLO's person-presence result from the
wrong point in time, drifting by up to ~20% of the video's duration by the
end. **35% of FineVideo's 43,751 videos** (native fps deviating ≥5% from
30) are affected to varying degrees; the same issue does not occur for
already-30fps sources.

**Fix:** `pipeline_pose/phase4_yolo_cleaner.py` was patched in place (added
`--resampled-npy-dir`, builds an explicit `native_idx <-> resampled_idx`
mapping via `np.round(np.linspace(0, N-1, M))` — the same endpoint-aligned
mapping `resample_pose()` uses going the other direction — instead of
assuming the two index spaces are the same). Verified against real data via
both a direct function call and the full CLI path before touching
production output. **The pre-fix output (`outputs/yolo_cleaned_30fps/`,
107GB) was moved — not deleted — to
`outputs/yolo_cleaned_30fps_buggy_fps_mismatch_2026-07-20/`** for rollback,
and a full-dataset rerun is queued (`slurm/submit_yolo.sh`, updated with
the new `--resampled-npy-dir` arg) behind the Phase 3 rerun above (job
`978074` must finish first, since Phase 4 reads Phase 3's output).

**Caveat for existing artifacts:** both trained models
(`vla-1.7b-pab-spline-25b-test`, `vla-1.7b-pab-spline-adaptive`) and the
`FineVideo-Phase4-YOLOPose`/agent-token/tokenized datasets described
elsewhere in this report were built from the **pre-fix** Phase 3/4 output.
This does not make them "wrong" outright — 65% of videos (already ~30fps)
were never affected by either bug — but the ~35% non-30fps videos in the
mix had systematically mistimed occlusion masking and person-presence
cleaning. Re-tokenizing/re-training on the fixed data is a separate,
not-yet-scheduled decision.

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

---

## 19. Status Check + Two Permutation/SNAC Bugs Fixed (Jul 15, 2026)

Re-verified the live cluster state of the §18 8-task breakdown (via `squeue`, `tmux capture-pane`, and log/output inspection, not just re-reading the docs) and found two tasks had actually finished since the last write-up, plus fixed a real correctness bug in `phase7_flatten.py` surfaced during a design discussion with Huu about speech-transcript augmentation.

### Task #2 (speech extraction) — confirmed COMPLETE

`tmux` session `speech_full` shows all 8 `extract_speech_segments.py` workers printed `DONE` ("All workers finished"). Aggregate across workers:

| Metric | Total |
|---|---|
| Videos processed | **40,437** |
| Activities with speech | **303,976** |
| Segments extracted | **2,608,543** |
| Garbled/skipped | ~58K (~2.2%) |

Output: `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/speech_segments/` — 40,490 `{video_id}_speech.jsonl` files (confirmed via `find | wc -l`). Task #2 is now a completed input for task #6.

### Task #5 (tokenizer rebuild) — confirmed COMPLETE

`tokenizer_vla_qwen3` (in progress as of the Jul 14 entry) finished building: vocab **257,901**, all 4 new wrapper tokens (`<caption>`, `</caption>`, `<speech>`, `</speech>`) plus every pre-existing token category (seed2/cosmos/avclm/pelvis/SNAC/agent/fps) spot-checked atomic in the `qwen3_rebuild` tmux pane. Combined with the already-complete `tokenizer_vla_adaptive_v2` (156,509 vocab), both tokenizers are ready; only the HF upload step (§18's pending user action) remains.

### Task #3 (A2 captioning) — still running, far from done

`squeue` shows job `14104156` running its full 32/32 array (~8h45m elapsed at check time), with a 3-job dependency chain (`14104157`→`14104158`→`14104159`, `afterany`) queued behind it — expected, since one `--time` window can't cover all ~913K task points. Worker 0 sample: 800/1275 videos, ~0.03 vid/s, current-job ETA ~287 min. 25,432 `{video_id}_captions.jsonl` files written so far to `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/captions/`. Not yet enough to run task #4 (`build_caption_dict.py`) at full scale.

### Two bugs fixed in `pipeline_pose/phase7_flatten.py`

Surfaced while double-checking, with Huu, whether sentence-permutation augmentation on `speech_transcript` (§2.4/§18 augmentation table) could conflict with real SNAC audio tokens injected into the same activity (Phase 6 v2, §2.3). Confirmed `speech_transcript` is genuinely ASR text (`timecoded_text_to_speech`, per §18's "no new Whisper compute" note) — not FineVideo's separate model-generated `description`/`text_prompt` commentary field — so the conflict is real, not a false alarm.

- **Bug A (the actual conflict):** `permute_sentences` augmentation shuffled `### Speech:` sentence order unconditionally, even for activities carrying real `<snac_N>` audio tokens (which preserve true temporal order). This teaches a spurious mismatch between what the model "hears" (SNAC, correct order) and "reads" (shuffled text). **Fix:** in `flatten_one_file()`, added `effective_permute_rate = 0.0 if sn > 0 else permute_sentences` right before the `process_transcript_into_chunks()` call — `sn` (SNAC token count) was already computed by the existing `count_token_types(kept_tokens)` call a few lines above, no new logic needed to detect SNAC presence.
- **Bug B (masking bug, found while testing fix A):** `permute_chunks_list()`'s `n = max(1, int(len(c) * permutation_rate))` forces at least one swap regardless of the requested rate — passing `permutation_rate=0.0` still performed 1 swap, which would have silently defeated Bug A's fix. **Fix:** added `or permutation_rate <= 0` to the existing `len(chunks) < 2` early-return guard.
- Verified with a standalone unit check: `permute_chunks_list(chunks, 0.0)` now returns the input list unchanged (previously it did not).
- **Not yet committed to git, not yet exercised at scale** — this fix takes effect the next time Phase 7 runs, i.e. as part of the still-unstarted task #7/#8 work from §18.

### Suggested next step

Task #3 (A2) will take a while longer, and tasks #6–8 need both speech (now 100% ready) and captions (not yet). Task #6 (`phase6_merge_adaptive.py` — add `--captions-dir` + `--speech-segments-dir`) can start now regardless, since it only needs the speech-segments input, which is fully ready.

---

## 20. Tasks #4/#6/#7 Coded, Tested, Committed; Task #6 Launched Full-Scale (Jul 17, 2026)

Task #3 (A2 captioning) finished completely overnight between the Jul 15 and Jul 17 sessions — confirmed via `sacct` (job chain `14104157`→`158`→`159` all `COMPLETED`, the last two finishing in under a minute each because `--skip-existing` found nothing left to do) and an exact line count: **912,998** caption lines across 40,798 files, matching task A1's target exactly. This unblocked tasks #4, #6, #7.

### Task #4 (`build_caption_dict.py`) — run, verified

Reshapes A2's flat per-anchor-point output into the `{activity_id: {chunk_idx: "<caption>...</caption>"}}` per-video shape `phase6_merge_adaptive.py` needs. Ran on all 40,798 files: **40,798 videos, 912,998 caption lines → 372,385 activities, 0 chunk collisions** — exact match with A1's known totals. Output: `captions_dict/`. Verified further by reconstructing the dict from 5 random videos' flat input and diffing against the actual output — byte-for-byte match.

### Task #6 (`phase6_merge_adaptive.py`) — coded, tested, one critical bug caught before it could corrupt data

Added `--captions-dir` / `--speech-segments-dir`, loading `build_caption_dict.py` and `extract_speech_segments.py` output respectively. Per-chunk insertion order is now:

```
[<caption>?] <cosmos>...</cosmos> <avc_lm>...</avc_lm> [<agent>?] [<snac>?] [<speech>?]
```

Caption anchors immediately before `<cosmos>` (found via a new `COSMOS_PATTERN` matched independently of `AVC_PATTERN`); agent/snac (existing) and speech (new) anchor immediately after `</avc_lm>`, in that order. `inject_chunk_tokens()` was rewritten from a single-anchor-per-chunk design to a two-anchor (`before-cosmos` / `after-avc_lm`) event-list-and-sort design to support this. `build_chunk_timing()` gained `has_caption`/`has_speech_inline` flags.

**Critical bug caught in dry-run, before touching real data at scale:** this script is designed to run a *second* time on top of `final_dataset_adaptive_v3` — which already has `<agent>`/`<snac>` injected from the original v2→v3 run. Passing `--agent-tokens-dir`/`--snac-tokens-dir` again (needed so `chunk_timing`'s `has_agent`/`has_snac` stay accurate) would, without a guard, **re-inject agent/snac a second time**, duplicating that content. Fixed with an idempotency guard in `process_activity()`: detect `"<agent>" in video_tokens` / `"<snac>" in video_tokens` before injecting; if already present, skip injection for that modality (report 0 injected, don't double-count misses) while still using the loaded dict for the `has_agent`/`has_snac` flag computation. Verified with a real dry-run on 3 videos/72 activities from `final_dataset_adaptive_v3`: agent/snac tag counts and content identical byte-for-byte before/after, while caption (138 injected) and speech (243 injected) were added correctly.

A second, unrelated bug was caught in the same dry-run: the script's default `--agent-tokens-dir` (`outputs/agent_tokens_adaptive`, relative to cwd) does not resolve to real data from this repo's working directory — the actual location is `/p/data1/mmlaion/shared/nguyen38/data/outputs/agent_tokens_adaptive`. Without the idempotency guard's fallback-safe design, this would have silently produced a `chunk_timing` with `has_agent` always `False` (wrong metadata, though not wrong training tokens, since injection is separately guarded). Full-scale submit script uses the correct absolute path.

### Task #7 (`phase7_flatten.py`) — coded, tested

`process_activity_per_chunk()`'s document-order state machine gained two event types: `caption` (buffered like `seed2`/`cosmos`, flushed at the `avc_lm` trigger in the order caption→seed2→cosmos, matching source document order) and `speech` (emitted immediately, like `snac`, no buffering needed since it has no `avc_lm`-relative ordering constraint). Neither is dropped (0% dropout, same treatment as `agent`) nor text-augmented (no synonym replacement / stopword drop) — both are anchored to an exact chunk, so paraphrasing would break the token-to-moment correspondence that's the entire point of adding them. This is a deliberately different treatment from the existing `### Speech:` header block (built from `activity["speech_transcript"]`), which is untouched and still augmented/permuted as before — the two are intentionally redundant (whole-activity dump vs. precisely-timed anchor), confirmed with the user rather than assumed. `count_token_types()` gained a `mode` tracker so caption/speech words (no distinguishing `<...>` prefix) don't silently land in the catch-all `agent` bucket (a stats-only fix, doesn't affect training text). Default I/O paths bumped `final_dataset_adaptive_v2` → `_v4`, `megatron_dataset_v4` → `_v5`.

**Testing:** 7+6 standalone unit-test groups (54 assertions total) covering insertion ordering, the idempotency guard, cross-chunk isolation (a caption at chunk *i* must not leak into chunk *i±1*'s output), dropout independence, and token-type counting accuracy — plus a real end-to-end dry run (3 videos → Phase 6 → Phase 7) with manual inspection of the flattened output text.

Committed as `5f5492e` (`pipeline_pose/phase6_merge_adaptive.py`, `pipeline_pose/phase7_flatten.py`), pushed to `origin/master`.

### Token growth: measured two independent ways, both converge on ~0.75%

Before running full-scale, measured how much the new caption/speech tokens actually add, since the number is central to deciding whether this pipeline is worth the compute:

1. **Sample-based:** ran the real merge on 3 full shards / 798 videos, then processed 5,312 real activities (the subset that survives Phase 7's `has_agent OR has_snac` filter, rank_0+rank_1) through `process_activity_per_chunk()` with production settings (`drop_rate_cosmos=0.5`, fixed random seed for a clean before/after comparison — an initial unseeded comparison across two separate CLI invocations showed a spurious ~1% discrepancy traced to cosmos-dropout randomness accumulating differently across activities, not a real bug; isolating with a fixed seed per activity resolved it). Result: **73,796,727 → 74,340,242 tokens, +0.737%**.
2. **Exact, full-dataset:** counted words directly across all real output on disk — **912,998 captions, 10,256,494 words** (`captions_dict/`, all 40,798 files) and **2,158,388 speech-chunks, 22,696,606 words** (`speech_segments/`, all 40,490 files) — giving 12,082,490 + 27,013,382 = **39,095,872 new tokens exactly**, against the known real Phase 7 v4 baseline of **5,217,000,000 tokens** (371,888 records, from the completed full-scale run referenced in §8/§16): **+0.749%**.

The two independent methods (a statistical sample through the real code path, vs. an exact word count from disk) agree to within 0.012 percentage points, ruling out a measurement bug.

**Why the number is legitimately small, and why that's expected — not a project setback:** average caption length is 11.2 words, average speech-chunk length 10.5 words; `cosmos` alone is ~75% of total tokens because it emits hundreds of numeric tokens *per chunk*, at every chunk, for the entire activity duration — natural-language text is inherently far more token-compressed than that. **Important scope correction, worth restating explicitly since it caused a real moment of "this seems wrong" for the user this session:** this caption+speech work was scoped from the start as the fix for root cause #2 ("no language anchor at modality-transition points"), a *qualitative* grounding problem — not as the mechanism for the separately-tracked "×4 more training records" goal (§13/§2.5c: the original ×4 figure is captioning **+ perspective framing** combined, where perspective framing — robot/human/cinematic re-framings of the same activity, not yet coded — is the lever that actually multiplies *record count*, not per-record token density). Caption/speech density was already measured and re-scoped once before, in the Jul 12 session (§2.5c: "Đính chính hiểu về mục tiêu ×4"); this entry re-confirms the same conclusion from the token-count angle rather than the anchor-point-count angle, arriving at it independently this time. If the dataset-size problem (2.84B tokens, small for a 1.7B model, per the top-level pretraining blockers) needs to be solved next, perspective framing or an external data source (SenseNova-SI-8M / stera-10m / MixtureVitae-Omni, per §8/§17) is the correct lever — not further tuning of caption/speech density.

### Task #6 launched full-scale

New submit script `slurm/submit_merge_adaptive_v4.sh` (32-array, `partition=batch`, `account=laionize`, `--time=03:00:00`, pattern copied from `submit_merge_adaptive_v3.sh`), input `final_dataset_adaptive_v3/final_vla_adaptive_v3_rank_*.jsonl` (160 files, 663GB), output `final_dataset_adaptive_v4/`, `--skip-existing` for resume safety. Submitted as job **`14114336`**; confirmed all 32/32 array tasks reached `R` (running) state within 15s of submit, worker 1's log showing normal progress (`5/160 files` shortly after start). Not yet confirmed complete as of this entry.

**Not yet started:** full-scale Task #7 (needs `final_dataset_adaptive_v4/` to exist first, i.e. blocked on the above job), Task #8 (dry-run was already done at small scale in this session, satisfying most of its intent; a final end-to-end check on the full-scale v4/v5 output is still worth doing before calling the corpus training-ready).

---

## 21. Tasks #6/#7 Confirmed Complete + Uploaded to HF; New External Dataset Survey (`datasets.md`); MINT Image License Rejected; SenseNova-SI-8M Download Started; Egocentric Perspective Rejected; Megatron Tokenize Pipeline Set Up for 3 Sources (Jul 18, 2026)

### Task #6/#7 full-scale run — confirmed COMPLETE (was "not yet confirmed" at end of §20)

Checked `sacct` for job `14114336` (Phase 6 v4 merge, 32-array) and `14114370` (Phase 7 v5 flatten, dependent): both **COMPLETED, exit 0:0**. `14114336` finished 08:26 (31 min), `14114370` finished 09:04 (38 min). Grepped all 32 `.err` logs for `error|traceback|exception` — 0 matches; the only content was a harmless "stage deprecated" module warning + tqdm progress bars.

**Output verified against pre-run estimates:**

| Metric | Predicted (§20) | Actual (measured) |
|---|---|---|
| Token growth vs v4 baseline | +0.737%/+0.749% (2 methods) | **+0.740%** |
| Caption tokens | 12,082,490 | 12,076,047 (0.05% off) |
| Speech tokens | 27,013,382 | 27,012,397 (0.004% off) |
| Total records | 371,888 (unchanged, filter didn't change) | **371,888** (exact match) |

Total: **5,255,589,397 tokens** (5.256B) — seed2 6.3%/cosmos 73.9%/agent 12.1%/snac 6.9%/caption 0.2%/speech_inline 0.5%. Spot-checked double-injection guard on 2,787 activities in `rank_0` (agent open/close tag counts) — 0 mismatches, guard worked as intended on top of v3's pre-existing `<agent>`/`<snac>`. Spot-checked actual flattened text content — caption/speech anchored correctly before `<cosmos>`/after SNAC respectively, content qualitatively sensible (not garbled/misaligned).

Output: `final_dataset_adaptive_v4/` (Phase 6, 160 files) → `megatron_dataset_v5/` (Phase 7, 160 files, 72GB) at the usual `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/` location.

### Uploaded to HuggingFace — `EmpathicRobotics/FineVideo-Phase7-Flattened` now live with v5 data

`tools/upload/vla_flattened_dataset_card.md` rewritten for v5 (stats table, "What Changed in v5" section, updated modality-dropout/vocab/format-example sections, version history row). `tools/upload/upload_flattened_hf.py` defaults bumped: `--source-dir megatron_dataset_v5`, `--shard-prefix flat_final_vla_adaptive_rank` (verified against the actual 160 files on disk — v5's naming dropped the `_v2` infix that v4 had), `--upload-dir hf_upload_flattened_v5`. User ran the upload with their own `HF_TOKEN`; confirmed live via HF API (`lastModified` = run date, 162 siblings = 160 shards + README + `.gitattributes`).

### New file: `datasets.md` — dataset-by-dataset survey (overview / download status+path / per-modality tokenize status / structure+extensible-token-types / Megatron-readiness)

Created at repo root to answer a recurring question ("what do we actually have, is it downloaded, is it tokenized, is it training-ready") without re-deriving it each session. Covers 14 entries: FineVideo-VLA (own), MixtureVitae-Omni (`valid_snac`), MINT-1T-HTML, SenseNova-SI-8M, OmniVideo-100K, MolmoAct2-BimanualYAM-Dataset, Cosmos3-DROID, Gen-EgoData, MixtureVitae-Backup (`data/multimodal`), VALID (`ontocord/VALID` on HF), stera-10m, FineVLA, abc.bot, and a newly-surfaced "MINT PDF data" (Huu says he already downloaded + permissive-filtered it, location unknown, on `leo`). All fields verified against real data/APIs this session (HF tree API for real byte sizes, live parquet schema inspection, live URL-liveness sampling) rather than dataset-card claims — several dataset-card numbers turned out to be wrong or misleading (see below).

**Cross-checked against Huu's own dataset list (18/7 chat)**: his list (MINT-1T-HTML, SenseNova-SI-8M, OmniVideo-100K, MolmoAct2-BimanualYAM-Dataset, Cosmos3-DROID, Gen-EgoData, stera-10m) matches `datasets.md`'s scope exactly — no candidate missed.

### MINT-1T-HTML — image download built, then rejected entirely on license grounds (important, don't redo without new info)

**Download confirmed complete this session** (was in-progress at end of §17/§18): log shows `snapshot_download completed successfully`, 6,159/6,159 parquet files, 2.7TB, at `/p/data1/mmlaion/shared/vla/mint1t_html/data_v1_1/`. tmux session `mint1t` had already exited (finished, not crashed).

**Schema verified with real data (not just the dataset card):**
- `texts[]` and `images[]` are the same length, positionally interleaved, **mutually exclusive** at each index (either `texts[i]` has content and `images[i]` is `null`, or vice versa) — this is the real document structure (text/image/text/image...), not two independent parallel lists.
- `image_hashes[]` and `images_metadata[]` (width/height) are **shorter** than `texts`/`images` and are NOT positionally aligned to them — they're ordered per-actual-image-only (skipping text-only slots). The top-level per-image `metadata` field is shorter still, with no clean index mapping either. **Only `images[i]` (the URL itself) should be used as ground truth** — the auxiliary metadata fields require a URL-based join if ever needed, not index-based.
- ~36.3% of "image slots" actually have a resolved URL (rest are `null` placeholders even at image-typed positions) — measured on a 5,000-record sample.

**Scale reality check (all measured, not estimated from dataset card):** ~850M total records (sampled 200/6,159 files' metadata, extrapolated), avg 3.33 real image URLs/record → **~2.83 billion image URLs total**. A live sample of 60 real URLs (proper `User-Agent` header) showed 91.7% still alive, avg ~97KB/image → **full download would be ~130-180TB**. Given this dwarfed the actual purpose (MINT was added for DISCUSS-1's "few billion tokens" text gap, images were a secondary "nice to have"), a 20-shard pilot (~9.2M images, ~900GB) was scoped instead of a full download, per user decision.

**Pilot build:** `tools/extract/extract_mint1t_manifest.py` (parquet → per-shard JSONL manifest, `{record_id, source_url, cc_dump, texts[], images[]}`) + `tools/extract/download_mint1t_images.py` (concurrent downloader, bucketed folder layout `{shard}/{record_idx // 1000}/{record_idx}_{img_pos}.{ext}` to avoid a flat-directory blowup at billions-of-files scale, resumable via a per-shard `_status/*.jsonl` audit log).

**Real bug found and fixed during the pilot: per-domain rate limiter accidentally capped aggregate throughput to ~10 img/s regardless of `--max-workers=64`.** Root cause: the corpus is dominated by a handful of shared blogspot CDN hosts (`1-4.bp.blogspot.com`); the original design serialized requests per exact domain with a fixed `--min-domain-delay=0.5s`, which — since most images in a random sample resolve to just ~4 unique CDN hostnames — capped the whole 64-worker pool to roughly `4 hosts × 2 req/s = 8 img/s`, confirmed by direct measurement (10 img/s over two consecutive 15-30s windows). **Fix:** replaced the serialize-with-delay `DomainLimiter` with a per-domain `threading.Semaphore(per_domain_concurrency=8)` — still protects small/fragile blogs from being hit by all 64 workers at once, but lets shared CDN hosts run at real concurrency. `--min-domain-delay` CLI arg removed, replaced with `--per-domain-concurrency` (default 8).

**License investigation (the actual reason images were dropped) — read the official MINT-1T-HTML README directly, not just the top-level HF license tag:**
- `cc_dump` is **not** license information — it's the CommonCrawl snapshot/dump ID (e.g. `"CC-MAIN-2017-22"`), a naming collision trap ("cc" = CommonCrawl here, not Creative Commons).
- No field in the schema carries per-image license info (`images_metadata` only has width/height; per-image `metadata` only has `document_url`/`src`/`rendered_width`/`rendered_height`).
- The official README's "Filtering Process" section lists text-quality/dedup/NSFW-safety/size/aspect-ratio filters — **no copyright/license filtering step of any kind**. The README's own License section states: *"We release MINT-1T under a CC-BY-4.0 license, designating it primarily as a research artifact... **users are responsible for ensuring its legal use**... **Users should independently verify compliance with applicable laws before employing MINT-1T for commercial purposes.**"* The `cc-by-4.0` tag applies to mlfoundations' own compiled dataset/text artifact, not a guarantee about each hotlinked image's underlying copyright (images are mostly personal 2011-era blogspot photos with no explicit license).
- **Team decision (18/7 Discord, Huu): drop the image download entirely.** Huu: *"if the mint doesn't have images ignore it"* (in response to being told images are URL-only and license-untrackable); Van Khue: *"the hf dataset is fine"* (confirming the **text** portion is still fine to use — same practical distinction the team already draws for LLM text pretraining under fair-use/TDM norms vs. redistributing raw media, which is why text wasn't dropped along with images).
- **Cleanup done:** the 20-shard pilot's downloaded images (130MB, 1,362 files — pilot was still early/rate-limited when stopped) deleted from disk. The 21GB manifest (`manifest/`, from `extract_mint1t_manifest.py`) was left in place but is no longer needed for the text-only path going forward (kept only as a record of the schema-verification work; safe to delete if space is needed).

**`stera-10m` also dropped the same session** — self-assessed "not permissive" by Van Khue in the same chat, no objection from Huu, same "permissive-only" bar applied.

### SenseNova-SI-8M — investigated with real data, decided to download in full, download in progress

**Verified structure via real data (not the dataset card, which is misleading about the `full` config):**
- The `full` config's stated file (`SenseNova-SI-8M.parquet`, 851MB) does **not** embed image bytes — schema confirmed live: `image: list<string>` (relative paths only), `conversations: list<{from, value}>` (ShareGPT format), 8,164,067 records total. This differs from the small `preview` config (`SenseNova-SI-8M_1000samples.parquet`, 823MB, 1000 samples) which DOES embed bytes (`image: list<{bytes, path}>`) — a separate, auto-converted convenience file for the HF dataset viewer, not representative of the real download.
- Real image bytes live in **53 independent zip files** (`images_part_001.zip`...`images_part_053.zip`, ~21.5GB each except the last ~7GB, confirmed via HF tree API: 1.10TB total). Each zip preserves a slice of one shared `images/` directory tree (not split volumes, not one-zip-per-record) — extracting all 53 into the same destination directory reconstructs the complete tree. **Join mechanism**: `full_path = f"{extract_dest}/{record['image'][i]}"`, e.g. `image: ["images/059/034763.jpg", ...]` — confirmed directly from real parquet content and the repo's own `extract_all.sh`/README.
- Total real size: **1.13TB** (53 zips + parquet).
- Content: multi-image (avg 2-4/record) multiple-choice spatial-reasoning VQA — object localization, relative compass-direction reasoning, cross-image object identification, mostly indoor-scene context (kitchen/living-room/bathroom). This is closer to genuine embodied/robot-relevant spatial reasoning than a generic web-QA dataset, and Apache-2.0 licensed with self-contained image bytes (no dead-link or license-via-crawl risk, unlike MINT).

**Decision: download the full 1.13TB** (all 53 zips + full parquet) — best available "static image" candidate right now (permissive, self-contained, on-topic for spatial/embodied reasoning). Explicitly skipped the redundant `SenseNova-SI-8M.jsonl` (5.83GB, same content as the parquet in a more verbose format) and the small preview parquet (redundant once the full one is downloaded).

**Script:** `tools/extract/download_sensenova_si8m.py` (same `snapshot_download` retry-loop pattern as `download_mint1t_html.py`, `allow_patterns=["*.zip", "SenseNova-SI-8M.parquet"]`, prints `X/53 zip parts, Y GB on disk` progress on every retry attempt for resumability). Target: `/p/data1/mmlaion/shared/vla/sensenova_si8m/`. Running in tmux (login node, needs internet). **As of this entry: ~71GB/1.13TB downloaded, ~45MB/s real measured throughput (verified via repeated `du -sb` deltas, not just trusting the tqdm display), ETA ~6.5h.** The `Fetching 54 files: 2%|...1/54` progress line appearing frozen for 20+ minutes was diagnosed as expected behavior, not a hang: 16 workers download 16 files' *partial* chunks in parallel, and the file-completion counter only increments on a fully-finished 21.5GB file (~2h at the per-file share of bandwidth), while real bytes-on-disk grow continuously and were confirmed growing via direct measurement. Extraction (53 zips → unified `images/` tree, join with parquet paths) is a follow-up step, not yet coded — planned once the download finishes.

### Egocentric perspective converter — designed, then REJECTED after value-of-output scrutiny (important negative result, don't restart without addressing the core issue below)

REPORT.md's Priority 3 (dating back to the earliest roadmap entries) framed ego-centric pose reprojection as a "free ×2 data-diversity multiplier" — same underlying motion, re-expressed in a head-camera-relative coordinate frame (rotate axes to body-forward via `head_top`/`thorax`, recenter origin at `head_top` instead of pelvis) instead of the current pelvis-centered-but-camera-oriented frame. A concrete design was worked out this session (new `<agent_ego>`/`</agent_ego>` wrapper tags — distinct from `<agent>` to avoid token-space ambiguity between two different coordinate systems sharing the same joint/xyz vocabulary; separate flattened records rather than injecting both perspectives into one record, to avoid context bloat and avoid ever needing the model to disambiguate within a single training step).

**Rejected on reflection, prompted by the user directly questioning the value:** the video tokens (seed2/cosmos/avc_lm) in an "ego" variant record would stay **completely unchanged** — still the original third-person YouTube footage — while only the pose *label* gets re-expressed in a coordinate frame that would only correspond to reality if a head-mounted camera had captured the scene, which never happened (FineVideo is 100% third-person YouTube video; no genuine egocentric camera footage exists to pair with it). Two consequences: (1) **the video→pose training pair becomes physically incoherent** — the model would be taught to map external-camera visual context to a pose coordinate system that visual context can't actually justify; (2) since the transform is a pure isometry (rotation + translation, invertible, no information loss), **`<agent>` and `<agent_ego>` are informationally identical for any downstream robot-retargeting use** — generating the "ego" variant adds zero new knowledge about the underlying motion, only a second way of writing down data already fully present in `<agent>`. Net assessment: likely near-zero value for the project's actual goal (video→action mapping) and a plausible source of training noise, not signal. Would only become justified if (a) genuine egocentric video is eventually available to pair with it (not the case for any current data source), or (b) used narrowly for a pose-only (no-video) generative sub-task, which is a much smaller/different scope than originally planned.

**Decision:** do not implement as scoped. Left `tools/extract/download_mint1t_images.py`-style code was never written for this (design-only, no scripts created). If revisited later, the design above (esp. the `<agent_ego>` disambiguation tag) is still the right starting point, but should only proceed once there's a real paired egocentric-video source.

### Megatron tokenize pipeline set up for 3 sources — scripts written, syntax-checked, NOT YET SUBMITTED

Long-open TODO (tracked since the Jul 8 session, restated in every subsequent session's "Immediate Action Items"): the currently-trained model's `.bin/.idx` (`/p/data1/mmlaion/shared/vla/tokenized_output/vla_adaptive/`, 2.84B tokens) predates SNAC/caption/speech entirely and uses the v1 (144,215-vocab, no-SNAC) tokenizer — **no Megatron tokenize has ever been run against v2/v3/v4/v5 FineVideo data or against MV-Omni**, despite both being "training-ready" content-wise for a while.

**Found the real tokenize script** (`/p/data1/mmlaion/nguyen38/mv-scale/tokenize_vla_adaptive.sbatch`) — lives in a cross-project shared tooling directory (`mv-scale/`, not this repo), Ray-distributed via a custom `mv_preprocess_data.py`, historically submitted under account `cstdl` (per the project's JUSUF/JUWELS/JUPITER account-mapping table) — user recalled last actually running it on JUSUF. This session tried switching to `--account=laionize --partition=batch` on JUWELS instead (per user instruction) — **verified before writing anything**: `sacctmgr show assoc user=nguyen38 format=account,partition` confirms `laionize` has a valid association with `batch` (among others), and the shared infra paths (`SHARED_MAMBA_ENV`, `MEGATRON_PATH`, `mv_preprocess_data.py`) are readable by this user despite living under a `ccstdl`-group-owned path (`/p/project1/ccstdl/...`) — SLURM account/billing is independent of filesystem read permissions, which were already granted.

**Format gotcha found before writing the copies:** `mv_preprocess_data.py --input` only reads `.jsonl`/`.jsonl.gz`/`.jsonl.zst` files with a flat `text` key (confirmed via its own argparse/`find_input_files()` source) — it does **not** read parquet. MV-Omni (`mv_omni_converted/*.jsonl.gz`) and FineVideo v5 (`megatron_dataset_v5/*.jsonl`) already match this shape directly. MINT-1T-HTML's raw parquet (`texts[]` list column, not a flat `text` key) does **not** — required a new conversion step.

**New scripts:**
- `tools/extract/convert_mint1t_text_jsonl.py` + `slurm/convert_mint1t_text.sbatch` (32-array, `account=laionize`/`partition=batch`, matches the project's established worker-split convention) — joins each record's non-null `texts[]` spans (in order) with a blank-line separator into one `{"text": "..."}` JSONL line, dropping the interleaved image positions (text-only path, per the MINT image decision above). Resume-safe (skips shards whose output already exists).
- `/p/data1/mmlaion/nguyen38/mv-scale/tokenize_mv_omni.sbatch`, `tokenize_finevideo_v5.sbatch`, `tokenize_mint1t.sbatch` — each a copy of `tokenize_vla_adaptive.sbatch` with `account=laionize`, `tokenizer-vla-adaptive-v2` (156,509 vocab, has SNAC+caption/speech — same tokenizer across all three so token IDs line up for training-time blending), and dataset-specific `INPUT`/`OUTPUT_PREFIX` (`mv_omni`, `finevideo_v5`, `mint1t_html` respectively). `tokenize_mint1t.sbatch` depends on the conversion job finishing first (separate submit, not a SLURM `--dependency` chain — user will check `squeue` between steps) and uses a longer `--time=24:00:00` given the much larger corpus. All 4 new/copied files syntax-checked (`bash -n` / `ast.parse`) before handoff.

**Explicitly out of scope for this run:** deciding MV-Omni's final training-blend ratio against FineVideo (to avoid diluting the agent-token share) and MINT's final DISCUSS-1 text-volume target — both are training-config-time decisions, independent of the tokenize step, and were deliberately deferred per user request ("mấy cái quyết định drop out để sau đi"). Tokenizing more of MINT than will ultimately be used doesn't force using all of it later, just costs compute time now.

**Status at end of session: none of the 4 new SLURM jobs (convert + 3 tokenize) have been submitted yet.** SenseNova-SI-8M download still in progress in the background (separate tmux). Next session should check both.

---

## 22. Megatron Tokenize Jobs Confirmed Real (MV-Omni Fixed + Completed, FineVideo-v5 Completed); Real Token-Count Discrepancy Found (10.55B vs Reported 5.256B); SenseNova License Retracted; New Permissive-Dataset Survey (RoboVQA/Open X-Embodiment/NVIDIA GR00T-Sim Found, 4 Non-Permissive Candidates Excluded); Gen-EgoData Schema Corrected; Ego/Exo Architecture Question Resolved; OmniVideo-100K + RoboVQA Downloads Started (Jul 18, 2026, afternoon)

### Megatron tokenize jobs — the 4 jobs from §21 were in fact submitted and run between that session and this one (by the user, not by this session); verified real status via `sacct`/logs/output dirs, not SLURM state alone

- **`tok_mv_omni` (job `14117680`, first attempt) — SLURM said COMPLETED but had actually FAILED.** Log showed `mv_preprocess_data.py:528 ray.init(address="auto") → RuntimeError: Failed to connect to GCS`, because no Ray head was ever started in the sbatch script before that call. The script's trailing `ls`/`du` succeeded regardless, so it exited 0 and printed a misleading "Tokenization Complete" despite zero real output — no `mv_omni/` directory existed under `tokenized_output/`.
- **Someone (not this session directly — found already fixed and resubmitted mid-session) fixed the Ray-cluster-startup gap and resubmitted both `tok_mv_omni` (job `14118393`) and `tok_mint1t` (job `14118392`) at 11:54.** Watched both progress live across this session:
  - **`tok_mv_omni` (14118393) — genuinely COMPLETED at 13:18** (4,951s wall time). No tracebacks in the log this time; real Ray workers connected and processed documents at ~150 docs/s/worker across 4 nodes. Output: `/p/data1/mmlaion/shared/vla/tokenized_output/mv_omni/` — 7 shards, 60.94GB.
  - **`tok_mint1t` (14118392) — still RUNNING as of this entry** (>1h40m elapsed), real progress (~1,500-1,580 docs/s/worker), no errors so far.
- **`mint1t_text` convert (job `14117682`, 32-array) — confirmed COMPLETED** (all 32 workers finished cleanly), unblocking `tok_mint1t` above.
- **`tok_finevideo_v5` (job `14117681`) — confirmed COMPLETED**, 4 shards, `/p/data1/mmlaion/shared/vla/tokenized_output/finevideo_v5/`.

### Real token counts — ran `count_tokens.py` against the actual `.bin/.idx` output (not the flatten-stage estimate); found a real discrepancy worth flagging

| Source | Real Megatron-tokenized count (`count_tokens.py`, BIN SIZE CHECK: PASS) | Documents |
|---|---|---|
| **FineVideo-VLA v5** | **10,554,076,391 (10.55B)** | 371,888 |
| **MV-Omni** | **16,357,256,571 (16.36B)** | 1,593,301 |
| MINT-1T text | not yet available (`tok_mint1t` still running) | — |
| **Total tokenized so far** | **~26.91B** | — |

**⚠️ Discrepancy flagged, not yet root-caused:** the real FineVideo-v5 Megatron token count (**10.55B**) is **~2.0x** the figure reported in §21 and carried through `PROGRESS.md`/`datasets.md` (**5,255,589,397 / 5.256B**). Document count matches exactly (371,888 both ways), so this isn't a record-count mismatch. Most likely explanation (not yet confirmed): the 5.256B figure was derived during the Phase 7 flatten stage via a modality-token-type breakdown (counting `<tag_N>` occurrences directly + a word-count approximation for free-text spans like `### Title:`/caption/speech), whereas the real Megatron tokenizer (GPT-NeoX-20b BPE + VLA extensions) subword-splits ordinary English text into meaningfully more tokens per word than a naive word count would suggest — the VLA-specific tags (`<seed2_N>`, `<cosmos_N>`, `<pelvis_x_N>`, etc.) are single atomic tokens either way (that part shouldn't differ), so the inflation is most plausibly concentrated in the natural-language spans (title/context/caption/speech text). **Not yet verified with a byte-level trace** — flagging as a real open question for next session rather than asserting the root cause. Practical upshot: the actual token budget is meaningfully larger than previously documented, which is good news for the "corpus too small for a 1.7B model" concern from earlier sessions.

### SenseNova-SI-8M license — retracted the earlier "no MINT-style risk" conclusion after Huu raised doubt (prompted by ChatGPT) in the 18/7 Slack chat

Investigated with `WebFetch` across the HF README, the GitHub repo (`OpenSenseNova/SenseNova-SI`), the arXiv paper (2511.13719, abstract + PDF), and the live parquet's `image` column itself (paths like `images/059/034763.jpg` — renumbered, no provenance trail). **Found no documentation anywhere stating the images' original source** — no claim of new photography, no source-dataset list (e.g. ScanNet/Matterport3D/ARKitScenes), no license chain for the underlying images. Both the paper and GitHub repo describe the 8.16M samples / 2.72M images as "**curated**", which suggests aggregation from existing sources rather than original capture — structurally the same shape of risk as MINT-1T-HTML's `cc_dump` trap: the HF `apache-2.0` tag most likely covers SenseNova's own QA/annotation layer, not a verified grant over the underlying image copyright. **Correction to §21: no longer treating this as "safe, self-contained, no MINT-style risk" — license status is open and unresolved.** Download was not halted (near-complete, no reason to waste the ~1TB already transferred) but usage-readiness is now blocked pending either the paper's un-extracted appendices or direct author confirmation.

### Gen-EgoData — schema corrected after actually reading the `genrobot-ai/das-datakit` toolkit README (previously "unknown, needs toolkit to read")

Not a "video-ego + human-body-pose" dataset as originally assumed. It's data from a **handheld "DAS device"** (UMI-style handheld gripper-interface) that an operator carries through a task. Real schema: 3 cameras (1 mid-fisheye + 2 stereo, fixed viewpoints, not strictly first-person), and the action/pose signal is `/robot0/vio/eef_pose` = **6-DoF end-effector pose (Pos_X/Y/Z, quaternion) + `Gripper_width`** — a single-arm eef-pose action space, structurally unrelated to the project's existing 17-joint H36M `<agent>` body-pose format. License: data itself is CC-BY-SA-4.0 (share-alike — a real legal constraint, distinct from the toolkit code's MIT license, which covers only the reader/parser, not the data). Re-classified from "ego-video source for FineVideo" into the same "robot-action modality" bucket as MolmoAct2/Cosmos3-DROID/Open X-Embodiment/GR00T-Sim — all blocked on the same not-yet-made architecture decision (how to tokenize robot joint/eef actions), not individually blocked by anything dataset-specific.

### Ego/exo architecture question — extended discussion with Van Khue, resolved: no change needed to FineVideo-VLA

Van Khue's chat decision ("I will go with everything egocentric") raised the question of whether FineVideo-VLA (exocentric 3rd-person YouTube video) needs to be converted or deprioritized in favor of egocentric sources. Traced through the actual pipeline code (`phase3_kinematics_processor.py`, `phase5_adaptive_pchip.py`) to settle it with real evidence rather than assumption: **`<agent>` pose tokens are already root-centred / pelvis-relative** (`retargeted[:, self.pelvis_idx] = 0.0`, docstring `"root-centred metric coordinates"`) — i.e. already body-relative regardless of what camera filmed the source video. "Egocentric" (camera viewpoint) and "root-centred" (skeletal coordinate convention) are orthogonal concepts; there is no "exocentric pose" to convert away from, and head-relative instead of pelvis-relative would be **worse** (head rotation would couple noise into every other joint's coordinates) — not "more egocentric" in any meaningful sense. **Conclusion: FineVideo-VLA's pose representation needs no change.** The real, legitimate concern underneath the ego/exo framing is a train/deploy **video** domain gap (training video is 100% 3rd-person YouTube; a deployed robot only ever sees its own onboard camera) — the correct fix for that is prioritizing the not-yet-integrated **Isaac Sim pipeline** (or a permissive simulation-action source like GR00T-Sim below) for actual embodiment-matched action grounding, not converting or re-scoping FineVideo-VLA's existing pose math, and not chasing small egocentric video datasets for broader pretraining (which don't fix the domain-gap problem anyway — see EgoDex/AgiBot World exclusions below, and Gen-EgoData's small scale above).

### New permissive-dataset survey — 3 solid new candidates found, 4 attractive-looking candidates checked and excluded on license grounds

Searched HF/web for additional robot-manipulation and egocentric-video candidates beyond the 14 already in `datasets.md`, applying the same "verify real license from source, not the top-level tag" standard used for MINT/SenseNova:

| Dataset | License (verified) | Verdict |
|---|---|---|
| **NVIDIA `PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`** | ✅ CC-BY-4.0 | **Strongest new find.** ~345K simulation trajectories used to post-train GR00T N1, includes humanoid (GR1 arms+waist, 240K traj.) and Unitree G1 — directly matches the project's "generalist humanoid VLA" goal and the same "robot-action modality" gap as MolmoAct2/Cosmos3-DROID. Simulation-origin, same nature as the project's own not-yet-integrated Isaac Sim pipeline. |
| **RoboVQA** (Google DeepMind) | ✅ CC-BY-4.0 (materials) + Apache-2.0 (software), confirmed from the official GitHub README | 238h, 3 embodiments (robot/human/human+tool), 829,502 (video,text) pairs. Official access is a GCS bucket via Colab notebook, not a simple HF snapshot — Van Khue found an unofficial mirror (`Tianli/robovqa`) with a matching `LICENSE.txt` (real Apache-2.0 text) — used that. |
| **Open X-Embodiment** (`jxu124/OpenX-Embodiment`) | ⚠️ Registry of 55-60 component datasets, top-level tag (CC-BY-4.0/Apache-2.0) does NOT provably apply to every component | Not blanket-usable — would need a per-component license audit before downloading, same shape of risk as trusting MINT's top-level tag. Deferred, not downloaded. |
| AgiBot World | ❌ CC BY-**NC**-SA 4.0 | Excluded — NonCommercial. |
| Apple EgoDex | ❌ CC-BY-**NC**-ND | Excluded — NonCommercial + No-Derivatives. Conceptually the best fit found all session (829h egocentric dexterous manipulation + full 3D hand/body pose + language) but license kills it. |
| Meta `facebook/ego-1k` | ❌ FAIR Noncommercial Research License | Excluded — also not single-viewpoint egocentric (12-camera rig around a VR headset wearer) and stored as PNG-in-tar, not video. |
| `ut-vision/EgoBrain` | ❌ CC-BY-**NC** 4.0, also off-topic | Excluded — neuroscience (EEG+IMU+video), not robot/manipulation relevant regardless of license. |

`datasets.md` fully updated with all of the above (new sections 15-17 for RoboVQA/Open X-Embodiment/GR00T-Sim, corrected sections 4 and 8 for SenseNova and Gen-EgoData, excluded-candidates row added to the summary table, "Việc còn mở" list extended).

### Downloads started: OmniVideo-100K and RoboVQA

Wrote `tools/extract/download_omnivideo_100k.py` and `tools/extract/download_robovqa.py`, same resumable `snapshot_download`-retry-loop pattern as `download_sensenova_si8m.py`. Both verified against live HF file listings before writing (`videos.tar.part_aa..ae` + jsonl for OmniVideo-100K, 52.9GB total; mp4s + RLDS-style tfrecord shards + json/txt for the `Tianli/robovqa` mirror, ~70.8GB total). User launched both in separate tmux sessions (`omnivideo_dl`, `robovqa_dl`); both progressing normally as of this entry (OmniVideo-100K: 22 files/8.8GB of 52.9GB; RoboVQA: 402 files/1.6GB of ~70.8GB). SenseNova-SI-8M download still in progress in parallel (52/54 files, 1.1TB, occasional resumed timeouts, not stalled).

### Status at end of session

`tok_mint1t` still running (check next session, verify real completion the same way as MV-Omni above — don't trust SLURM state alone). Token-count discrepancy (10.55B vs 5.256B for FineVideo-v5) not yet root-caused. SenseNova/Gen-EgoData/Open X-Embodiment license questions all open, pending either more investigation or a Huu/Van Khue decision. Robot-action-modality architecture decision (blocking MolmoAct2/Cosmos3-DROID/Gen-EgoData/Open X-Embodiment components/GR00T-Sim, 5 candidates total) still not made — GR00T-Sim is the recommended starting point if/when that gets prioritized.

---

## 23. Wrong Tokenizer Fixed + Real Token Counts; SenseNova License Traced to Paper's Appendix (22 Named Source Datasets, Non-Uniform Licensing Confirmed); MINT-1T Paper's Own Datasheet Confirms Image-License Gap; `data_prep/` Built for RoboVQA + OmniVideo-100K (Flatten, Tokenize, Dependency-Free TFRecord Reader, Frame Extraction In Progress) (Jul 18, 2026, late afternoon/evening)

### Wrong tokenizer caught before more compute was wasted — all 3 running jobs were using `tokenizer_vla_adaptive_v2` (GPT-NeoX, 156,509 vocab), not the intended Qwen3 tokenizer

User asked directly whether the tokenize jobs referenced the latest tokenizer ("nhớ là lần này dùng qwen"). Checked `TOKENIZER_MODEL=` in all 3 sbatch scripts — confirmed all pointed at `tokenizer_vla_adaptive_v2`, not `tokenizer_vla_qwen3` (which exists, built 1/7, verified atomic for every VLA tag tested including in-range SNAC IDs — an out-of-range SNAC test id initially looked like a bug but was just picking a numeric ID outside the actual registered range, `<snac_128266>`/`<snac_148745>` both atomic). **`tok_mint1t` (14118392) was still running at the time — cancelled it (`scancel`) before more compute was spent on the wrong tokenizer.** Edited `TOKENIZER_MODEL` in `tokenize_finevideo_v5.sbatch`/`tokenize_mv_omni.sbatch`/`tokenize_mint1t.sbatch` to `tokenizer_vla_qwen3`. Deleted the 215GB of already-produced wrong-tokenizer output (`finevideo_v5`/`mv_omni`/`mint1t_html` under `tokenized_output/`) — necessary because all 3 scripts use `--resume`, which would otherwise silently skip re-tokenizing. Resubmitted all 3 (`14118929`/`14118930`/`14118931`).

### Real Qwen3 token counts — resolves the earlier "5.256B vs 10.55B" discrepancy as NOT tokenizer-related

| Source | Real token count (Qwen3, `count_tokens.py`, BIN CHECK: PASS) | vs. wrong-tokenizer count |
|---|---|---|
| FineVideo-VLA v5 | **10,550,998,369 (10.55B)** | 10,554,076,391 — within 0.03%, essentially identical |
| MV-Omni | **20,389,561,883 (20.39B)** | 16,357,256,571 — real +25% increase (larger Qwen3 vocab matters more for MV-Omni's natural-language-heavy content than FineVideo's mostly-atomic-tag content) |
| MINT-1T text | still running (`14118931`) as of this entry | — |

FineVideo-v5's near-identical count under two different tokenizers **rules out "wrong tokenizer" as the explanation** for the long-standing 5.256B-vs-10.55B gap — confirms the earlier hypothesis (flatten-stage word-count approximation vs. real BPE subword count) was the right track, though still not root-caused with a byte-level trace.

### New downloads: OmniVideo-100K, RoboVQA, and SenseNova-SI-8M all reached COMPLETED this session

- **OmniVideo-100K**: `snapshot_download completed successfully`, 52.9GB.
- **RoboVQA**: completed despite HF rate-limiting (HTTP 429, auto-retried), 70.8GB, 20,736 files.
- **SenseNova-SI-8M**: the tmux session tracking this had silently died (log stale 53 min, no process, session gone from `tmux ls`) — diagnosed via `ps aux`/`tmux ls`/log-mtime rather than assumed; a new tmux launched by this session failed immediately (`HF_TOKEN` unset in this session's shell, `Bearer ` header error) and was killed — **user re-ran it themselves with their own token**, which completed normally: 53/53 zips + parquet, 1,121.4GB.

Also ran and validated 2 small new tokenize jobs (1-node, not 4-node — data too small, ~320MB combined, for a multi-node Ray cluster to be worth the spin-up cost): `tok_robovqa` (14118960, 58,588,270 / 0.06B tokens, 221,912 docs) and `tok_omni_qa` (14118961, 30,689,299 / 0.03B tokens, 99,983 docs), both COMPLETED cleanly.

### `data_prep/` created — dedicated per-dataset folders (`omnivideo_100k/`, `robovqa/`), matching the structure CLAUDE.md already described but that didn't exist on disk yet

**`data_prep/robovqa/flatten_text.py`**: flattens `json/{train,val}/*.json` (181 shards) into `{"text":...}` Megatron-ready JSONL. Real run: 221,912 records in, 221,912 out, 0 skipped.

**`data_prep/omnivideo_100k/flatten_qa_text.py`**: flattens `train_oe_70k.jsonl` + `train_mcq_30k.jsonl` into the same shape. **Two real bugs caught by validating against full-corpus type audits (not just a spot-check sample), per explicit user instruction to validate carefully:**
- 2,740 MCQ records (`task: event_sequence_ordering`) use `question_textual`/`options_textual` instead of `question`/`options` — were being silently dropped. Fixed with a fallback (`rec.get("question") or rec.get("question_textual")`).
- 6,372 OE records (same task type) have `answer` as a `list[str]` instead of `str` — was rendering as Python's `repr()` (`"A: ['B', 'C', 'A']"`) instead of readable text. Fixed to join with `" -> "`.
- Post-fix: ran a full-corpus type audit (`collections.Counter` over every field's `type(...).__name__` across all 99,983+ records, not samples) confirming no remaining unhandled type, then a regex sanity check (`re.search(r"A: \['", t)`) confirming zero Python-repr artifacts remained. Final: 70,017 OE + 29,966 MCQ = 99,983 written, 0 skipped.

Both scripts' outputs written to `sample/`-adjacent shared paths (`/p/data1/mmlaion/shared/vla/robovqa_flat/`, `/p/data1/mmlaion/shared/vla/omnivideo_100k_flat/`), and sample records also written into the repo's `samples/` dir (`omnivideo_100k_scripts_sample.json`, `omnivideo_100k_train_oe_sample.json`, `robovqa_sample.json`) per user request, for quick inspection without querying the shared filesystem.

### OmniVideo-100K video extracted + segment-level caption/speech mapping built (video track now genuinely ready for Step A; text-QA track already tokenized)

Extracted `videos.tar.part_aa..ae` (`cat ... > videos.tar && tar xf videos.tar`) — 5,214 real `.mp4` files, 49GB, count matches `scripts.jsonl`'s video count exactly.

Wrote `data_prep/omnivideo_100k/build_segment_captions.py`: parses `scripts.jsonl`'s `segments[]` (each has `visual[]` for caption text and `transcription[]` for speaker-labeled speech text, timestamps as `MM:SS` strings) into per-video, per-segment records with `start_sec`/`end_sec`/`caption`/`speech`, ready for a future merge step to consume once Step A determines real chunk boundaries. Verified before writing: `transcription` is non-empty in 96.5% of a 200-record sample (the one empty example seen earlier was an atypical silent title-card intro, not representative). Real run: 5,214 videos, 47,467 segments, 0 unparseable timestamps.

**User clarified an important infra constraint mid-session: Step A (video → seed2/cosmos/avclm) must run on JUPITER (has the GH200 GPUs), not JUWELS (where all of this session's work — downloads, tokenize, data_prep — has been happening). Nothing has been submitted to JUPITER yet; this is prepared and waiting for the user's go-ahead.**

### RoboVQA video investigation — real finding that corrects an earlier wrong claim, dependency-free TFRecord parser built, frame extraction running

Earlier in-session claim ("only 9,999/221,912 = 4.5% of RoboVQA videos have an mp4, the rest are effectively missing") was **wrong and has been corrected**: the missing 95.5% aren't missing, they're packed inside the `tfrecord/` shards (184 total: 175 train + 9 val) as `tf.SequenceExample` records with a `feature_lists.images` entry (real per-timestep JPEG bytes) — the standalone `videos/` mp4s are just a small convenience-export subset from the `Tianli/robovqa` mirror, not the primary source.

No tensorflow/protobuf available in any project env (`env_tools`, `env_pose`, `finevideo-vla/env_motion_final` all checked) — per project convention, asked the user before ad-hoc-installing anything; user approved `pip install pypdf` for a separate PDF-reading need (see below) but for TFRecord reading, wrote a **dependency-free protobuf wire-format decoder** instead (`data_prep/robovqa/tfrecord_lite.py`): implements just enough of the wire format (varint/tag decode, length-delimited recursion) to parse `tf.SequenceExample`'s specific message shapes (`Features`/`FeatureLists`/`Feature`/`BytesList`/`Int64List`). Verified in stages before trusting it: (1) exploratory schema-less walk (`inspect_tfrecord.py`) found the field names and JPEG magic bytes (`FFD8FF`) in raw bytes; (2) rendered one extracted JPEG and visually confirmed it's a real robot-arm manipulation scene matching the episode's text; (3) ran the proper parser across 500 real episodes confirming 100% structural consistency (always exactly 16 image timesteps, always the same 6 `feature_lists` keys, always exactly 1 `texts` blob); (4) **cross-checked decoded `texts` content against the already-validated `json/train` data** — found the join key is `video_filename` (tfrecord) == `video` (json), NOT `unique_id`/`uid` (a red herring — different, unrelated ID space) — once joined correctly, decoded text matched byte-for-byte.

Wrote `data_prep/robovqa/extract_frames.py`: writes each episode's 16 JPEG frames to `robovqa_flat/robovqa_frames/<video_stem>/frame_00..15.jpg` + a manifest JSONL (`video_filename`, `timestamps`, `frame_dir`). First test run (`--limit-shards 1`) hit a real bug immediately (`fls["images"]` is a list of 16 `(kind, values)` tuples, one per timestep — not a single tuple — code originally tried to unpack it as one), fixed, re-verified (valid 288×288 JPEGs, timestamps evenly spaced ~600ms apart ≈ 1.6 fps effective sampling rate). Resumable via per-shard done-markers. **Launched as a background process on the JUWELS login node (`nohup ... & disown`, not SLURM)** — judged appropriately lightweight (single-threaded, I/O-bound, comparable to the earlier download processes that also ran directly on the login node) per user's "size-appropriate SLURM-vs-tmux" guidance. **Still running at end of session: 67/184 shards done, 82,669/221,912 episodes extracted.**

**Open architecture question, NOT resolved — flagged clearly to the user, don't assume it's solved:** RoboVQA's video is **16 sparse, discretely-timestamped JPEG frames per episode** (~1.6 fps effective), not continuous video the way FineVideo/OmniVideo-100K are. Step A (Seed2/Cosmos/AVC-LM) is built around continuous video input (H.264 encoding for avclm, temporal windows for cosmos) — whether it's valid to feed it 16 sparse frames as a "fake video," or whether a different treatment (e.g. per-frame static-image tokenization, the same open idea floated earlier for SenseNova's images) is more appropriate, is an unresolved design question. Do not assume "extraction done" == "ready for Step A" for RoboVQA the way it now genuinely does for OmniVideo-100K.

### SenseNova-SI-8M license — traced to primary source, real per-dataset breakdown across 22 named upstream datasets (not a blanket verdict either way)

Huu challenged the earlier "not permissive" claim directly in the team chat, asking for the actual basis and citing his own test (permissive if: self-created by the paper's authors, pre-1926, government source, or a real CC-BY-class license) — correctly pointed out that "silence in the README" (what was found earlier) is weaker evidence than an explicit disclaimer (which is what MINT-1T actually had). User asked to read the PDF directly to find it. **No PDF text-extraction tool existed in any project env** (`pdftotext`, `pypdf`, `PyMuPDF`, `pdfplumber` all absent; the `Read` tool's PDF support needs `pdftoppm`/poppler-utils, also absent) — asked the user via `AskUserQuestion` rather than either writing a risky hand-rolled PDF parser (accuracy-critical here, wrong extraction could misrepresent the paper to Huu) or guessing; user approved `pip install pypdf` into `env_tools`.

**Extracted all 39 pages, found the real citable answer** in Section 3.2 "Data Sources" + Appendix B.1.2 "Dataset-specific Processing": the 8.16M/8.5M QA pairs are built from **22 named upstream datasets** across 3 groups — "General QA" (VSR, SPEC, GQA, VQA, IconQA, 0.6M pairs), "Community Datasets on Spatial Intelligence" (Open3D-VQA, CLEVR-series, REL3D, SAT, GRiD-3D, MultiSpa, MindCube, ViCA, VLM-3R, VSI-590K, 3.3M pairs), and "Further Scaling" (MessyTable, ScanNet, ScanNet++, SUN RGB-D, CA-1M, Ego-Exo4D, Matterport3D, 4.5M pairs, generated by re-projecting these datasets' existing 3D point-clouds/poses onto 2D images — i.e. these are literally reused source images, not new photography).

**Checked real license terms per dataset (WebSearch, official terms-of-use pages, not just tags) — confirms Huu's "some yes, some no" framing exactly, not a blanket verdict:**
- **Genuinely permissive (confirmed CC-BY-4.0 or equivalent):** GQA, VQA, VSR, CLEVR-series, MindCube (MIT).
- **Confirmed non-commercial/gated/no-redistribution (checked official terms directly):** IconQA (CC-BY-NC-SA-4.0), MultiSpa (CC-BY-NC-4.0), ScanNet (gated, non-commercial), ScanNet++ (gated, "commercial use is strictly prohibited"), Matterport3D ("non-commercial academic purposes only," must attach the agreement to any published derivative), CA-1M (CC-BY-NC-ND), Ego-Exo4D (gated custom license), and — checked separately since it recurs as an upstream source for several of the "Community" datasets — **ARKitScenes (CC-BY-NC-SA-4.0)**.
- **"Nested derivative" risk — real pattern worth flagging generally, not just for SenseNova:** VSI-590K (tagged Apache-2.0), ViCA-322K, and VLM-3R are themselves built from ScanNet/ARKitScenes/Matterport3D-derived imagery, but published under their own (permissive-looking or unstated) license — a fresh permissive tag on a derivative work doesn't retroactively clear the original non-commercial source's terms.
- **Unresolved (no license found via search, would need to check each GitHub repo's LICENSE file directly):** SPEC, Open3D-VQA, REL3D, SAT (dataset itself — its *model* is MIT, which is a different asset), GRiD-3D, SUN RGB-D, MessyTable.

**Net: SenseNova-SI-8M's blanket `apache-2.0` HF tag does not hold for the majority of its images** — the largest contributing group (4.5M/8.5M, "Further Scaling") is confirmed non-commercial/gated at the source. A minority (~0.6M, "General QA") is genuinely permissive. This is a nuanced, per-source verdict, not a simple yes/no — reported to the user in full breakdown form for relaying to Huu.

### MINT-1T-HTML — re-verified from the actual paper (arXiv 2406.11271), not just the HF README, per user request; confirms the earlier image-drop decision was correct, now with stronger primary-source evidence

Downloaded and extracted the real MINT-1T paper (not in `documents/` yet, found via WebSearch, downloaded to scratchpad). Its own **"Datasheet for Datasets" self-disclosure** (a standard ML-paper questionnaire) states explicitly: *"(a) If your work uses existing assets... did you cite the creators? [N/A] (b) Did you mention the license of the assets? [N/A]"* — the authors themselves flag that they did not document licenses or cite creators for the scraped assets. Also: *"There are no restrictions regarding downloading images from these external urls"* — conflates technical accessibility (URLs aren't blocked) with legal permission, the same conflation flagged in the original README-based investigation, now confirmed as the paper's own framing too. Consent section confirms no affirmative consent process, only passive `robots.txt` opt-out. **This is stronger evidence than the README disclaimer used in the original decision (§21) — the earlier decision to drop MINT's images while keeping text stands, now with primary-source backing.**

### Status at end of session

`tok_mint1t` (14118931) still running (~3h). RoboVQA frame extraction still running in background on the login node (67/184 shards). Neither is blocking anything else. OmniVideo-100K is genuinely Step-A-ready (video extracted, captions mapped) pending a JUPITER submission the user hasn't greenlit yet. RoboVQA's video track has an unresolved architecture question (sparse-frame vs. continuous-video treatment) that needs a decision before any Step A work makes sense for it. SenseNova-SI-8M's per-source license breakdown is ready to relay to Huu but the team hasn't yet decided what to do about the ~4.5M non-commercial-sourced images (drop them like MINT, keep only the ~0.6M genuinely-permissive-sourced subset, or something else) — next session should check whether that conversation happened.

---

## 24. OmniVideo-100K Step A Driver Written and Run on JUPITER; Two Real `env_stable_vla` Seed2 Bugs Found and Fixed; One Self-Inflicted Disk-Quota Bug Found, Fixed, and Full-Scale Job Resubmitted (Jul 18, 2026, late night)

### New driver — `data_prep/omnivideo_100k/step_a_tokenize_video.py`

Per user instruction ("reuse the old pipeline but don't write new code into it — write it into `data_prep/omnivideo_100k`"), wrote a new driver that imports the 3 low-level tokenizer classes (`Seed2Tokenizer`/`CosmosVideoTokenizer`/`AVCLMTokenizer`) directly from `/e/project1/reformo/nguyen38/prototype/pipeline.py` — the real runtime copy with actual checkpoint weights (the git-tracked `pipeline_video/pipeline.py` has weights gitignored). Required `sys.path.insert` + `os.chdir(PROTOTYPE_DIR)` before the import: the 3 classes load checkpoints via CWD-relative paths (`"./seed2"`, `"pretrained_ckpts/..."`), and `pipeline.py`'s own `import cosmos_tokenizer` (a local, non-pip package) only resolves if `sys.path[0]` points at `prototype/`.

All new logic — video listing/sharding (`video_list[RANK::WORLD_SIZE]`, simpler than FineVideo's `dataset.shard()` since OmniVideo-100K is just a flat file list), chunking, caption/speech injection, output writing — lives entirely in the new file, none of it touches `pipeline.py`.

**Caption/speech anchor design decision, made explicit before coding:** `omnivideo_100k_segment_captions.jsonl` has ~9.1 segments/video averaging ~11-12s each, each with a rich caption (300-500 words) and speech transcript. An 8-frame/30fps chunk is ~0.267s, so a literal "insert at every chunk overlapping the segment" reading of `JUPITER_STEP_A_TASK.md`'s spec would repeat the same 300-500-word paragraph ~40 times per segment. Flagged this to the user via `AskUserQuestion`; confirmed choice — insert once, at the first chunk whose start falls at/after the segment's start (`build_segment_anchors()`), mirroring the anchor-point pattern already used for FineVideo speech injection in `phase6_merge_adaptive.py` (ASR segment start snapped to one chunk, not repeated across the segment's duration).

### Two real `env_stable_vla` seed2 bugs found and fixed (affect FineVideo too, not just OmniVideo-100K)

First pilot (`970063`, 2 nodes×4 GPU, 48 videos) ran clean but produced `seed2=0` for every video, no crash. Root-caused to `transformers` in `env_stable_vla` having drifted to `4.57.6` (the seed2 checkpoint's own `config.json` declares `transformers_version: 4.52.4` — a large version gap from whenever this env last worked, likely from an unrelated later `pip install` pulling in a newer `transformers` as a dependency):

- **Bug 1 (import path):** `apply_chunking_to_forward`/`find_pruneable_heads_and_indices`/`prune_linear_layer` moved from `transformers.modeling_utils` to `transformers.pytorch_utils` in 4.57.6. The vendored `seed2/seed2_tokenizer.py` (hand-copied ~2021-era BERT modeling code) still imports all 3 from the old location — `ImportError`, silently swallowed by `pipeline.py`'s `except Exception as e: print(...)` (message only, no traceback).
- **Bug 2 (deeper — `tie_weights()` behavior, only surfaced after fixing bug 1):** `AttributeError: 'NoneType' object has no attribute 'predictions'`. Got the full traceback via a standalone diagnostic script (`debug_seed2_load.py`, run as a 1-GPU job, `970070`) since `pipeline.py`'s bare exception handler only prints the message. Traceback: `PreTrainedModel.from_pretrained()` → `tie_weights()` → `tie_embeddings_and_encoder_decoder()` → `get_output_embeddings()` → `self.cls.predictions.decoder`, with `self.cls is None`. Found the real cause by grepping `seed2_tokenizer.py`: line 2601 deliberately sets `self.Qformer.cls = None` (the Qformer's encode-only path never needs the MLM head — that head exists on this shared BERT-derived class only for the separate `.decode()` path). Older `transformers` tolerated a `None` return from `get_output_embeddings()` here; `4.57.6`'s `from_pretrained()` now calls `tie_weights()` unconditionally on every submodule and crashes on the intentional `None`.

**Fix, confined entirely to `step_a_tokenize_video.py`** (per user's constraint — never touched `seed2_tokenizer.py`, `pipeline.py`, or the shared env): (1) reassign the 3 relocated functions onto `transformers.modeling_utils` before importing `pipeline`; (2) import `seed2_tokenizer` early and monkeypatch `get_output_embeddings`/`set_output_embeddings` on `BertLMHeadModel`/`BertForMaskedLM` to return `None` safely when `self.cls is None`, restoring the original author's intent instead of crashing. Verified incrementally: `debug_seed2_load.py` → `LOADED OK` (job `970072`), then a real pilot (`970073`) — 48/48 videos, seed2 producing 2000-5700 real tokens/video, 9m23s.

### Self-inflicted disk-quota bug — found at full-scale, fixed, re-verified, resubmitted

With both seed2 bugs fixed, submitted the first full-scale job (`970087`, 8 nodes×4 GPU=32 GPU, all 5,214 videos) — throughput sizing based on the pilot's measured ~93.8s/video/GPU, scaled down from FineVideo's 40-node/160-GPU allocation since this dataset is ~1/8 the size. Almost every video failed with `[Errno 122] Disk quota exceeded`.

**Root cause — in the new driver itself, not `env_stable_vla` or `pipeline.py`:** the first version of `extract_30fps_frames()` dumped an entire video's frames (up to 5,400 PNGs for a 180s video, at native unscaled resolution) to a temp directory before processing any of them. With 32 ranks running concurrently, several were observed holding 1-2.7GB of temp PNGs simultaneously — enough to exceed the user's per-directory disk quota. The 8-rank pilot never hit this because its smaller concurrent footprint stayed under the same limit; the bug was scale-dependent and invisible until full-scale.

**Fix:** rewrote frame extraction to stream one 8-frame chunk at a time (`extract_chunk_frames()`, one `ffmpeg -ss/-t` call per chunk instead of one call for the whole video — mirroring the pattern `AVCLMTokenizer.encode_mp4_segment()` already uses), with `-vf scale=512:512` (matches `Seed2Tokenizer`'s own `target_size` default, so no quality loss there; `CosmosVideoTokenizer` downsamples to 160 regardless of input size, so no loss there either). This bounds the on-disk working set to ~8 small frames per rank at any instant, independent of video length. `tokenize_video()` was restructured to derive chunk count from the segment-captions JSONL's `duration` field (per `JUPITER_STEP_A_TASK.md`'s own suggestion — no ffprobe dependency needed) rather than from a pre-loaded frame list, and seed2's 1fps sampling is picked directly from whichever chunk's already-extracted buffer contains that global frame index (no separate per-second ffmpeg call needed).

Before re-running, cancelled `970087` (`scancel`) and deleted ~40GB of leftover temp PNGs. Re-verified with a fresh pilot (`970095`, same 48 videos): 0 quota errors, per-rank temp footprint now 1-2.5MB (was 1-2.7GB), and — incidentally — **faster** than the original whole-video-extraction pilot (7m22s vs. 9m23s), so the fix cost nothing in throughput.

**Resubmitted full-scale as `970099`** (same 8×4=32 GPU sizing, `--time=05:00:00`) — running as of this entry. Resume is global (scans all existing `step_a_rank_*.jsonl` files before processing, skips already-done `video_id`s), so re-running the same sbatch after any timeout/crash is safe.

### State at end of session

`970099` is running. Once complete: tokenize with `tokenizer_vla_qwen3` (257,901 vocab) — the JUWELS session's own hard lesson (§23) applies here too, do not use `tokenizer_vla_adaptive_v2`. Blend ratio with FineVideo-VLA/MV-Omni is a training-time decision, not scoped here.

### Confirmed complete (Jul 19, 2026)

`sacct -j 970099` — `COMPLETED`, exit `0:0`, ran 2026-07-18T19:30:45→22:01:21 (2h30m). Output verified directly, not just from SLURM state:

| Check | Result |
|---|---|
| Output files | 32/32 `step_a_rank_*.jsonl`, 39GB total |
| Videos processed | **5,214/5,214 lines** — exact match to the full OmniVideo-100K video count |
| Error log (`970099_omni100k_stepA_full_err.log`) | Only harmless deprecation warnings (`GenerationMixin`, `torch_dtype`) — no Traceback/Exception |
| Seed2 bug regression check | Sampled `rank_0` (163 videos): **0 videos with seed2=0** — the two `env_stable_vla` fixes (§24 above) held at full scale |
| Content sanity | `rank_0` sample: 546,912 seed2 tokens / 12,809,800 cosmos tokens / 317,687,832 avclm tokens / 1,511 caption blocks / 1,468 speech blocks across 163 videos — all four expected token types present with plausible volumes |

**Step A (the GPU-dependent stage, JUPITER-only) is done for OmniVideo-100K.** Megatron-side tokenization with `tokenizer_vla_qwen3` will be run from the JUWELS side, out of scope for this task.

---

## 25. OmniVideo-100K Video Track Flattened + Tokenized (456.5M Real Tokens); Token-Count Gap vs FineVideo-v5 Explained (Document-Count, Not Density); Content-Type Survey Corrects Earlier Wrong Claim (24.1% Real Sports Content Found); Pose-Pipeline Pilot Scoped and Handed Off to JUPITER (Jul 19, 2026, morning/midday)

**Main work:** With Step A confirmed complete (§24), finished the rest of the video track: wrote and ran `flatten_step_a_video.py` (raw Step A token stream → Megatron-ready JSONL), then a real Megatron tokenize job. Cross-checked the resulting token count against FineVideo-v5 to explain an apparent "too few tokens" discrepancy the user flagged — turned out to be a document-count artifact, not a real problem. Surveyed the full 5,214-video corpus by content type after the user pushed back on an earlier too-hasty "no physical activity in this dataset" claim — found a real 24.1% sports/physical-activity subset. Scoped a pose-pipeline pilot on that subset and wrote a handoff task doc for the JUPITER side, deliberately excluding hand/finger-keypoint improvement (current pipeline's HRNet config is COCO-17 body-only) and excluding the talking-head subset (framing + low-motion-value reasoning) from the pilot.

### 1. Flatten + Megatron tokenize for the video track — both real, both complete

`data_prep/omnivideo_100k/flatten_step_a_video.py`: parses Step A's raw `<seed2>`/`<cosmos>`/`<avc_lm>` (raw integer IDs, not yet atomic vocab tokens) plus inline `<caption>`/`<speech>` blocks into the atomic `<seed2_N>`/`<cosmos_N>` form the Qwen3 tokenizer registers, using the same drop-rate convention as `pipeline_pose/phase7_flatten.py` (avc_lm payload always dropped, cosmos dropped 50%/chunk, seed2/caption/speech always kept). Real run: input `/p/data1/mmlaion/shared/vla/omnivideo_100k_video_flat/` (32 files, copied from JUPITER's Step A output) → output `omnivideo_100k_video_flattened/` (32 files), **5,214/5,214 lines preserved exactly**, 0 videos lost.

Megatron tokenize job `14120433` (`tok_omni_video`): `sacct` confirms `COMPLETED`, ran 06:09:16→06:26:45 (17m29s). Output: `/p/data1/mmlaion/shared/vla/tokenized_output/omnivideo_100k_video/data_shard_00000.bin` (1.83GB) + `.idx`. No sbatch script was found saved in-repo for this job (likely run as a direct command, not committed) — worth reconstructing via `sacct -j 14120433 --format=SubmitLine` if this needs to be repeated. Verified the real token count by parsing the `.idx` header directly (Megatron `MMIDIDX` mmap format: magic + version + 1-byte dtype code `4` = `np.int32` + uint64 sequence count `5214`, matching the video count exactly) rather than trusting any printed job summary: **456,487,128 tokens (456.5M)**.

### 2. Token-count "discrepancy" investigated at the user's request — resolved, not a bug

User noticed the 456.5M figure looked small next to FineVideo-v5's 10.55B and asked why. Real comparison:

| | FineVideo-VLA v5 | OmniVideo-100K video |
|---|---|---|
| Total tokens | 10,554,076,391 | 456,487,128 |
| Documents | 371,888 | 5,214 |
| Tokens/document | ~28,375 | ~87,556 |

OmniVideo-100K actually has **~3.1x more tokens per document** than FineVideo — the ~23x gap in totals is fully explained by the ~71x gap in document count, which is a structural artifact: FineVideo splits each source video into multiple scene/activity records (371,888 records from ~40K raw videos, ~9.3 records/video), while OmniVideo-100K's Step A driver was deliberately designed as 1 video = 1 document (§24, no scenes/activities structure to split on). Not a flatten or tokenize bug.

### 3. Content-type survey — corrected an earlier wrong claim about the dataset

Initial assessment in this session ("OmniVideo-100K content is news/cartoons/challenges, no real physical activity") was **wrong** and was corrected after the user pushed back, recalling the dataset had "quite a lot of people" in it. Re-investigated properly: wrote a keyword classifier over the `video_summary` field (video-level synopsis in `omnivideo_100k_segment_captions.jsonl`) across the full 5,214-video corpus (not just a small manual sample):

| Category | Count | % |
|---|---|---|
| Sports/physical activity (basketball/soccer/dance/boxing/gym/wrestling/tennis/etc.) | 1,256 | 24.1% |
| News/talking-head | 1,210 | 23.2% |
| Cartoon/animation | 325 | 6.2% |
| Gambling/slot machine | 129 | 2.5% |
| Gaming/gameplay | 115 | 2.2% |
| Vlog/travel | 79 | 1.5% |
| Other/misc | 2,503 | 48.0% |

This is a coarse text-summary heuristic, not a verified visual check — flagged explicitly as such, with segment-level `caption` fields available as a finer-grained fallback if the video-level heuristic proves too noisy in the pilot.

### 4. Pose-pipeline pilot scoped for the sports subset — with an explicit, user-confirmed limitation

Checked `pipeline_pose/phase1_hrnet_gpu.py`'s actual model config before making any claims about what running the pose pipeline on new video would achieve: it uses `td-hm_hrnet-w48_8xb32-210e_coco-256x192`, i.e. **COCO-17, body-only keypoints** (most distal upper-limb point is the wrist; no finger/hand keypoints at all), mapped to H36M-17. Flagged to the user that no choice of source video — sports or otherwise — would improve hand/finger detail under the current pipeline; that requires a different pose model (e.g. COCO-WholeBody-133) as a separate future effort. **User confirmed this is acceptable** — hand data is expected to come from a separate dataset/effort later; this pilot's goal is just adding arm/body motion diversity beyond FineVideo's largely lifestyle/vlog motion profile.

Talking-head content (23.2% of the corpus, second-largest category after sports) was deliberately excluded from the pilot despite clearly containing people, for two reasons laid out to the user: (1) typical medium/close-up framing likely puts hip/knee/ankle keypoints outside the frame, so `coco_to_h36m()`'s confidence-threshold zero-fill (`CONF_THRESHOLD = 0.5`, phase1_hrnet_gpu.py:31) would degenerate much of the lower-body skeleton; (2) near-static standing/sitting motion has low training value and likely overlaps with motion profiles FineVideo already covers well. Explicitly caveated as unverified reasoning from text summaries, not confirmed against real extracted frames (OmniVideo-100K's mp4s live on JUPITER's `/e`, not mounted from this JUWELS session) — offered to let the JUPITER-side pilot report per-category yield stats to settle it empirically if the user wants.

### 5. Deliverables written and committed (`95f2927`)

- `data_prep/omnivideo_100k/select_sports_subset.py` — the classifier above; real run output 1,256/5,214 (24.1%).
- `data_prep/omnivideo_100k/sports_subset_video_ids.txt` — the real resulting video_id list (one per line, no extension).
- `data_prep/omnivideo_100k/JUPITER_POSE_PILOT_TASK.md` — handoff task for JUPITER's Claude instance. Documents: why only the sports subset (not all 5,214); what data already exists on JUPITER vs. needs transfer (video mp4s already moved there for Step A; only the new ID-list file needs to arrive, via `git pull` or manual copy); the hand-keypoint limitation (§4) already resolved with the user, out of scope; and — most importantly — flags that `phase1_hrnet_gpu.py` is hard-coded to read from the FineVideo HF arrow dataset (`load_from_disk` + `cached_video_ids.json`) exactly the way `pipeline_video/pipeline.py` was for Step A (§24), so a new driver is needed (reuse the model-loading + `coco_to_h36m()` logic verbatim, replace the input path with direct `cv2.VideoCapture` on `$DATA/omnivideo_100k/videos/{video_id}.mp4`). Also flags that Phase 6 (`phase6_merge_adaptive.py`) almost certainly needs an equivalent rewrite (it expects FineVideo's `scenes[].activities[]` structure), and recommends the same staged-pilot discipline that caught 3 real bugs in Step A (§24) — a tiny ~20-30-video pilot before committing to the full 1,256.

### State at end of session

OmniVideo-100K's video track is now fully tokenized and training-ready (456.5M real tokens, verified via `.idx` header). The pose-pipeline pilot is scoped and handed off as a task doc but **not yet run** — waiting on the JUPITER side to pull the repo and execute per `JUPITER_POSE_PILOT_TASK.md`. Blend ratio with FineVideo-VLA for training is still an open, training-time decision.

---

## 26. Pose-Pipeline Pilot Executed on JUPITER: Two Infra Regressions Found and Fixed (broken `outputs/` symlink, stale env paths), New Phase 1 Driver Preserves Continuous Confidence, 24-Video Pilot Clean (Jul 19, 2026, afternoon)

**Main work:** Pulled `JUPITER_POSE_PILOT_TASK.md` (§25 handoff) on the JUPITER side and executed it. Before writing the new driver, verified — per explicit user instruction not to discard confidence scores wherever Phase 1/2 store them — that `phase1_hrnet_gpu.py`'s `coco_to_h36m()` actually **binarizes** HRNet/detector confidence to `1.0`/`0.0` at `CONF_THRESHOLD`, discarding the real continuous score that MotionBERT's `infer_wild.py` (via `WildDetDataset`) reads as a model input feature; confirmed separately that Phase 2's own output (`X3D.npy`) carries no confidence channel at all, so there is nothing to preserve on that side. Wrote `data_prep/omnivideo_100k/phase1_hrnet_omnivideo.py`, a new Phase 1 driver that fixes this (keeps the real float score) while reading OmniVideo-100K's flat mp4s directly instead of the FineVideo HF arrow dataset. While preparing to run it, found and fixed two real infrastructure regressions unrelated to any of this session's code, then ran a smoke test and a real 24-video SLURM pilot — both clean.

### 1. Two infra regressions found before any GPU job was submitted

- **`outputs/` symlink was gone.** CLAUDE.md documents `3d-human-pose/outputs/` as a symlink to `/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs/` (145GB+ of real Phase 1-6 data: `2d_json/`, `3d_npy/`, `states_jsonl/`, etc.). In this checkout it had become a plain, nearly-empty local directory (only `fps_lookup.json`, no symlink) — `git status`/`ls -la` showed a real directory, not an `l`-mode entry. Any Phase 1-6 script run with `outputs/2d_json` (a CWD-relative path, as all of them use) from `/e/project1/.../3d-human-pose/` as CWD would silently read/write the wrong, disconnected location instead of the real 145GB+ corpus. Root cause not established (no evidence in prior session logs of an intentional change) — likely fallout from a filesystem/quota event, not something introduced this session. **Fixed:** moved the stray local directory to `outputs_local_backup/` (preserving `fps_lookup.json`, not deleted), recreated `outputs` as a real symlink to `/e/data1/datasets/playground/mmlaion/shared/nguyen38/outputs/`, verified it resolves (`outputs/2d_json/` lists real FineVideo `*_2d.json` files).
- **`env_hrnet_datasets_v1`/`env_motion_final` conda envs were not where `setup_hrnet_gpu.sh`/`setup_motionbert.sh` expected.** Both scripts `conda activate /e/project1/reformo/nguyen38/3d-human-pose/env_{hrnet_datasets_v1,motion_final}` — neither directory exists there anymore (confirmed via `conda env list` from that base, not just `ls`). Found both envs, fully intact (real `python3.9` binaries, correct `mmpose`/`mmdet`/`torch` installs), at `/e/data1/datasets/playground/mmlaion/shared/nguyen38/3d-human-pose/env_{hrnet_datasets_v1,motion_final}/` instead — same data1 mount as the `outputs/` fix above, suggesting a related migration that moved heavy artifacts off `/e/project1` (lighter quota) without updating the two setup scripts that reference them by absolute path. **Fixed:** updated both scripts' `conda activate` lines to the correct data1 path. Verified both activate cleanly with real GPU access (`torch.cuda.is_available() == True`, `NVIDIA GH200 480GB`), and that `env_hrnet_datasets_v1` imports `mmpose==1.3.2`/`mmdet==3.3.0` successfully.

Flagged both to the user before touching anything (via `AskUserQuestion`), since a repo-wide symlink/env-path fix has blast radius beyond this one task; user approved the direct fix.

### 2. New driver: `data_prep/omnivideo_100k/phase1_hrnet_omnivideo.py`

Per `JUPITER_POSE_PILOT_TASK.md` §3: does not modify `pipeline_pose/phase1_hrnet_gpu.py`. Reuses the model-agnostic parts verbatim (HRNet/Faster-RCNN config+checkpoint paths, `init_pose_model`/`init_detector` calls, the COCO→H36M joint mapping structure) but replaces the FineVideo-arrow input path with direct `cv2.VideoCapture` on `$DATA/omnivideo_100k/videos/{video_id}.mp4`, and shards via `video_ids[RANK::WORLD_SIZE]` (matching `step_a_tokenize_video.py`'s convention, §24) over `sports_subset_video_ids.txt` rather than the original's more complex `--offset/--total_workers` scheme (sized for FineVideo's 200-worker/40K-video run; unnecessary at this dataset's 1,256-video scale). Output format is unchanged (`{"frame_id", "keypoints": [[x,y,conf]×17]}` per file) and written to the same `outputs/2d_json/` directory FineVideo uses, so Phase 2 (`phase2_motionbert_gpu.py`) can consume it with zero changes — safe because the two corpora's video IDs don't collide.

**Confidence fix (§ intro):** `coco_to_h36m()` was rewritten so `get_pt()` still zeroes the `(x, y)` position below `CONF_THRESHOLD` (unchanged behavior — avoids feeding MotionBERT garbage coordinates for undetected joints) but now returns the real float confidence in all cases instead of collapsing it to `1.0`/`0.0`. Derived joints (pelvis, neck, spine, head-top — the 4 points synthesized from pairs of raw COCO keypoints) now store `min()` of their two contributing raw confidences rather than a hardcoded `1.0`, and their presence-gating condition was changed from the original's `> 0` (which relied on the old binarization) to an explicit `>= CONF_THRESHOLD` to preserve the same gating semantics now that raw scores can be small-but-nonzero below threshold.

### 3. Smoke test (1 video, interactive) then real SLURM pilot (24 videos) — both clean

Before spending any SLURM queue time, ran the new driver directly in an interactive shell on one video (`iGVvChGEQdM`, first entry in the sports subset) — completed in a few minutes, 2,564 frames, no errors. Verified the confidence fix worked as intended by inspecting the output JSON directly: 915 distinct non-binary confidence values across the file (e.g. `0.004`, `0.008`, `0.012`, ...), not just `{0.0, 1.0}`.

Submitted `data_prep/omnivideo_100k/submit_phase1_pilot.sbatch` (1 node × 4 GPU, first 24 videos of `sports_subset_video_ids.txt`, 6/rank) as job `976467`. `sacct` confirms `COMPLETED`, exit `0:0`, 26m18s. Output written to a pilot-only directory (`$DATA/omnivideo_100k/pose_2d_json_pilot/`, deliberately kept separate from the shared `outputs/2d_json/` production path until quality is confirmed, rather than defaulting straight into it).

Aggregate quality check across all 24 output files:

| Metric | Result |
|---|---|
| Total frames | 60,506 |
| Frames with a detected person (any non-zero keypoint) | 47,639 (**78.7%**) |
| Videos ≥80% frame-level detection | 16/24 |
| Videos <20% frame-level detection | 2/24 (`28jYYH6WrA0`: 5.1%, `dXv4oInXqiE`: 17.6%) |

The two low-yield outliers are consistent with the `select_sports_subset.py` heuristic's known false-positive risk (§25 §3 — text-summary keyword match, not a visual check); not investigated further this session. 78.7% frame-level detection is well above the 24-41% *joint-level* skeleton coverage figure §2.2 reports for FineVideo (different metric, not a direct comparison, but a reasonable directional signal that the sports subset is yielding real person-motion content as intended).

### State at end of session

Per explicit user instruction, **stopped here for review rather than proceeding to the full 1,256-video Phase 1 run** — the 24-video pilot is clean and the aggregate numbers look good, but the user wants to look at the results before committing the larger GPU allocation. Phase 2-6 (§ `JUPITER_POSE_PILOT_TASK.md` §3: MotionBERT likely reusable as-is, Phase 6 merge almost certainly needs a new driver mirroring `flatten_step_a_video.py`) not yet started.

---

## 27. Pose-Pipeline Phase 1: Self-Review Found and Fixed a Real Regression, External-Review Points Triaged Against Real Data, Native-fps Convention Confirmed, Full-Scale 1,126-Video Run Submitted (Jul 19, 2026, afternoon/evening)

**Main work:** Before scaling from the 24-video pilot to the full 1,126-video (post-filter) run, did a deliberate self-review pass on the code and infra changes made earlier in the session. Found and fixed a real regression introduced by the earlier `outputs/` symlink fix. Hardened `phase1_hrnet_omnivideo.py` against two failure modes (silent "success" on an unopenable video, `VideoCapture` fd leak on a mid-loop exception), verified both fixes with a real 8-video SLURM pilot plus a synthetic corrupted-file test. The user then brought an independent (external, ChatGPT-style) code review of the same script; went through each of its 10 points against real measured data from this session rather than accepting or dismissing them on priors — 2 were genuine, actionable gaps; several others were either already-verified non-issues or inherited behavior from `phase1_hrnet_gpu.py` (unchanged from a script that already processed 40,804 real FineVideo videos without incident), not new defects. Separately answered a design question about fps handling — confirmed the existing native-fps-until-Phase-2.5 convention should carry over unchanged for OmniVideo-100K, but flagged a real prerequisite gap (`fps_lookup.json` has zero OmniVideo-100K entries) that would silently drop the whole dataset at Phase 2.5 if left unaddressed. Applied 4 small hardening fixes, verified them, then submitted the full-scale Phase 1 run.

### 1. Self-review caught a real regression before it could cause silent data loss

Re-reading the `outputs/` symlink fix (§26) turned up a genuine bug it introduced: `outputs/fps_lookup.json` (43,751 entries, consumed by `phase2_5_resample_30fps.py` and `phase3_kinematics_processor.py`) turned out to **only exist in the stray local directory** that got renamed to `outputs_local_backup/` — it was never present in the real `/e/data1/.../nguyen38/outputs/` target. So the "cleanup" in §26 had, as a side effect, silently made `fps_lookup.json` unreachable from the new symlink (`outputs/fps_lookup.json` → 404), which would have made `phase2_5_resample_30fps.py` fall back to treating every video's native fps as unknown the next time anyone ran it. Fixed immediately by copying `outputs_local_backup/fps_lookup.json` into the real data1 `outputs/` directory; verified byte-identical (43,751 entries both sides).

### 2. Hardened `phase1_hrnet_omnivideo.py` against 2 failure modes, verified with real tests

- **Silent "success" on an unopenable video:** `cv2.VideoCapture` failing to open (or a stream that reads 0 frames) previously still wrote a valid-but-empty 2D JSON and counted as `done` — the resume check (`file exists → skip`) would then permanently skip that video on any future rerun, with no error trail. Now checks `cap.isOpened()` and raises if 0 frames are read, landing the video in the `error` bucket instead.
- **`VideoCapture` fd leak on exception:** `cap.release()` previously ran only after the frame loop completed normally; an exception mid-loop (e.g. an OOM on one frame) left `cap` unreleased until GC. Wrapped the whole loop in `try/finally` so release always happens — relevant at the ~280-videos-per-rank scale of a full run, where several bad videos in a row could otherwise exhaust file descriptors and turn isolated per-video failures into a whole-rank crash.

Verified both with real tests rather than just code inspection: a fresh 8-video SLURM pilot (job `976556`, 8 videos never processed before, `sports_subset_video_ids_filtered.txt` indices 24-31) — `COMPLETED`, 0 errors, 13m25s, **87.7% frame-level person detection** (higher than the original 24-video pilot's 78.7%, consistent with the animation filter — see §3 below — having removed the worst-yield videos from the pool); and a synthetic corrupted-mp4 test (a plain text file renamed `.mp4`) confirming the new error path lands in `error` with zero leftover `.tmp` files, instead of the old silent "OK, 0 frames" behavior.

### 3. Discovered while investigating: 130/1,256 sports-subset videos were animated content, not sports

Investigating the 24-video pilot's 2 lowest-yield videos (`28jYYH6WrA0` 5.1%, `dXv4oInXqiE` 17.6%) found both were animated/cartoon content that had slipped through `select_sports_subset.py`'s generic keyword match — `dancing` matched a review video's aside about "a clip of a dancing Robot", `running` matched cartoon characters "running away" in a fairy-tale short. A more concerning case surfaced in the same pilot: `Ncl93lkMpJM`, an animated dinosaur music video per its own `video_summary`, still scored **56.3%** frame-level HRNet detection — Faster-RCNN/HRNet can false-positive on stylized anthropomorphic characters, so this isn't just a missing-data case, it's a case where MotionBERT (trained exclusively on real Human3.6M humans) would silently lift a real-human-confidence-scored but real-human-proportion-wrong 2D pose into likely-wrong 3D output.

Wrote `data_prep/omnivideo_100k/filter_animation_content.py`: excludes any video whose `video_summary` contains `animat(ed|ion)|cartoon|anime|CGI|computer-generated|claymation|stop-motion`. Real run: 130/1,256 excluded, **1,126 remain** in `sports_subset_video_ids_filtered.txt`. Explicitly scoped as narrow/incomplete: character-driven animated content that never says "animated" in its own summary (e.g. `28jYYH6WrA0`'s fairy-tale characters, referred to only by invented names) won't match this text heuristic — real per-video frame-detection stats from Phase 1 remain the backstop for whatever it misses. `phase1_hrnet_omnivideo.py`'s default `--video-ids-file` now points at the filtered list.

### 4. External code review triaged against real session data — 2 genuine gaps, several overstated or already-disproven claims

The user brought an independent review (ChatGPT-style, 10 numbered points) of `phase1_hrnet_omnivideo.py` for a second opinion before committing to full-scale. Rather than accepting or refuting points on priors, checked each one against data already gathered this session:

**Genuine, actionable:**
- *Identity switching* (picking the largest bounding box independently per frame can jump between different people in crowded sports footage) — real limitation, but **inherited from `phase1_hrnet_gpu.py` unchanged** (the same logic already ran across 40,804 real FineVideo videos), not a regression. Partially mitigated downstream by Phase 3's anti-teleportation filter and Phase 4's YOLO cleaner (a sudden identity jump often looks like a position teleport). A proper fix (IoU-based tracking across frames) is a real architecture change beyond this task's "adapt I/O only, keep model logic verbatim" scope — logged as a known limitation rather than fixed now.
- *Mid-decode truncation not detected* — real gap: the earlier fix (§ above) only caught total failure (0 frames), not a stream that decodes some-but-not-all frames. Naively implementing the review's exact suggestion (compare decoded frame count against `duration × 30fps`, error below 90%) was tested directly against the 8 real pilot videos from §2 and would have produced **2 false-positive errors out of 8 (25%)** — `07WqS-ccIrw` and `0OxHEDu5dFE` decoded 100% of their real frames but only hit 83.1% of the duration-based estimate, because (verified via `cv2.CAP_PROP_FPS`) both are native **25fps**, not 30fps, while `0GPO9qLraB8`/`iGVvChGEQdM` are native 30fps — OmniVideo-100K has mixed native fps (see §5). Implemented instead as a **soft warning** (not a hard failure) comparing against the video's own `cv2.CAP_PROP_FRAME_COUNT` (the container's real reported count, not a duration-derived guess) at <90%.

**Already verified as non-issues, or inherited/out-of-scope:**
- *SLURM_NTASKS misconfiguration risk* (if launched without `srun`, only rank 0 would run and silently process ~25% of the dataset) — real concern in the abstract, but empirically disproven: `submit_phase1_pilot.sbatch` already uses `srun python -u ...`, and both real pilot jobs' logs show all 4 ranks correctly assigned (`Rank 0/4` .. `Rank 3/4`, 6 videos each = 24 total). No change made.
- *Confidence-continuous-but-position-zeroed "discontinuity" at the 0.5 threshold* — the review's framing has it backwards: MotionBERT's `WildDetDataset`/`infer_wild.py` is designed to consume detections from generic "in the wild" 2D detectors, which normally *do* report continuous confidence — it's the original `phase1_hrnet_gpu.py`'s binarization that's the non-standard departure, not this session's fix toward continuous values. No change made (already correct per the confidence-preservation requirement this driver was built for — §26).
- *mmpose returning torch.Tensor instead of numpy* — legitimate defensive-coding suggestion; the exact code in question is copied verbatim from `phase1_hrnet_gpu.py` (already run cleanly on 40,804 videos) and was directly exercised without incident across 32 real videos this session on the pinned `mmpose==1.3.2` env. Added the defensive conversion anyway (§6) since it's free and removes a version dependency, but it wasn't fixing an observed bug.
- *`init_default_scope()` called every frame (perf overhead)* — real overhead, but inherited unchanged from the original, and measured throughput (263s/video/GPU) already sizes a full run at ~2.6h/32 GPU, an acceptable cost. Not changed — the review itself acknowledged this needs a scope-conflict-safety check before removing it, which wasn't performed.
- *Entire video's frame data held in RAM until `json.dump()`* — reviewer's estimate ("hundreds of MB") checked against real duration data for the filtered 1,126-video set: max duration is 180s (5,400 frames @ 30fps), giving a realistic per-video RAM footprint on the order of ~10MB, not hundreds — negligible against a GH200 node's RAM. Not changed.
- *Output-path/model-config paths depend on CWD* — true, but this is the **entire pipeline's established convention** (every Phase 1-7 script uses bare relative paths; every submit script `cd`s into the repo root first, including `submit_phase1_pilot.sbatch`/`submit_phase1_full.sbatch`). Making this one script use absolute `SCRIPT_DIR`-relative paths would make it inconsistent with the rest of the codebase for a risk that's already handled operationally. Not changed.

### 5. FPS design question: confirmed native-fps convention, found and documented a real prerequisite gap

User asked directly whether OmniVideo-100K should be resampled to 30fps now (at Phase 1) or kept at native fps through Phase 2, matching the existing FineVideo convention. Answer: **keep the existing convention** — Phase 1/2 already run at native fps for FineVideo (confirmed in `phase2_5_resample_30fps.py`'s own docstring: "Resample native-fps 3D pose arrays to 30 fps"), `phase1_hrnet_omnivideo.py` already reads frame-by-frame via `cv2.VideoCapture` with no forced resampling (matches this without any change needed), and OmniVideo-100K genuinely has non-uniform native fps across videos (confirmed via `cv2.CAP_PROP_FPS`: `07WqS-ccIrw`/`0OxHEDu5dFE` = 25.000, `0GPO9qLraB8`/`iGVvChGEQdM` = 30.000) — exactly the scenario Phase 2.5 exists to normalize.

**Real gap found and documented (not yet blocking at the current Phase 1 stage):** `outputs/fps_lookup.json` (43,751 entries) is FineVideo-only — zero OmniVideo-100K video_ids are in it. `phase2_5_resample_30fps.py`'s own docstring says videos missing from this file are "skipped with a warning" — run as-is, Phase 2.5 would silently drop the entire OmniVideo-100K pose corpus. Documented as a required prerequisite step in `JUPITER_POSE_PILOT_TASK.md` (run `tools/extract/extract_fps.py` against `$DATA/omnivideo_100k/videos/`, then merge — not overwrite — into the shared `fps_lookup.json`) before Phase 2.5 is ever run for this dataset.

### 6. Final hardening pass and full-scale submission

Applied 4 small fixes to `phase1_hrnet_omnivideo.py` based on the triage in §4 before committing GPU-hours to the full run:
- Dedup `video_ids` after reading the ID list (order-preserving, logs the count removed) — checked the real filtered list first (1,126 lines, 1,126 unique, 0 duplicates currently), so this is prevention rather than a fix for an observed problem.
- `getsize(final_json) > 2` added to the resume check, so a stray empty/near-empty leftover JSON from an older run no longer skips a video forever.
- Defensive `torch.Tensor`→numpy conversion for mmpose's `keypoints`/`keypoint_scores` outputs.
- Soft (non-fatal) warning when decoded frames fall under 90% of `cv2.CAP_PROP_FRAME_COUNT` (§4's revised version of the truncation check).

Verified: syntax check, a direct interactive run with an intentionally duplicated video ID (confirmed the dedup log line fires and the correct de-duplicated count is assigned), and reconfirmed the corrupted-video error path and the normal-completion path both still behave correctly. Full completion of an end-to-end successful video run with the new warning-check line specifically was not separately re-verified before submission (time-constrained per explicit user request to stop iterating and submit) — the change is a simple, low-risk conditional+print with no control-flow impact on the surrounding success path, and the full-scale job's own real output will surface anything wrong immediately given it processes 1,126 videos.

**Submitted:** `data_prep/omnivideo_100k/submit_phase1_full.sbatch` — job `976705`, 8 nodes × 4 GPU (32 GPU), all 1,126 videos in `sports_subset_video_ids_filtered.txt` (no `--limit`), `--time=04:00:00` (measured pilot throughput ~263s/video/GPU → ~2.6h estimated, with safety margin). Confirmed `RUNNING` shortly after submission (`sacct`: state `RUNNING`, all 8 nodes allocated `jpbo-009-[01-08]`), error log clean (only harmless deprecation/scope-switch warnings, no tracebacks). Global resume (`getsize`-checked file-exists per video_id) makes it safe to resubmit unchanged if the job times out or crashes partway.

### State at end of session

Full-scale Phase 1 (job `976705`) running as of this entry. Commits this session: `2f3d675` (pilot + infra fixes), `7dc1ca0` (animation filter), `8e688c4` (silent-success/leak hardening), `2024da4` (English-only comments/logs), `f9eb687` (fps prerequisite doc + final hardening + full-scale submit) — all pushed. Phase 2 onward not started; per `JUPITER_POSE_PILOT_TASK.md` §3, Phase 2 (MotionBERT) is expected to be largely reusable as-is but must be I/O-verified before trusting that assumption, and Phase 2.5 has the `fps_lookup.json` prerequisite from §5 above. Phase 6 (merge into training-ready records) almost certainly needs a new driver, following the pattern of `flatten_step_a_video.py`.

---

## 28. Phase 4 FineVideo Rebalanced + Resubmitted via SLURM; Wrapper-Token Fix Designed and Applied to 3 Datasets (synth-llava, omnivideo-100k-final, FineVideo pending); Self-Correction: FineVideo v5 Caption/Speech Coverage Was Never Missing — Wrong Local Path Checked (Jul 21, 2026)

**Context:** JUPITER maintenance mostly cleared (18/~5,600 `booster` nodes still `maint`, not blocking scheduling). User asked to resubmit Phase 4 FineVideo (YOLO cleaning) via SLURM instead of the login-node tmux workaround used during maintenance.

### 1. Phase 4 SLURM resubmit — uneven-split bug found and fixed

First submit (`slurm/submit_yolo.sh`, job `1004323`, 128 workers/4 GPU/1 node via NVIDIA MPS) inherited progress from the earlier 4-worker login-node run (17,706/40,305 already done, correctly skipped). User observed some workers finishing almost instantly while others stayed slow. Root cause: `phase4_yolo_cleaner.py`'s worker split (`split_chunk_indices()`) is a **contiguous** slice over the *entire* input-dir listing, not the *remaining* one — the previously-done files (from the earlier 4-worker contiguous run) were clustered at the start of each of the old 4 blocks, so only the SLURM workers whose 1/128 slice happened to overlap those clusters got a cheap ride (**48/128 workers finished in <8 min**), leaving the other ~80 workers with close to their full ~315-file share of real work instead of the ~171 they'd get from an even split of just the *remaining* 21,841 videos.

**Fix:** `scancel`'d job `1004323`; built a symlink farm (`outputs/states_jsonl_30fps_remaining_phase4/`, containing only the 21,841 not-yet-done `_states.jsonl` files — NFS symlink creation is slow enough that this needed two backgrounded/resumed passes) and wrote `slurm/submit_yolo_remaining.sh` (identical to `submit_yolo.sh` except `--input-dir` points at the symlink farm). Resubmitted as job **`1004747`** — verified 0 `[SKIP]` across all 128 workers, each assigned exactly 170-171 files. Status at time of writing: **27,056/40,305 (~67%)**, running ~1h, 1 non-fatal soft error logged for 1/40,305 videos (window-resolution failure, script logged and moved on — not a crash).

**Worker logs note for future sessions:** the *old* (uneven) job's logs were moved to `logs/yolo_workers_run1_unbalanced/` before resubmitting — `logs/yolo_workers/` always holds the *current* job's logs only.

### 2. Wrapper-token fix — decided with user, applied to source + 2/3 datasets, third pending Phase 4

Comparing `EmpathicRobotics/emotional-roleplay-finetuning-dataset-flattened` (has `<snac>`/`</snac>` wrapper) against `FineVideo-Phase7-Flattened`/`omnivideo-100k-final` (bare tokens, no wrapper) surfaced a real inconsistency: `<seed2>`, `</seed2>`, `<cosmos>`, `</cosmos>`, `<avc_lm>`, `</avc_lm>`, `<agent>`, `</agent>` are all **already registered as atomic special tokens** in `tools/tokenizer/expand_vocab.py`/`build_tokenizers.py`, but `pipeline_pose/phase7_flatten.py` and `data_prep/omnivideo_100k/phase6_merge_omnivideo.py` (the actual production merge+flatten path for OmniVideo — **not** `flatten_step_a_video.py`, which despite its docstring claiming reuse is not actually invoked by the production chain) strip these wrapper tags uniformly for every payload type (seed2/cosmos/avc_lm/agent/snac), keeping only `<caption>`/`<speech>` — a deliberate, consistent-with-itself convention from early in the project, just inconsistent with the newer, independently-designed SNAC pipeline (`laion_emotional_roleplay/tokenize_snac.py`, "decided after review with Van Khue, session 2026-07-20" per its own docstring).

Decided with user to add the wrapper back everywhere, reasoning: (a) the tokens are already paid for in vocab, currently untrained dead weight; (b) explicit span boundaries give the model an unambiguous "this modality is over" signal decoupled from "what comes next" — a plausible contributing factor to the still-open modality-transition failure (§13's Problem Diagnosis), compounded by per-chunk dropout making "valid next modality" vary example-to-example; (c) matches real multimodal/agent-LLM conventions (explicit span/tool-call wrapper tokens). Cost is small (~1-2% token-count increase).

**Applied to:**
- `pipeline_pose/phase7_flatten.py` (`process_activity_per_chunk()` + `count_token_types()`) — unit-tested with synthetic input before use. **Not yet regenerated against real data** — waiting on Phase 4 (§1) to finish so Phase 5→6→7 can run once, cleanly, with both this fix and the fps-mismatch fix (§24-27's bug) baked in together.
- `data_prep/omnivideo_100k/phase6_merge_omnivideo.py` — regenerated `omnivideo_100k_video_agent_merged/` + `omnivideo_100k_final/` in full; verified counts match the pre-fix run exactly (5,214→5,214, 0 malformed, 799 videos with `<agent>`, 62,631 agent windows, 5,214/5,214 with QA) — confirms the fix only changes token *structure*, not content. Re-uploaded to `EmpathicRobotics/omnivideo-100k-final`.
- `data_prep/omnivideo_100k/step_a/flatten_step_a_video.py` — mirrored for consistency even though it's confirmed not to be the real production path.
- `data_prep/synth_llava/tokenize_seed2.py` — also fixed a **separate, real bug** found by the user: the source `synth_llava`/`synth_llava2` rows ship pre-formatted as `<caption><image_0>caption text</caption>` (Huu's own dataset), and the original in-place placeholder substitution preserved that nesting verbatim, putting seed2 tokens *inside* `<caption>` — wrong per the project's own convention (`<seed2>`/`<caption>` are sibling tags in `step_a_tokenize_video.py`, never nested). Fixed both the source script and (via 2 regex passes over the already-tokenized 603,999-row `synth_llava_flat/` — the second pass fixing a self-inflicted bug in the first, where the last seed2 token lacked a trailing space and got swept into the caption text) the existing data. Re-uploaded to `EmpathicRobotics/synth-llava`.

**Near-miss:** reported "ready to upload" once before actually re-running the wrapper-token fix against `synth_llava_flat/` (only the source script had been fixed at that point) — user ran the upload, it succeeded but pushed data missing the wrapper. Caught via the symptom "why did compress skip" (stale `.gz` cache existed from the earlier bad run) rather than by design; fixed and re-uploaded correctly on the second pass. `omnivideo-100k-final` was not affected (user had not yet run its upload when this happened).

### 3. Self-correction: FineVideo v5 already has rich captions + Whisper speech — earlier claim in this session was wrong

While assessing overall data readiness for a v0.2 training run, claimed (incorrectly, in conversation only — nothing wrong was written to `PROGRESS.md`/`PROGRESS_VI.md`/`REPORT.md`) that FineVideo-VLA had zero captions, based on checking `/p/data1/mmlaion/shared/vla/vla_adaptive/` and `.../FineVideo-VLA/hf_upload_flattened_adaptive/`. User pushed back, correctly. Root cause of the mistake: `vla_adaptive/` is real and correctly documented — per §22's finding and the "Long-open TODO" note near the end of §21, it is the **pre-caption/pre-SNAC v1/v2 source** that the *currently-trained* model's `.bin/.idx` was actually tokenized from (2.84B tokens) — a session log (`logs/count_tokens_4datasets.log`, not committed) had mislabeled a count from this path as `finevideo-vla-v5`, which led this session to (wrongly) treat that path as "the v5 data" without checking it against §21's own documented upload record.

**The real v5, live on HF as `EmpathicRobotics/FineVideo-Phase7-Flattened` since Jul 7 (§21):** `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/megatron_dataset_v5/` (160 files). Verified directly against 500 real records: **100% have `<caption>`, 96% have a `### Speech:` header, 80% have inline `<speech>`** — matches §21's documented format exactly. Full-corpus re-verification of `seed2`/`cosmos` token counts (332,592,448 / 3,882,954,800) matched the earlier `count_tokens_4datasets.log` figures for "finevideo-vla-v5" *exactly* — meaning that log's total (5,225,883,497 ≈ §21's documented 5.217B) was in fact computed from the correct source; only this session's own follow-up caption spot-check used the wrong local path. Net effect: the combined-dataset token count already reported (~12.5B across FineVideo/OmniVideo/synth_llava/roleplay) stands as correct; the caption/speech "gap" does not exist.

**Actionable reminder for the Phase 4→7 rerun in progress:** Phase 6 must be rerun with the caption+speech-anchor flags (`--captions-dir` / `--speech-segments-dir`, as in `slurm/submit_merge_adaptive_v4.sh`, §18) rather than a plain merge, or the regenerated corpus (fps-mismatch fix + wrapper-token fix) will silently regress and lose the caption/speech content v5 already has. `captions_dict/`/`speech_segments/` (per-video files, already generated) still exist on disk and do not need regenerating.

### Dataset landscape as of this entry (combined, not counting MV-Omni)

| Dataset | Records | Modalities | Tokens |
|---|---|---|---|
| FineVideo-VLA (v5, mid-regen for fps-mismatch + wrapper fix) | 371,888 | seed2 + cosmos + agent (18.8% full-chain) + snac + caption (Qwen2.5-VL) + speech (Whisper, header+inline) | 5.217B |
| omnivideo-100k-final | 5,214 | seed2 + cosmos + agent (799/5,214, sports-subset only) + caption + speech + QA (99,983 pairs, incl. cross-modal reasoning hints) | 274.6M |
| synth-llava/synth-llava2 | 603,999 | seed2 (static image, 32 tok/img) + caption only — no video/pose/audio | 70.8M |
| emotional-roleplay-finetuning-dataset-flattened | 67,459 | snac (audio) + text only — no video/pose | 26.9M |
| **Total** | | | **~12.53B** |

Combined with MV-Omni (+6.93B, mostly SNAC/audio) that's ~19.4B tokens — within the "10-20B sufficient" range §13 estimated, short of the more ambitious "20-40B" target. Verdict given to user: sufficient to attempt a v0.2 run once the in-progress FineVideo regen (§1, §2) completes with captions preserved (§3's reminder) — no further data-gap work (e.g. RoboVQA) is blocking that attempt.

---

## 29. Phase 4 Confirmed Complete (Occlusion-Filter Cost Quantified: 45.9%); Discord-Driven Strategic Pivot Away From Detector Improvement Toward Pre-Annotated Pose Datasets; Harmony4D Selected (MIT, Fills Occlusion + Multi-Person Gaps); Phase 5 Rerun; JUWELS (`/p`) vs JUPITER (`/e`) Storage/Compute Mismatch Found and Resolved; Harmony4D Download Started (Jul 21, 2026, afternoon)

**Main work:** Checking on Phase 4 (job `1004747`, believed ~67% per the prior entry) found it had actually **COMPLETED** at 08:19 that morning — real numbers superseded the stale in-progress snapshot. This produced the occlusion-filter cost figure Huu had asked for in a Discord conversation with Van Khue (pasted into this session for analysis): **45.9% of all 8-frame windows are dropped** by the YOLO person-presence filter (42,994,892 → 23,245,694), and 8.3% of videos (3,359/40,300) lose every window. That Discord thread traced the root cause of occlusion-dropping to a **detector limitation** (HRNet hallucinates on non-person content, e.g. mistaking a tree for a person, hence the YOLO filter as a guard) rather than a deliberate design choice, and also surfaced a second known gap: the pipeline keeps only the single most-confident bounding box per frame, discarding all other people in multi-person scenes. Huu's explicit decision: stop investing in detector improvement ("it might be a rabbit hole"), given time pressure ("summer is almost over"); instead source pre-annotated pose-video datasets from HF. Three candidates were investigated (MotionVid, OCHuman-Pose, Harmony4D) plus a survey of `xiaobai1217/Awesome-Video-Datasets`; **Harmony4D was selected** (MIT-licensed, verified against the primary HF/GitHub source rather than an incomplete third-party mirror) specifically because it addresses both gaps at once — 208 sequences, 24 subjects, always 2 people per scene (wrestling/grappling, sword fighting, ballroom, karate, MMA, hugging), multi-view video with 3D pose (17-joint, world-metric) plus SMPL mesh fit with contact-aware occlusion handling. JRDB-Pose3D was a technically closer fit (multi-person SMPL with explicit occlusion labels, 5–35 people/frame) but was rejected outright under this project's permissive-only license policy (CC BY-NC-SA). Phase 5 was rerun against the fixed Phase 4 output. Preparing Phase 6 surfaced an infrastructure mismatch between JUWELS (`/p`) and JUPITER (`/e`) storage/compute that could have silently produced a corrupted merge; resolved by moving all processing to `/e`. Harmony4D's ~352GB download was kicked off in parallel.

### 1. Phase 4 FineVideo — confirmed COMPLETED; occlusion-filter cost quantified

`sacct` confirmed job `1004747` COMPLETED at 08:19:09 (nothing left on `squeue`) — the ~67% figure in the prior entry was a stale mid-run snapshot. Final: **40,300/40,305 videos (99.99%)**, 5 failures (3 CUDA OOM from 128 workers sharing one GPU via MPS, 2 "windows could not be resolved") — a 0.012% failure rate, not worth individually retrying.

Real window counts, Phase 3 output vs Phase 4 output (`xargs -P16 wc -l`, summed correctly across every parallel batch — a first attempt using `tail -1` undercounted because `xargs -P` runs multiple `wc -l` invocations, each emitting its own "total" line; caught and fixed before trusting the number):

| | Windows (8-frame) |
|---|---|
| Before filter (Phase 3) | 42,994,892 |
| After filter (Phase 4) | 23,245,694 |
| **Dropped** | **19,749,198 (45.9%)** |

3,359/40,300 videos (8.3%) lost every single window (0 pass the filter). This is the number Van Khue owed Huu in the Discord thread below.

### 2. Discord thread analysis → strategic pivot on pose-data sourcing

User pasted a Discord conversation between Huu and Van Khue and asked for analysis. Chain of reasoning in the thread: Huu asks why occluded poses are dropped → traced to the YOLO person-presence filter existing specifically because HRNet (2D pose) hallucinates on non-person content (e.g. detects a tree as a person) — occlusion-dropping is a **detector limitation**, not an intentional design choice — meaning the dataset systematically excludes scenes like "person walking behind a tree." A second gap surfaced in the same thread: multi-person scenes only keep the single most-confident bounding box per frame, discarding everyone else. Huu's decision: **do not invest further in detector improvement** — explicitly called it "a rabbit hole" given the team is time-constrained ("we are in kind of a race now, summer is almost over"). Season's two-track training plan as stated by Huu: a smaller model on omni-VLA + RL in simulation, and a larger model on the "MV2" dataset positioned as a POC that a full omni model can be performant in language too. Instead of fixing detectors: **source pre-annotated pose-video datasets from HF**.

Notable detail: an AI-generated candidate list Huu pasted included `FineVideo-Phase2-3DPose` — **the team's own already-uploaded dataset** — listed as if third-party; Huu's own read was that this demonstrates how scarce permissively-licensed pose-video data actually is in public.

Saved to project memory (`project_pivot_pose_dataset_sourcing`): do not propose detector-improvement work going forward per this decision; "VALID" and "Leo" (JUWELS/Leonardo) were mentioned by Huu as future compute/data targets but remain unconfirmed proper nouns — verify before assuming a specific referent.

### 3. Candidate dataset investigation (background research agent) + Awesome-Video-Datasets survey — Harmony4D selected

A background research agent (read-only, no code changes) investigated the 3 named candidates plus surveyed the GitHub list:

- **MotionVid** — rejected: the HF repo has only captions + 2D DWPose keypoints, **no bundled video** (would require reassembling from 9 differently-licensed source datasets, several non-commercial).
- **OCHuman-Pose** — rejected: still images only (no video/temporal data), 2D-only, eval-split only (no train split), license ambiguous (HF tags MIT but other sources claim CC BY-NC).
- **Harmony4D** — **selected**: MIT license verified against the primary author source (`jyuntins/harmony4d` on GitHub, `Jyun-Ting/Harmony4D` on HF), not the incomplete `Voxel51/Harmony4D` mirror that under-states the license. 208 sequences, 24 subjects, always 2 people/scene, multi-view video + 3D pose (17-joint) + SMPL mesh with contact-aware fitting — addresses both the occlusion gap and the multi-person gap directly.
- `xiaobai1217/Awesome-Video-Datasets` survey (action-recognition-oriented list, not very productive for pose specifically): surfaced **JRDB-Pose3D** (SMPL multi-person 3D, 5–35 people/frame, explicit occlusion/truncation labels — technically a closer fit than Harmony4D) but **rejected outright on license** (CC BY-NC-SA), no further discussion, per this project's permissive-only policy. NTU RGB+D/120 and UAV-Human were also screened out (non-commercial license / viewpoint mismatch / unverified license respectively).

User finalized the decision directly: **Harmony4D**, explicitly setting aside JRDB-Pose3D despite its better technical fit — "we only care about permissive, we're doing open source." Also clarified the project's decision-making dynamic in the same exchange: Huu is the PI, not an approval gate — Van Khue decides technical/data-sourcing questions directly. Both points saved to project memory (`feedback_decision_authority_and_license_policy`) so future sessions don't add unnecessary "confirm with Huu first" friction to routine technical calls, and don't re-litigate the permissive-only license rule per dataset.

### 4. Phase 5 rerun against fixed Phase 4 output

The existing `outputs/agent_tokens_adaptive/` (18,847 files, Jun 19) was built on pre-fps-fix Phase 3/4 output. Phase 5's skip logic is a naive `os.path.exists(output_file)` check with no input-change awareness, so re-running it as-is would have silently skipped all 18,847 stale videos and missed the fps-mismatch fix entirely. Moved (not deleted) the stale output to `outputs/agent_tokens_adaptive_buggy_fps_mismatch_2026-07-20/` before resubmitting — same archival convention already used for `states_jsonl_30fps_buggy_2026-07-20/` and `yolo_cleaned_30fps_buggy_fps_mismatch_2026-07-20/`. Job `1006884` (64 workers) **COMPLETED in 5m28s**: 19,076 videos produced agent tokens, 21,224 produced none (mostly because arm joints are NaN in nearly all YouTube frames, consistent with §2.2's documented finding), 0 skipped.

### 5. Phase 6 — JUWELS (`/p`) vs JUPITER (`/e`) storage/compute mismatch found and resolved

Preparing to submit the existing `slurm/submit_merge_adaptive_v4.sh` (which already had the needed `--captions-dir`/`--speech-segments-dir` flags per §21's reminder) surfaced two problems that would have silently corrupted the regen:

1. Its `--input-glob` pointed at `final_dataset_adaptive_v3` — built from **pre-fps-fix** Phase 3/4/5 output (v3 dates to Jul 12; the fps-mismatch fix landed Jul 20). Running it as-is would merge fresh captions/speech onto stale agent/pose tokens, wasting the just-completed Phase 4+5 regen.
2. The script's working directory, `/p/data1/mmlaion/nguyen38/3d-human-pose`, is a **separate repo checkout 3 commits behind** `/e`'s (missing both the fps-mismatch fix and the wrapper-token fix) — resynced via `git pull`.
3. Its `--agent-tokens-dir` pointed at `/p/.../outputs/agent_tokens_adaptive`, which turned out to be an **unextracted `.tar` backup** from Jul 19, not a live directory.

User clarified `/p/data1` is **JUWELS** storage and `account=laionize` is a JUWELS-only SLURM account, not usable from this JUPITER session (`sacctmgr show assoc` confirmed only `reformo`/`jureap59` on cluster `jupiter` are available here). A direct `srun` test on the `booster` partition confirmed **JUPITER compute nodes cannot see `/p` at all** (only the login node can) — explaining why the historical merge scripts had to target JUWELS. After briefly considering staging everything to `/p` (following an earlier same-day storage-policy instruction from the user), this constraint led to the opposite, final decision: **all pipeline processing stays on `/e`** — reversing that instruction the same day once the infrastructure limitation was understood (saved to project memory `feedback_data_storage_location`). Copied the three auxiliary directories needed for the merge (`snac_tokens` 6.5GB/40,779 files, `captions_dict` 114MB/40,798 files, `speech_segments` 334MB/40,490 files — pre-existing, unaffected by the pose-pipeline bugs, no regeneration needed) from `/p` to `/e`. Wrote `slurm/submit_merge_adaptive_v5.sh` (account `reformo`/partition `booster`, `--input-glob` corrected to the raw `training_ready_rank_*.jsonl` base — not v3 — with fresh agent tokens plus the copied snac/caption/speech dirs) and submitted job `1007805` (32 array tasks), all `RUNNING` immediately with 0 errors. A live spot-check on partially-written output (112 videos / 1,148 activities) showed agent-token coverage at 16.9% — close to the historical "18.8% full-chain" figure from v3/v4 (§21) — and snac/caption coverage at ~90.7%/90.8%, both healthy signals.

### 6. Harmony4D download started

Wrote `tools/extract/download_harmony4d.py` following this repo's existing download-script convention (`snapshot_download`, resumable, retry loop, `HF_HUB_DISABLE_XET=1`). Verified the real HF repo structure directly (`Jyun-Ting/Harmony4D`): `train/` has 15 zips (~287GB, `01_hugging.zip` .. `15_mma4.zip`), `test/` has 7 zips (~65GB) — the README is only 24 bytes, no documentation, so format will need to be learned by inspecting an actual extracted zip. License MIT confirmed directly on the HF repo card. Target directory was initially set to `/p` then corrected to `/e` per §5's decision. User is running the download themselves in a `tmux` session (`logs/harmony4d_download.log`); needed `activate_env_tools.sh` (a lighter-weight env than the two main ones documented in `CLAUDE.md`, sufficient for `huggingface_hub`) after a manual `module load` + `env_stable_vla` activation failed with a missing `libpython3.12.so.1.0`. **~139/352GB (~39%) downloaded at time of writing, 0 errors.**

**Minor incident:** the user's real `HF_TOKEN` was pasted into chat while debugging the missing-module error — flagged for the user to revoke/rotate it once done.

### 7. Phase 6 (merge v5) — COMPLETED, verified clean

Job `1007805` finished in **22 minutes** (faster than the 45min–1.5h estimate). 32/32 tasks COMPLETED, exit 0:0, 160/160 output files, 0 errors. Verified: 40,804 videos, 398,775 activities (**exact match** with historical v2/v3 figures), agent-token injections at 2,326,095 (**higher** than the old 2,148,474 — +8.3%, plausibly because the fps-mismatch fix produces more correct timestamp matches, recovering more real signal rather than noise), snac injections at 38,824,718 (**exact match** with v2 — expected, since audio was never affected by the pose-pipeline bugs).

### 8. Phase 7 (flatten v6) — one failed submission, root-caused, fixed, rerun

First submission (`job 1007976`), using `setup_motionbert.sh` (env_motion_final, matching the Phase 5/6 convention) — **failed immediately** with `ModuleNotFoundError: No module named 'wn'` (the WordNet library used for text augmentation; `import wn` is an unguarded top-level import, no try/except). Notably, `sacct` still reported `COMPLETED 0:0` despite the real crash, because the submit script's trailing `echo` runs regardless of the python command's exit status — **lesson: never trust `sacct` alone, always check the real `.err` content.**

Root cause: `wn` only exists in the `env_tools` venv (`/p/data1/mmlaion/nguyen38/env_tools`) — but that venv has **the exact same JUWELS/JUPITER mismatch** found throughout this session: its `python3` is a symlink into JUWELS' own software tree (`/p/software/juwels/...`), which doesn't exist on JUPITER, so "activating" it from JUPITER is effectively a no-op — the Python that actually executes is JUPITER's module-loaded one, using packages from `~/.local` (per-user site-packages, mounted cluster-wide). `wn` had never been installed there. Fixed by running `python3 -m pip install --user wn` + `python3 -m wn download oewn:2024` from a login node (using `python3 -m pip` specifically, not the `pip`/`pip3` scripts directly — those have the same broken JUWELS-interpreter shebang as the rest of that venv).

Updated `submit_phase7_flatten_v6.sh` to drop `source activate_env_tools.sh` (produces a harmless-but-confusing `No such file or directory` on compute nodes) in favor of a direct `module load`. Resubmitted (`job 1007994`) — **ran correctly, producing real output** (verified: the first output file has 2,031 lines of correctly-formatted `### Context:`/`### Keywords:`/`### Speech:` content). A transient cluster-wide SLURM controller connectivity issue (`squeue`/`scontrol` both briefly unreachable, unrelated to this job) caused one false alarm that the job had failed — verified via direct filesystem inspection (real output content, growing file count) instead of trusting the SLURM query tools, confirming the job was healthy and did not need to be killed/resubmitted. **At time of writing: ~96/160 files done, still running, 0 errors.**

### 9. Harmony4D — download continuing, one connection drop, auto-resumed

At time of writing: **~308/352GB (~87%)**. One connection was dropped mid-transfer on `02_grappling.zip` (at 28.3/44.1GB) — the script's built-in resume logic ("Trying to resume download...") handled it automatically, no intervention needed.

### 10. Strategic discussion: "is the data enough, what is the omni model's actual output?" — three conflicting framings surfaced

User raised a top-level question, expressing genuine confusion: is the current data sufficient for the next omni-model training round, and what is the model's output actually supposed to be. Reviewing the documented history surfaced **three different, not-fully-reconciled framings running in parallel**: (1) the original `CLAUDE.md` framing — humanoid VLA, output = pose/action tokens; (2) Huu's Jul 20 framing — "omni means all modes... cross-modal bindings," with no defined concrete output; (3) the framing from the Discord thread analyzed earlier this session — two training tracks, a smaller omni-VLA + RL-in-simulation model (has a measurable target) and a larger MV2-trained model positioned as a POC that's "performant in language too" (no eval currently measures this at all). Cross-checked against `REPORT.md` itself: **this is not a new problem** — Van Khue had already flagged the identical concern on Jul 20 ("not yet a single fixed central research question..."), still unresolved.

User asked for a confident recommendation rather than a list of options. Proposed (grounded in where real engineering effort has actually gone, not in stated intentions): **the real target should be a VLA model whose core capability is generating valid action/pose tokens, with the ultimate yardstick being closed-loop task success in simulation (RL rollout)** — not a language benchmark. Evidence: the entire 7-phase pipeline exists solely to serve this; both eval metrics run so far (agent completion, modality transitions) measure action generation, not language ability; today's Harmony4D decision was made specifically to close an action-token gap. Consequence: "is the data enough" should be measured by **agent-token volume and diversity** (currently insufficient — occlusion filtering removes 45.9% of windows, coverage is lower-body-only, single-person-only — exactly why Harmony4D is worth doing), not by aggregate token count across all modalities. Explicit caveat given: if Huu genuinely wants the "performant in language too" claim to be real, **no eval currently measures that at all** — it would need a proper language benchmark run, an entirely separate, currently-unaddressed workstream.

### Status at end of entry

- **Phase 4, 5, 6 FineVideo** — all **DONE**.
- **Phase 7 (flatten v6)** — running (job `1007994`), ~96/160 files, 0 errors, ETA a few more minutes.
- **Harmony4D download** — in progress, ~308/352GB (~87%).
- Installed `wn` (WordNet) + the `oewn:2024` lexicon into `~/.local` — needed for every future Phase 7 run on JUPITER.
- Open: verify + re-upload `FineVideo-Phase7-Flattened` once Phase 7 finishes; analyze Harmony4D's real structure once downloaded and design a SMPL/COCO-17 → this project's 17-joint H36M-style conversion pipeline; **not yet run past Huu**: the proposed omni-model framing/eval protocol from §10; split `pipeline_pose/`+`pipeline_video/` into `data_prep/finevideo/` (still deferred); awaiting Huu's decision on `MixtureVitae-Backup` SNAC + the "moss" token's intended meaning; RoboVQA remains blocked on its unresolved architecture question (16 sparse frames/episode vs. Step A's continuous-video assumption).

## 30. Four Megatron Tokenize Jobs Verified Against HF and Submitted on JUWELS (FineVideo-v6, OmniVideo-100K-video, synth-llava, emotional-roleplay) (Jul 21, 2026, evening)

**Main work:** This session ran on **JUWELS**, not JUPITER, per the handoff written into `TOKENIZE_TODO.md` (repo root) by a prior JUPITER-side session — JUPITER compute nodes can't mount `/p`, where the Megatron tokenize infra (`mv-scale/`) lives. Pulled the latest repo (`bcb6180`→`72ef387`, which is what brought `TOKENIZE_TODO.md` in), read it alongside `data_prep/` to confirm the 4 datasets needing a fresh tokenize: **FineVideo-VLA v6, OmniVideo-100K (video), synth-llava, emotional-roleplay**.

Per explicit user instruction, verified each dataset's local `/p` copy against its HF upload *before* submitting any job — not just trusting `TOKENIZE_TODO.md`'s claims. Downloaded the full HF repo for the 3 datasets that have one (`EmpathicRobotics/omnivideo-100k-final`, `EmpathicRobotics/synth-llava`, `EmpathicRobotics/emotional-roleplay-finetuning-dataset-flattened`) via `huggingface_hub.snapshot_download`, decompressed, sorted records by id, and compared a sha256 hash of (id, text) pairs against the same hash computed on the local flat JSONL. **All 3 matched exactly, byte-for-byte** — local is not stale relative to what's live on HF. FineVideo-VLA v6 has no HF upload yet (HF's `FineVideo-Phase7-Flattened` is still the old v1/19GB/2.84B-token dataset) — nothing to cross-check there; verified instead that the local record count (371,892) matches what `TOKENIZE_TODO.md` claimed.

Created/edited 4 sbatch files in `/p/data1/mmlaion/nguyen38/mv-scale/` (shared tokenize infra, **not** part of this git repo):
- `tokenize_finevideo_v6.sbatch` — new, copied from `tokenize_finevideo_v5.sbatch` (4-node Ray cluster), `INPUT`/`OUTPUT_PREFIX` updated to v6, Ray port bumped to 20160 (checked against every other active sbatch's port to avoid collision).
- `tokenize_omnivideo_100k_video.sbatch` — edited in place: `INPUT` changed from the stale `omnivideo_100k_video_flattened` (predates the 21/07 wrapper-token fix) to `omnivideo_100k_final/`, the plain (pre-gzip) flat JSONL directory `phase7_finalize_omnivideo.py` writes directly — not `hf_upload/`, which is the same content gzipped and train/test-split for HF only; confirmed `mv_preprocess_data.py`'s file glob is non-recursive so it correctly ignores the `hf_upload/` subdirectory.
- `tokenize_synth_llava.sbatch` / `tokenize_roleplay.sbatch` — both new (never tokenized before), single-node pattern matching `tokenize_omnivideo_100k_video.sbatch`/`tokenize_robovqa.sbatch` rather than the 4-node `tokenize_mv_omni.sbatch` pattern, since both sources are small (547MB / 339MB) and don't need a multi-node Ray cluster.

All 4 use the same `tokenizer_vla_qwen3` (257,901 vocab) confirmed from the actual sbatch templates already running, keeping token IDs consistent across shards for training-time mixing.

**Submitted to SLURM** (JUWELS, account `laionize`, partition `batch`): `14127888` (tok_finevideo_v6, 4 nodes), `14127889` (tok_omni_video, 1 node), `14127890` (tok_synth_llava, 1 node), `14127891` (tok_roleplay, 1 node). All reached RUNNING within 2 minutes of submission (`squeue` confirmed).

**Update, same evening — all 4 jobs confirmed genuinely COMPLETED with real token counts, not just SLURM state.** Tracked via a backgrounded `squeue`/`sacct` poll loop (per the standing Jul 18 lesson: a job can report COMPLETED on SLURM while having silently failed at the Ray-connection step) until all 4 finished; grepped every log for `Traceback` (none found); counted real tokens by reading each `.idx` header directly (same logic as `mv-scale/count_tokens.py`) and verified the bin-size-vs-summed-token-lengths consistency check on every shard (all PASS):

| Job | Real tokens | Docs | Wall time |
|---|---|---|---|
| finevideo_v6 | **10,926,767,551 (10.93B)** | 371,892 | 53m12s |
| omnivideo_100k_video | 536,149,780 (0.54B) | 5,214 | 19m47s |
| synth_llava | 103,097,102 (0.10B) | 603,999 | 27m06s |
| roleplay | 52,469,577 (0.05B) | 67,459 | 6m07s |
| **Total, this session's 4 jobs** | **11,618,484,010 (11.62B)** | 1,048,564 | |

finevideo_v6's real count (10.93B) is ~2x the flatten-stage word-count estimate (5.443B, per the `TOKENIZE_TODO.md` table) — the same gap already documented for v5 (10.55B real vs 5.256B estimated, Jul 18 §22), not a new discrepancy; root cause remains the free-text-span word-count approximation understating real BPE token count, not something specific to v6.

Also re-counted MV-Omni (tokenized 18/07, not part of this session) with the same method for a complete picture: **20,389,561,883 (20.39B) tokens, 1,593,301 docs, PASS** — matches the figure already on record in project memory, confirming it as still valid. **Combined real-token total now available for training (excluding RoboVQA/OmniVideo-100K-QA, unverified, and the two shards already spoken for by trained models):** 11.62B + 20.39B = **~32.01B tokens**.

**Still open, out of this session's scope (see `TOKENIZE_TODO.md` §2-4):** MV-Omni / OmniVideo-100K-QA / RoboVQA existing tokenized outputs are "probably still valid" but unverified against their current sources — spot-check before reuse, don't blindly retokenize. `vla_25b`/`vla_adaptive` tokenized outputs belong to already-trained models — leave untouched. Training mix ratio across shards remains deliberately deferred to train-config time (Van Khue's call). Eval protocol (MPJPE / modality-transition / instruction-following) is still undefined.

## 31. `qwen3_1.7b_vla_v2` Training Confirmed Complete (Loss 6.47→1.75, Clean Convergence); Qualitative Sanity Eval Run (Greedy + Sampling); First Real Cosmos→MP4 Decode From Model's Own Generation (Jul 22, 2026)

**Context:** This session picked up mid-flight — the actual training job (`1009758`, config `qwen3_1.7b_vla_v2.yaml`, the 5-source ~32B-token mix: FineVideo-v6, MV-Omni, OmniVideo-100K, synth-llava, emotional-roleplay, tokenizer `tokenizer_vla_qwen3`) had been submitted in a prior session per `PROGRESS_VI.md`'s Jul 21 entry, with checkpoint-conversion-on-completion left to an orchestrator that had since been killed. First task this session: confirm whether that run actually finished.

### Training status — confirmed COMPLETE, clean

Job `1009758` ran all **7,632/7,632 iterations** (`train_iters` in config), 64 nodes × 4 GH200. Loss curve: 6.472 (iter 50) → 2.840 (500) → 1.827 (4000) → 1.694 (7600), **0 skipped/NaN iterations** anywhere in the run. Final eval: val loss 1.7526 (PPL 5.77), test loss 1.7722 (PPL 5.88). `train_iters=7632 × global_batch_size=1024 × seq_length=4096 = 32.01B tokens` — exactly 1 epoch over the full mix, consistent with `TOKENIZE_TODO.md`'s ~32.01B combined total.

The job's own checkpoint-conversion step (Megatron→HF) **failed** — compute node has no internet, and `AutoTokenizer.from_pretrained()` tried to hit `huggingface.co` instead of using the local tokenizer path (`OSError: couldn't connect to huggingface.co`). A second job (`1010685`) re-ran just the conversion step and succeeded, producing all 16 checkpoints (`hf/iter_0000500` … `hf/iter_0007632`) in HF format. `squeue` confirmed no jobs left running — nothing was silently still in flight.

Architecture (from `hf/iter_0007632/config.json`): Qwen3ForCausalLM, 28 layers, hidden_size 2048, intermediate_size 6144, 16 attention heads / 8 KV heads (GQA), head_dim 128, `qk_layernorm=true`, `rope_theta=1e6`, tied embeddings, `max_position_embeddings=4096`, vocab 257,920 (padded from tokenizer's real 257,901). **1.94B total params.**

### Eval — new script, adapted from the v1 sanity checker

Wrote `tools/eval/eval_vla_v2_sanity.py` (based on the existing `eval_vla_sanity.py` used for the first model), retargeted at the new checkpoint/tokenizer, with atomicity tests added for the new `snac_`/`caption`/`speech` token categories and 5 generation prompts (`full_prompt`, `agent_continuation`, `agent_from_scratch`, `roleplay_speech`, `image_caption`). Script supports both greedy (default) and sampling (`--sample --temperature --top-p --repetition-penalty`) decoding, and prints the **full** prompt/output/ground-truth per test (not just a 40-token preview) for readability.

**Token atomicity: 36/37 pass.** One real bug found: `<snac_140553>` (the last ID in the 12,288-token SNAC range) splits into 11 BPE sub-pieces instead of staying atomic — an off-by-one at the vocab boundary, same class of bug as the v1 tokenizer issue but isolated to a single edge token this time. Not yet fixed in the canonical tokenizer at `/e/data1/datasets/playground/mmlaion/shared/nguyen38/tokenizer_vla_qwen3/` (a scratch-local patched copy was used for eval only, to work around an unrelated `transformers==4.57.6` / `extra_special_tokens` format incompatibility with this Qwen3-base tokenizer_config.json — that patch is cosmetic/version-shim only, unrelated to the snac bug).

**Qualitative result — the exact failure mode the v1 model had is gone.** `CLAUDE.md`'s existing record on the first model: *"Modality transitions: FAIL — model stays in seed2 mode, never transitions to cosmos/avclm/agent from text alone."* This model transitions freely across all 6 categories it was trained on (seed2, cosmos, snac, speech, caption, agent) in both greedy and sampled generation. Strongest single piece of evidence: given **only** the 32 real `<seed2_N>` tokens from a held-out `synth_llava2` record (id `synth_llava2_003266024`, no other text hint), the model emitted a topically-correct caption ("portrait of a young boy... green graduation cap and gown... looking at the camera") closely matching the real ground-truth caption, then closed `</caption><|im_end|>` cleanly — genuine image↔text cross-modal binding, not template noise.

Weakness found: **greedy decoding degenerates into repeated-token loops inside long `cosmos` runs** (e.g. `<cosmos_42631>` × 6–8 in a row), burning the max-token budget before reaching `<fps_N>`/`agent` in 3 of 5 prompts. Sampling (T=0.8, top_p=0.9, repetition_penalty=1.3) fixes this for at least one full test (`agent_from_scratch`): the model completed **two full 8-frame agent windows** end-to-end within a single generation (seed2→cosmos→agent→snac/speech→seed2→cosmos→agent), decoding to valid, non-degenerate 3D coordinates — at the cost of occasionally hallucinating caption detail not present in the source image (a name, "Timothy Kelly", invented under sampling that greedy did not produce).

### First real cosmos→video decode from this model's own output

Extracted two clean 200-token `<cosmos_N>` chunks from the `agent_from_scratch` (sampled) generation and ran them through the existing `tools/decode/decode_cosmos.py` (Cosmos-Tokenizer-DV8x16x16 decoder, previously verified 20/07 only against real training data, never against a trained model's own generation before). **Both decoded to valid playable MP4s** — first confirmed instance of this model's `cosmos` output round-tripping to an actual video a human can watch (vs. the pose/text modalities, whose round-trip was already established). `avc_lm`/`seed2`/`snac` remain the 3 modalities where model-generation-to-real-media has *not* been demonstrated: `avc_lm` because the model essentially never produces it (0 count across all 5 test prompts — matches `decode_avclm.py`'s note that avc_lm is stripped out at the flatten stage before training data is built, so the model barely saw any); `seed2`→image and `snac`→audio because **no decoder script exists in this repo at all** for either direction yet (unlike cosmos/avc_lm, which both have a working `tools/decode/*.py`).

### Artifacts saved

`samples/qwen3_1.7b_vla_v2_eval/`: `eval_greedy_iter0007632.log`, `eval_sample_T0.8_iter0007632.log` (full transcripts), `cosmos_decoded_agent_from_scratch_sample_chunk{0,1}.mp4` + their raw token-id `.txt` sources.

### Status / open items (as first written)

- Training + checkpoint conversion: **DONE**, nothing in flight.
- Eval so far is qualitative/manual only — no MPJPE, no BLEU/CIDEr, no closed-loop task-success metric yet (same open item as §30/§29 — eval protocol still undefined).
- ~~`<snac_140553>` atomicity bug~~ — **retracted, see follow-up below: not a real bug.**
- Model card + HF upload for this checkpoint: in progress, not yet pushed (see `tools/upload/upload_vla_v2_model.py`).
- Not yet decided: whether to also run the sampling-vs-greedy comparison across all 5 prompts systematically, or build a seed2→image and snac→audio decoder to complete the modality-verification picture.

### Follow-up same day: `<snac_140553>` retracted (not a bug), SNAC audio decoder built, `cosmos` found to dominate generation, `seq_length` doubled for the next run

**`<snac_140553>` atomicity "bug" retracted.** Re-checked the real tokenizer's `added_tokens` list directly (not just spot-testing an assumed ID): the SNAC vocab is **3 disjoint bands** of 4,096 IDs each — L0 `128266–132361`, L1-even `132362–136457`, L1-odd `144650–148745` — with a real gap, `136458–144649`, that was never added by design (matches the "listen format" 2-of-3-codebook-levels encoding in `tokenize_snac.py`). `<snac_140553>` sits inside that gap, so it correctly was never a token — my original atomicity test picked an ID that doesn't exist rather than finding a genuine tokenizer defect. All 12,288 real SNAC IDs remain atomic.

**Built `tools/decode/decode_snac.py`**, the first seed2/snac-direction decoder in this repo (previously only `decode_cosmos.py`/`decode_avclm.py` existed, both video). Reconstructs the SNAC codec's 3-level hierarchical codes from listen-format triplets, zero-filling the never-encoded level-2 (finest, 50Hz) band, and calls `SNAC.from_pretrained("hubertsiuzdak/snac_24khz").decode()` to get a real waveform. Ran it on 159 `<snac_N>` tokens (53 base frames, ~4.5s) concatenated from the model's own sampled generation (§31 above) — **produced a real, non-silent, non-clipping WAV** (RMS 0.12, range [-0.46, 0.55]). This closes one of the two remaining "no decoder exists" gaps flagged in §31 — only `seed2`→image reconstruction remains undemonstrated in this repo.

**`cosmos` tokens dominate the model's own generation.** Aggregated `Breakdown` counts across all 10 test runs (5 prompts × greedy + sampled): of tokens belonging to an actual VLA/multimodal category (excluding plain narrative `text`), **`cosmos` is 61–77%** of that total — `agent`/`seed2`/`snac` combined are a minority. Root cause is structural, not necessarily a training bias per se: one `cosmos` chunk costs a fixed 200 tokens by convention, versus ~2–4 tokens per `agent` sample or 1 token per `seed2`/`snac` sample — so `cosmos` is inherently the most token-expensive category to represent, independent of how often it's semantically "meant" to appear. Practical consequence: this is very plausibly the direct cause of the repeated-token degeneration and context-budget exhaustion noted in §31 (a long, expensive `cosmos` run eating the generation budget before `<fps_N>`/`agent` is reached). This is the same class of concern `CLAUDE.md`'s v1-era "Next steps" plan already flagged (*"reduce modality dropout from 99%/90% to 80-90%/50-70% for avclm/cosmos"*) but that adjustment was never actually applied for this v2 training run — worth doing together with the `seq_length` change below, not addressed yet.

At the **data-mix** level (as opposed to generation-level), the project had already caught and corrected an analogous dominance concern before this v2 run even started: MV-Omni is 63.71% of raw tokens but was deliberately down-weighted to 39.71% of training weight specifically because it carries no `<agent>` tokens (see `qwen3_1.7b_vla_v2.yaml`'s header comment) — so the *data*-level imbalance was already handled; today's finding is a *new, generation-level* imbalance (`cosmos`'s per-chunk token cost) that mix-reweighting alone doesn't address.

**`seq_length` doubled for the next run.** Wrote `oellm-autoexp/config/experiments/nguyen38/qwen3_1.7b_vla_v3.yaml` — identical to `qwen3_1.7b_vla_v2.yaml` (same architecture, same tokenizer, same 5-source ~32.01B-token mix/weights) except `seq_length: 4096 → 8192`, with `train_iters`/`lr_warmup_iters`/`lr_decay_iters`/`lr_wsd_decay_iters`/`eval_interval` all recomputed to still cover exactly ~1 epoch (7,632 → 3,816 iters, same warmup/decay ratios). Rationale, directly evidenced by §31 + this entry: a single full seed2→cosmos→agent→snac/speech cycle already runs several hundred to ~1,500 tokens, and `cosmos` alone can be 200+ tokens per chunk — at 4,096 tokens per training window, only 1–3 such cycles fit, well short of a real multi-second activity. **Not yet submitted** — this is a config-only change, pending a decision on whether to also apply the modality-dropout adjustment above before the next real training run.

Artifacts added to `samples/qwen3_1.7b_vla_v2_eval/`: `snac_decoded_sample_generated.wav`, `snac_raw_ids_generated.txt`.
