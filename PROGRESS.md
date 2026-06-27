# PAB-Spline VLA — Project Progress

**Author:** Van Khue Nguyen  
**Last updated:** June 27, 2026  
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

---

### Phase 7: Flatten + Augment

**Script:** `pipeline_pose/phase7_flatten.py`

Converts hierarchical JSON → flat Megatron-LM JSONL. Key decisions:

**Agent-only filter:** Only activities with `<agent>` blocks are emitted (every training record has action data).

**Modality dropout (token balancing):**
| Modality | Raw ratio vs agent | Drop rate | Resulting ratio |
|----------|-------------------|-----------|----------------|
| AVC-LM | ~373× | 99% | ~4× |
| Cosmos | ~19× | 90% | ~2× |
| Seed2 | ~1× | 0% | 1× |
| Agent | baseline | 0% | 1× |

**Text augmentation:** 15% synonym replacement, 5% stopword dropout, 10% sentence permutation, random speech/token interleaving, random layout block shuffling.

Output: 160 files, **69,844 records**, 19.2 GB.

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

**Priority 1 — Vocab expansion for SNAC + seed tokens**
- Add `<snac_0>` ... `<snac_4095>` (~4096 tokens) and `<seed_0>` ... `<seed_8191>` (~8192 tokens) via `add_tokens(special_tokens=True)`
- New vocab: ~156,500 tokens
- Unlocks MV-Omni's **6.93B tokens** for training
- Effort: 1–2 days

**Priority 2 — Adjust modality dropout in Phase 7**
- AVC-LM: 99% → 80–90% drop (keep 10–20%)
- Cosmos: 90% → 50–70% drop (keep 30–50%)
- Re-flatten + re-tokenize
- Model will see full modality transition chains, fixing root cause 3
- Effort: 1 day code + 1–2 days SLURM

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

**Priority 6 — SNAC tokenization for FineVideo**
- Run Orpheus SNAC2 on audio tracks of ~18K FineVideo videos (those with agent tokens)
- Inject `<snac_N>` tokens into training records alongside existing seed2/cosmos/avclm/agent
- Adds first-person + third-person + audio modality binding
- Expected: meaningful cross-modal binding (speech ↔ motion)

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

---

## Published Artifacts

| Artifact | Location | Status |
|----------|----------|--------|
| Tokenizer (144,215 vocab) | `EmpathicRobotics/tokenizer-vla-adaptive` | Live |
| FineVideo-Phase7-Flattened (69,844 records) | `EmpathicRobotics/FineVideo-Phase7-Flattened` | Live |
| FineVideo-Phase5-AgentTokens (~399K activities) | `EmpathicRobotics/FineVideo-Phase5-AgentTokens` | Live |
| FineVideo-Phase4-YOLOPose (millions of windows) | `EmpathicRobotics/FineVideo-Phase4-YOLOPose` | Live |
| VLA Model v1 (broken tokenizer) | `EmpathicRobotics/vla-1.7b-pab-spline-25b-test` | Live (deprecated) |
| VLA Model v2 (fixed tokenizer) | `EmpathicRobotics/vla-1.7b-pab-spline-adaptive` | Live |
| Megatron .bin/.idx shards (2.84B tokens) | `/p/data1/mmlaion/shared/vla/tokenized_output/vla_adaptive/` | Local |
| Data inventory checkpoint | `tools/inventory_checkpoint_v2.json` | Local |

---

## Immediate Action Items (Next 2 Weeks)

- [ ] Vocab expansion: add `<snac_N>` and `<seed_N>` tokens to tokenizer
- [ ] Adjust Phase 7 dropout rates (AVC-LM → 80–90%, Cosmos → 50–70%)
- [ ] Start writing ego-centric perspective converter
- [ ] Start writing captioning pipeline code (SmolVLM2 / Qwen2.5-VL on keyframes)
- [ ] Investigate leo seed2 + euro_pat token counts
- [ ] Plan Cosmos3-DROID pipeline (download strategy, SLURM script)
