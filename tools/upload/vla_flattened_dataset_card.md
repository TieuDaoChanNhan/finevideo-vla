---
license: apache-2.0
task_categories:
  - video-classification
  - robotics
  - text-generation
tags:
  - 3d-pose
  - human-pose-estimation
  - motion
  - finevideo
  - vla
  - multimodal
  - tokenization
  - adaptive-pchip
  - snac
  - caption
  - speech-transcription
  - megatron-lm
  - pretraining
language:
  - en
size_categories:
  - 100K<n<1M
---

# FineVideo-Phase7-Flattened — Megatron-LM Multimodal Pretraining Dataset (v6)

## Overview

This is the **final, training-ready** flattened dataset from the FineVideo-VLA pipeline. Each record is a single `{"text": "..."}` JSON line containing interleaved multimodal tokens — ready for Megatron-LM tokenization and LLM pretraining.

Six token/text modalities are interleaved **per 8-frame chunk** in temporal order:

- **Seed2** — 1 FPS semantic keyframe tokens (vocab: 8192), kept at 100%
- **Cosmos** — every 8-frame spatial video tokens (vocab: 64,000), kept at 50%
- **Agent** — adaptive PCHIP 3D human pose tokens (17 joints, variable control points per joint)
- **SNAC** — audio tokens in listen format (~10 tokens per 8-frame chunk, vocab: 12,288)
- **Caption** *(new in v5)* — natural-language VLM caption anchored to the activity's opening frame and every person-appears/disappears event, plain BPE text (no vocab expansion)
- **Speech** *(new in v5)* — inline ASR transcript segment anchored to its exact chunk (distinct from the whole-activity `### Speech:` header, which is unchanged), plain BPE text

Source: ~40,000 YouTube videos from [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo).

**Key property:** Every record contains at least one of `<agent>` or `<snac>` tokens. Records without either are discarded.

## Dataset Statistics (v6, Jul 21, 2026)

| Metric | Value |
|--------|-------|
| Total records | **371,892** |
| Full-chain records (agent + snac) | 73,037 (19.6%) |
| Partial-chain records (snac only) | 298,818 (80.4%) |
| Bad records (neither agent nor snac) | 0 (verified full scan) |
| Total shards | 160 |
| Train shards | 152 |
| Test shards | 8 |
| Split ratio | 95/5 (seed 42) |
| Compression | gzip level 5 |

### Token counts

| Modality | Tokens | % |
|----------|--------|---|
| seed2 | 353,379,612 | 6.5% |
| cosmos | 3,921,239,352 | 72.0% |
| agent | 689,088,435 | 12.7% |
| snac | 440,678,767 | 8.1% |
| caption | 12,076,095 | 0.2% |
| speech_inline | 27,012,431 | 0.5% |
| **TOTAL** | **5,443,474,692** | **5.443B** |

## What Changed in v6 (Jul 21, 2026)

Two independent fixes, both re-run from scratch on top of the raw Step A `training_ready_rank_*.jsonl` (not on top of v5's intermediate files):

1. **Agent tokens rebuilt on fps-mismatch-fixed pose data.** Phase 3 (kinematics) and Phase 4 (YOLO person-presence cleaning) both had a bug where, for any video whose native fps deviated from 30 (**~35% of FineVideo's 40,804 videos**), occlusion masking and person-presence filtering were computed against the wrong point in time — up to ~20% of a video's duration of drift by the end. This was fixed in the pipeline scripts and Phase 3→4→5→6 were re-run on the corrected data before this flatten. Effect: Phase 6 injected **2,326,095 agent blocks**, +8.3% vs the pre-fix v5 figure (2,148,474) — more windows now correctly have all 17 joints finite simultaneously once timing is right.
2. **Modality wrapper tokens restored.** `<seed2>...</seed2>`, `<cosmos>...</cosmos>`, `<agent>...</agent>`, and `<snac>...</snac>` are now kept around each block (previously stripped during flattening for every modality except `<caption>`/`<speech>`). These tokens were already registered in the tokenizer vocab but unused — the outer wrapper gives the model an explicit "block ended, modality changes now" signal, distinct from "here comes the next token in the same block", which the earlier (unwrapped) format conflated. This is a direct response to the "modality transitions: FAIL" result from the v1 tokenizer evaluation (model never self-initiates a seed2→cosmos→agent transition).

Both changes touch the raw token counts (fix #1 adds more agent windows; fix #2 adds 2 tokens per block for every wrapped modality), which is why every modality's token count moved slightly, not just agent's. Record count is essentially unchanged (371,888 → 371,892) — this is a token-content and token-format fix, not a new data source.

## What Changed in v5 (Jul 17, 2026)

Added `<caption>` and `<speech>` language anchors at modality-transition points, to give the model a language signal for *why* the token stream is about to switch modality (previously identified as root cause #2 for the model's inability to self-initiate modality transitions at inference).

- `<caption>...</caption>` is inserted immediately before the `<cosmos>` block of its anchor chunk. Anchor points: the activity's opening chunk, plus every chunk where a person appears/disappears (YOLO `has_agent` flip, 5s-debounced against flicker).
- `<speech>...</speech>` is inserted immediately after `</avc_lm>`, anchored to its exact chunk's ASR segment — distinct from the existing whole-activity `### Speech:` header (intentionally redundant: header = full-activity dump for global context, inline = precisely-timed local anchor).
- Neither is dropped (0% dropout, same treatment as `agent`) nor text-augmented — both are anchored to an exact chunk, so paraphrasing/permutation would break the token-to-moment correspondence that's the entire point of adding them.
- Measured token growth: **+0.740%** over the v4 baseline (5.217B → 5.256B), independently cross-checked two ways (a controlled real-pipeline sample and an exact full-dataset word count) that agreed to within 0.012 percentage points.
- This is a *qualitative* fix (a language anchor at modality-transition points), not a record-count multiplier — it should not be confused with the separately-tracked "perspective framing" idea (robot/human/cinematic re-framings of the same activity), which is what would actually multiply total record count and is not yet implemented.

## What Changed in v4 (Jul 2, 2026)

v4 fixes two critical bugs present in v3:

### Bug 1 — Temporal misalignment (CRITICAL, fixed)

**v3 behavior:** All agent tokens were appended after all video tokens; all snac tokens came last. At seq_len=4096, only 31% of full-chain records had any agent tokens within the training context window — the model rarely saw video and pose simultaneously.

**v4 behavior:** State machine walks Phase 6 output in document order, emitting per-chunk: `[seed2?][cosmos?][agent?][snac?]`. Each 8-frame chunk produces ~490 aligned tokens (cosmos 200 + agent ~280 + snac 10), giving 8–10 fully aligned multimodal tuples per 4096-token context window.

### Bug 2 — Speech injection into agent grammar (CRITICAL, fixed)

**v3 behavior:** `interleave_speech_and_tokens()` scattered speech words (e.g. `"turn"`, `"left"`) into the middle of agent joint token sequences, breaking the `<pelvis_x_N>` grammar and causing the model to see invalid joint sequences in 42.9% of full-chain records.

**v4 behavior:** Speech is placed exclusively in a `### Speech: ...` header field, completely separated from the token sequence.

## Modality Dropout (Token Balancing)

In the raw data, image tokens massively outnumber action tokens. **Modality dropout** is applied during flattening to balance modalities:

| Modality | Drop rate | Reason |
|----------|-----------|--------|
| AVC-LM | **100%** | Removed until ablation studies confirm benefit |
| Cosmos | **50%** | Per-chunk coin flip — keeps ~50% of spatial context |
| Seed2 | 0% | Keep all — primary visual signal |
| Agent | 0% | Keep all |
| SNAC | 0% | Keep all |
| Caption *(new)* | 0% | Anchored to an exact chunk — paraphrasing would break the token-to-moment link |
| Speech (inline) *(new)* | 0% | Same reason as caption |

## Data Format

Each line is a JSON object with a single `text` field:

```json
{
  "text": "### Title: Launching\n### Context: A video showcasing diverse vocation paths...\n### Keywords: educational, informative\n### Speech: turn left and walk forward\n<caption> The person is standing on top of a large log. </caption> <seed2> <seed2_6750> <seed2_680> ... </seed2> <cosmos> <cosmos_18232> <cosmos_41007> ... </cosmos> <agent> <fps_30> <pelvis> <pelvis_t_0> <pelvis_x_128> ... </pelvis> <r_hip> ... </r_hip> ... </agent> <snac> <snac_132247> <snac_132788> ... </snac> <speech> We're in West Bank, in the heart of the reserve. </speech>"
}
```

### Structure within `text`

Each record has text headers followed by the flat token sequence. Headers are randomly shuffled:

```
### Title: <scene title, augmented>
### Context: <global context + activity prompt, augmented>
### Keywords: <scene thematic + mood, augmented>
[### Speech: <speech transcript, augmented>]   ← only if speech present
<flat token sequence in per-chunk temporal order>
```

### Per-chunk token order

Tokens are emitted in document order, one 8-frame chunk at a time:

```
chunk 0:  [<caption>?] [<seed2>...</seed2>?] [<cosmos>...</cosmos>] [<agent><fps_30>...</r_wrist></agent>?] [<snac>...</snac>?] [<speech>?]
chunk 1:               [<cosmos>...</cosmos>] [<agent><fps_30>...</r_wrist></agent>?] [<snac>...</snac>?]
chunk 2:  [<caption>?] [<seed2>...</seed2>?] [<cosmos>...</cosmos>] [<agent><fps_30>...</r_wrist></agent>?] [<snac>...</snac>?] [<speech>?]
...
```

*(new in v6)* Each `<seed2>`, `<cosmos>`, `<agent>`, and `<snac>` block is now wrapped in its own open/close tag — e.g. `<agent> <fps_30> <pelvis>...</pelvis>...</r_wrist> </agent>` — giving the model an explicit end-of-block signal distinct from "next token in this block". Previously these wrappers were stripped during flattening; `<caption>`/`<speech>` were always wrapped.

- seed2 appears at 1fps keyframe chunks (every ~3.75 chunks at 30fps)
- cosmos present at 50% of chunks (random per chunk)
- agent present only at chunks with a detected person
- snac present at ~100% of chunks (audio available for most activities)
- caption appears only at anchor chunks (opening frame + person-appears/disappears events, 5s-debounced) — most activities get ~2.45 captions on average
- speech (inline) appears only at chunks with an ASR segment mapped to them

### Agent token format (Adaptive PCHIP)

Each 8-frame chunk of pose uses adaptive control points per joint:

```
<fps_30>
<pelvis> <pelvis_t_0> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128>
         <pelvis_t_7> <pelvis_x_130> <pelvis_y_128> <pelvis_z_130> </pelvis>
<r_hip>  <r_hip_t_0>  <r_hip_x_140>  <r_hip_y_120>  <r_hip_z_115>
         <r_hip_t_7>  <r_hip_x_141>  <r_hip_y_121>  <r_hip_z_116>  </r_hip>
...17 joints total...
```

- **t tokens**: frame index 0–7 within the 8-frame window
- **xyz tokens**: quantized uint8 [0, 255], mapping [-2.0m, +2.0m]
- **Dequantize**: `position_metres = token_value / 255.0 * 4.0 - 2.0`
- **CP tiers**: 2 CPs (low curvature) / 4 CPs (medium) / 8 CPs (high motion)
- **Token count per chunk**: 171 (all 2-CP) to 579 (all 8-CP), typical ~250–300
- **Reconstruct 8 frames**: parse t/x/y/z per joint → apply PCHIP interpolation

### Joint names (H36M 17-joint skeleton)

| Joint | Joint | Joint |
|-------|-------|-------|
| pelvis | r_hip | r_knee |
| r_ankle | l_hip | l_knee |
| l_ankle | spine | thorax |
| nose | head_top | l_shoulder |
| l_elbow | l_wrist | r_shoulder |
| r_elbow | r_wrist | |

### SNAC token format

SNAC tokens use the listen format from [Orpheus SNAC2](https://huggingface.co/canopylabs/orpheus-3b-0.1-pretrain):

```
<snac_132247> <snac_132788> <snac_147076> ...
```

- 9 or 12 tokens per 8-frame chunk (alternating, due to 3.33 base frames/chunk at 30fps)
- Vocabulary: `<snac_128266>` ... `<snac_148745>` (L0: 128266–132361, L1A: 132362–136457, L1B: 144650–148745)
- Full activity audio encoded once, then split proportionally across chunks (preserves audio context)

## Data Augmentation

Text fields have augmentation applied during flattening:

| Augmentation | Rate | Description |
|-------------|------|-------------|
| Synonym replacement | 15% | Content words (>5 chars) replaced with WordNet synonyms |
| Stopword dropout | 5% | Common stopwords randomly removed |
| Sentence permutation | 10% | Speech transcript sentences randomly reordered |
| Layout block shuffling | — | Title/Context/Keywords/Speech blocks randomly reordered |

## Vocabulary & Tokenizer

Use **[EmpathicRobotics/tokenizer-vla-adaptive-v2](https://huggingface.co/EmpathicRobotics/tokenizer-vla-adaptive-v2)** (156,509 vocab) for this dataset — it includes SNAC tokens (absent in v1, 144,215 vocab) plus the 4 `<caption>`/`</caption>`/`<speech>`/`</speech>` wrapper tokens needed for v5. A Qwen3-based tokenizer with the full VLA vocabulary is also available: [EmpathicRobotics/tokenizer-vla-qwen3](https://huggingface.co/EmpathicRobotics/tokenizer-vla-qwen3) (257,901 vocab).

All VLA tokens are registered via `add_tokens(special_tokens=True)` — the BPE tokenizer treats every VLA token as atomic and never splits them.

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("EmpathicRobotics/tokenizer-vla-adaptive-v2")
tok.encode("<seed2_1137>")      # -> single token
tok.encode("<pelvis_x_128>")    # -> single token
tok.encode("<snac_132247>")     # -> single token
tok.encode("<caption>")         # -> single token (new in v5)
```

| Token family | Range | Count |
|-------------|-------|-------|
| Base GPT-NeoX-20b | — | 50,277 |
| `<seed2_N>` | 0–8191 | 8,192 |
| `<cosmos_N>` | 0–63999 | 64,000 |
| `<avclm_N>` | 0–8191 | 8,192 |
| `<fps_N>` | 0–59 | 60 |
| Joint tokens (xyz, t, wrappers) | — | 13,226 |
| Modality wrappers | — | 8 |
| `<snac_N>` (L0 + L1A + L1B) | 128266–148745 | 12,290 |
| `<caption>`/`</caption>`/`<speech>`/`</speech>` *(new)* | — | 4 |
| **Total** | | **156,509** |

Caption/speech text content itself is regular English — tokenized with the base BPE vocabulary, no new numbered token family needed (unlike seed2/cosmos/agent/snac, which each got a dedicated `<..._N>` range).

## Related Resources

| Resource | Description |
|----------|-------------|
| [EmpathicRobotics/tokenizer-vla-adaptive-v2](https://huggingface.co/EmpathicRobotics/tokenizer-vla-adaptive-v2) | Recommended tokenizer for this dataset (156,505 vocab, includes SNAC) |
| [EmpathicRobotics/tokenizer-vla-adaptive](https://huggingface.co/EmpathicRobotics/tokenizer-vla-adaptive) | v1 tokenizer (144,215 vocab, no SNAC) |
| [EmpathicRobotics/FineVideo-Phase5-AgentTokens](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase5-AgentTokens) | Pre-flattening hierarchical dataset (full metadata, no dropout) |
| [EmpathicRobotics/FineVideo-Phase4-YOLOPose](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase4-YOLOPose) | Raw 3D pose data (float arrays, not tokenised) |

## Pipeline Summary

| Phase | Description | Status |
|-------|-------------|--------|
| Step A | Seed2 + Cosmos + AVC-LM tokenisation (40 nodes × 4 GPU) | Done |
| Phase 1 | HRNet 2D pose detection | Done |
| Phase 2 | MotionBERT 2D→3D lifting | Done |
| Phase 2.5 | Resample to 30fps | Done |
| Phase 3 | Kinematics: bone normalisation, root centering, smoothing | Done |
| Phase 4 | YOLO person-detection cleaning | Done |
| Phase 5 | Adaptive PCHIP per-joint tokenisation (18,847 videos) | Done |
| Phase 6 v2 | Merge agent + SNAC tokens into multimodal dataset | Done |
| Phase 7 v4 | Per-chunk temporal flatten + modality dropout | Done |
| Phase 6 v4 | Inject caption + inline speech language anchors | Done |
| Phase 7 v5 | Flatten with caption/speech events, 0% dropout on both | Done |
| Phase 3/4 fix | fps-mismatch fix in kinematics + YOLO cleaning (~35% of videos affected) | Done |
| Phase 5 rerun | Agent tokens rebuilt on fps-fixed Phase 3/4 (19,076 videos) | Done |
| Phase 6 v5 | Fresh merge on fixed agent tokens + snac/caption/speech | Done |
| **Phase 7 v6** | **Flatten with wrapper tokens restored on all modality blocks (this dataset)** | **Done** |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("EmpathicRobotics/FineVideo-Phase7-Flattened", streaming=True)

for sample in ds["train"]:
    text = sample["text"]
    # text contains: headers + per-chunk token sequence
    # tokens: <seed2_N>, <cosmos_N>, <fps_30>, <pelvis>..., <snac_N>...
    print(text[:200])
    break
```

## Version History

| Version | Date | Records | Total tokens | Key change |
|---------|------|---------|-------------|------------|
| v1 | Mar 2026 | 69,844 | ~1.35B | Agent only, 99% AVC-LM drop, 90% cosmos drop |
| v2 | Jun 2026 | 69,844 | ~1.35B | 100% AVC-LM drop, 50% cosmos drop |
| v3 | Jul 2, 2026 | 371,888 | ~5.52B | Added SNAC, expanded filter to agent OR snac |
| v4 | Jul 2, 2026 | 371,888 | 5.217B | Fixed per-chunk temporal ordering, speech in headers |
| v5 | Jul 17, 2026 | 371,888 | 5.256B | Added inline `<caption>`/`<speech>` language anchors at modality-transition points (+0.740%) |
| **v6** | **Jul 21, 2026** | **371,892** | **5.443B** | **Rebuilt agent tokens on fps-mismatch-fixed Phase 3/4 (+8.3% agent blocks); restored `<seed2>`/`<cosmos>`/`<agent>`/`<snac>` wrapper tokens (+3.6% total tokens)** |

## Citation

Part of the FineVideo-VLA project. If you use this data, please cite:

```bibtex
@misc{Farré2024FineVideo,
  title={FineVideo},
  author={Farré, Miquel and Marafioti, Andi and Tunstall, Lewis and Von Werra, Leandro and Wolf, Thomas},
  year={2024},
  howpublished={\url{https://huggingface.co/datasets/HuggingFaceFV/finevideo}},
}
```

## License

Apache 2.0
