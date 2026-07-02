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
  - megatron-lm
  - pretraining
language:
  - en
size_categories:
  - 100K<n<1M
---

# FineVideo-Phase7-Flattened — Megatron-LM Multimodal Pretraining Dataset (v4)

## Overview

This is the **final, training-ready** flattened dataset from the FineVideo-VLA pipeline. Each record is a single `{"text": "..."}` JSON line containing interleaved multimodal tokens — ready for Megatron-LM tokenization and LLM pretraining.

Four token modalities are interleaved **per 8-frame chunk** in temporal order:

- **Seed2** — 1 FPS semantic keyframe tokens (vocab: 8192), kept at 100%
- **Cosmos** — every 8-frame spatial video tokens (vocab: 64,000), kept at 50%
- **Agent** — adaptive PCHIP 3D human pose tokens (17 joints, variable control points per joint)
- **SNAC** — audio tokens in listen format (~10 tokens per 8-frame chunk, vocab: 12,288)

Source: ~40,000 YouTube videos from [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo).

**Key property:** Every record contains at least one of `<agent>` or `<snac>` tokens. Records without either are discarded.

## Dataset Statistics (v4, Jul 2, 2026)

| Metric | Value |
|--------|-------|
| Total records | **371,888** |
| Full-chain records (agent + snac) | 69,811 (18.8%) |
| Partial-chain records (snac only) | 302,044 (81.2%) |
| Total shards | 160 |
| Train shards | 152 |
| Test shards | 8 |
| Split ratio | 95/5 (seed 42) |
| Compression | gzip level 5 |

### Token counts

| Modality | Tokens | % |
|----------|--------|---|
| seed2 | 332,592,448 | 6.4% |
| cosmos | 3,882,981,800 | 74.4% |
| agent | 637,924,374 | 12.2% |
| snac | 363,029,331 | 7.0% |
| **TOTAL** | **5,216,527,953** | **5.217B** |

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

## Data Format

Each line is a JSON object with a single `text` field:

```json
{
  "text": "### Title: Launching\n### Context: A video showcasing diverse vocation paths...\n### Keywords: educational, informative\n### Speech: turn left and walk forward\n<seed2_6750> <seed2_680> ... <cosmos_18232> <cosmos_41007> ... <fps_30> <pelvis> <pelvis_t_0> <pelvis_x_128> ... </pelvis> <r_hip> ... </r_hip> ... <snac_132247> <snac_132788> ..."
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
chunk 0:  [<seed2_N>...] [<cosmos_N>...] [<fps_30> <pelvis> ... </r_wrist>] [<snac_N>...]
chunk 1:               [<cosmos_N>...] [<fps_30> <pelvis> ... </r_wrist>] [<snac_N>...]
chunk 2:  [<seed2_N>...] [<cosmos_N>...] [<fps_30> <pelvis> ... </r_wrist>] [<snac_N>...]
...
```

- seed2 appears at 1fps keyframe chunks (every ~3.75 chunks at 30fps)
- cosmos present at 50% of chunks (random per chunk)
- agent present only at chunks with a detected person
- snac present at ~100% of chunks (audio available for most activities)

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

Use **[EmpathicRobotics/tokenizer-vla-adaptive-v2](https://huggingface.co/EmpathicRobotics/tokenizer-vla-adaptive-v2)** (156,505 vocab) for this dataset — it includes SNAC tokens that are absent in v1 (144,215 vocab).

All VLA tokens are registered via `add_tokens(special_tokens=True)` — the BPE tokenizer treats every VLA token as atomic and never splits them.

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("EmpathicRobotics/tokenizer-vla-adaptive-v2")
tok.encode("<seed2_1137>")      # -> single token
tok.encode("<pelvis_x_128>")    # -> single token
tok.encode("<snac_132247>")     # -> single token
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
| **Total** | | **156,505** |

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
| **Phase 7 v4** | **Per-chunk temporal flatten + modality dropout (this dataset)** | **Done** |

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
| **v4** | **Jul 2, 2026** | **371,888** | **5.217B** | **Fixed per-chunk temporal ordering, speech in headers** |

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
