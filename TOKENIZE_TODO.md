# Megatron Tokenization TODO (JUWELS side)

**Read this first if you're a Claude session running on JUWELS after a fresh `git pull` of this
repo.** All flat/pre-tokenize JSONL sources now live on `/p` (JUWELS storage) so this can run
there. The actual tokenize infra (`mv_preprocess_data.py` + sbatch templates) is **not** part of
this git repo — it's shared JUWELS infra at `/p/data1/mmlaion/nguyen38/mv-scale/`, already
reachable once you're on JUWELS. Written 2026-07-21 after moving `FineVideo-VLA` v6 from `/e`
(JUPITER, where it was built — JUPITER compute nodes can't see `/p`) over to `/p` for exactly
this handoff. See `feedback_data_storage_location` reasoning in project memory / `PROGRESS_VI.md`'s
2026-07-21 entries if you need the full "why /e for compute, /p for tokenize" story.

**Tokenizer for everything below:** `/p/data1/mmlaion/shared/vla/tokenizer_vla_qwen3` (Qwen3 base +
full VLA token set — seed2/cosmos/avclm/agent/snac/caption/speech — 257,901 vocab). Confirmed by
reading the actual sbatch templates (not the possibly-stale `tokenizer_vla_adaptive_v2` mentioned in
older PROGRESS.md entries) — every existing `tokenize_*.sbatch` in `mv-scale/` uses qwen3
consistently. Use the same one for anything new so token IDs stay compatible for mixing at train
time.

## 1. Needs a fresh tokenize run (source data changed or was never tokenized)

| Dataset | Source (flat JSONL, on `/p`) | Sbatch template to copy | Notes |
|---|---|---|---|
| **FineVideo-VLA v6** | `/p/data1/mmlaion/shared/vla/finevideo_v6_flat/` (160 files, 74GB, 371,892 records, 5.443B tokens) | Copy `mv-scale/tokenize_finevideo_v5.sbatch` → `tokenize_finevideo_v6.sbatch`. Update `INPUT` to the path above, `OUTPUT_PREFIX` to `${OUTPUT_DIR}/finevideo_v6` | Supersedes the existing `tokenized_output/finevideo_v5/` (stale — built 18/07 before the Phase 3/4 fps-mismatch fix + wrapper-token fix). Don't reuse that output. |
| **OmniVideo-100K (video)** | `/p/data1/mmlaion/shared/vla/omnivideo_100k_final/hf_upload/` (the actual final, wrapper-token-fixed dataset uploaded to HF as `EmpathicRobotics/omnivideo-100k-final`) | Copy `mv-scale/tokenize_omnivideo_100k_video.sbatch`. **Change `INPUT`** — it currently points to `omnivideo_100k_video_flattened`, which predates the 21/07 wrapper-token regen. Point it at `omnivideo_100k_final/hf_upload/train` (or wherever the flat pre-compression `.jsonl` sits — check `data_prep/omnivideo_100k/phase7_finalize_omnivideo.py` for the exact pre-gzip path if `hf_upload/` is already gzipped) | Existing `tokenized_output/omnivideo_100k_video/` (1.8G, built 19/07 06:26) is stale for the same reason as finevideo_v5. |
| **synth-llava** | `/p/data1/mmlaion/shared/vla/synth_llava_flat/` (151 files, 603,999 records, 19.3M seed2 tokens + captions) | **No sbatch exists yet** — write `tokenize_synth_llava.sbatch` by copying `tokenize_mv_omni.sbatch` (single flat-JSONL-dir input, no multi-source merge needed) | Never tokenized. Already uploaded to HF as `EmpathicRobotics/synth-llava` in flat `{"text":...}` form, ready as-is. |
| **emotional-roleplay** | `/p/data1/mmlaion/shared/vla/laion_emotional_roleplay/flattened/` (14 files, `roleplay_snac_flat_*.jsonl`) | **No sbatch exists yet** — write `tokenize_roleplay.sbatch`, same pattern as above | Never tokenized. Already uploaded to HF as `EmpathicRobotics/emotional-roleplay-finetuning-dataset-flattened`. |

## 2. Probably still valid — verify before assuming, don't blindly retokenize

| Dataset | Existing tokenized output | Why it's probably fine |
|---|---|---|
| MV-Omni | `tokenized_output/mv_omni/` (76G, 18/07) | Source (`mv_omni_converted/`, SNAC conversion) finalized 27/06, untouched since — not affected by the FineVideo-specific fps-mismatch/wrapper-token fixes. |
| OmniVideo-100K (QA) | `tokenized_output/omnivideo_100k_qa/` (120M, 18/07) | QA text wasn't touched by the wrapper-token fix (that only affected seed2/cosmos/agent/snac blocks) — but its source dir (`omnivideo_100k_flat/`) may or may not have been rebuilt since. Spot-check record count against `omnivideo_100k_final` before trusting it as-is. |
| RoboVQA | `tokenized_output/robovqa/` (228M, 18/07) | Exists, but RoboVQA's own flattening was left intentionally incomplete (`flatten_text.py` dropped `video_id`, `extract_frames.py` only 130/184 shards) per earlier PROGRESS.md entries — treat this tokenized output as exploratory, not necessarily "the" RoboVQA data to mix in. Confirm intent before using in a training mix. |

## 3. Don't touch — belongs to already-trained models

`tokenized_output/vla_25b/` and `tokenized_output/vla_adaptive/` (+ the `/e` mirror at
`vla_adaptive_tokenized/`) are what the two existing trained models
(`vla-1.7b-pab-spline-25b-test`, `vla-1.7b-pab-spline-adaptive`) actually used. Leave as-is.

## 4. After tokenizing

Still open, not this file's scope but flagged so it isn't forgotten:
- Decide the mix ratio across sources (deliberately deferred to train time, per Van Khue's
  instruction — "mấy cái quyết định drop out để sau đi").
- Eval protocol (MPJPE / modality-transition / instruction-following) is still undefined —
  `REPORT.md`'s "Pre-training Blockers" section, item 3, still open as of 21/07/2026.
- Training config (`oellm-autoexp/config/experiments/nguyen38/vla_adaptive.yaml`) still points at
  the old 2.84B-token `.bin/.idx` — will need a new config once the new tokenize runs land.
