# tools/ — Standalone Utilities

Utilities that support the pipeline but aren't pipeline phases themselves — they run independently,
on demand. Grouped by what they do:

```
tools/
├── upload/       HuggingFace upload scripts + dataset cards
├── tokenizer/    Vocab expansion, tokenizer build, verification
├── inventory/    Token/dataset counting, overlap checks, flattened-data validation
├── eval/         Model sanity checks, agent-token decoding
├── visualize/    Skeleton/pose rendering for visual QA
├── analysis/     One-off compression/tradeoff analyses (results captured in PROGRESS.md)
└── extract/      Small per-video data extraction helpers
```

Deprecated/one-off scripts that already served their purpose live in `../archive/tools_deprecated/`,
not here.

---

## `upload/` — HuggingFace Upload

| Script | Purpose |
|--------|---------|
| `upload_tokenizer.py` | Create + upload the v1 VLA tokenizer (GPT-NeoX-20b + 93,938 tokens via `add_tokens`) to [EmpathicRobotics/tokenizer-vla-adaptive](https://huggingface.co/EmpathicRobotics/tokenizer-vla-adaptive) |
| `upload_tokenizers_v2.py` | Upload the v2 (+SNAC) and Qwen3 tokenizers built by `tokenizer/build_tokenizers.py` |
| `upload_flattened_hf.py` | Compress + upload the flattened Megatron-LM dataset to [EmpathicRobotics/FineVideo-Phase7-Flattened](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase7-Flattened) |
| `upload_vla_agent_hf.py` | Upload merged adaptive-PCHIP shards to [EmpathicRobotics/FineVideo-Phase5-AgentTokens](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase5-AgentTokens) |
| `upload_phase4_hf.py` | Upload Phase 4 YOLO-cleaned pose data to [EmpathicRobotics/FineVideo-Phase4-YOLOPose](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase4-YOLOPose) |
| `upload_3d_npy_to_hf.py` | Upload raw 3D pose numpy arrays as parquet shards |
| `upload_parquet_hf.py` | Upload rebuilt parquet shards (resume-safe) |
| `upload_vla_model.py` | Upload the current trained model checkpoint (v2, `vla-1.7b-pab-spline-adaptive`) with a full model card |
| `push_dataset_cards.py` | Push the `*_dataset_card.md` files in this folder as README.md to their HF dataset repos — reads them via a path relative to its own location, so keep it in the same folder as the cards |

```bash
export HF_TOKEN='hf_...'
python tools/upload/upload_flattened_hf.py          # compress + upload
python tools/upload/upload_flattened_hf.py --skip-upload   # compress only
python tools/upload/upload_flattened_hf.py --skip-compress # upload only (reuse compressed)
```

---

## `tokenizer/` — Vocabulary & Tokenizer

| Script | Purpose |
|--------|---------|
| `expand_vocab.py` | Extend GPT-NeoX-20b vocab (`../../vocab/gpt-neox-20b-vocab.json`) with all VLA tokens → `../../vocab/vocab_expanded.json` (JSON lookup only, not a working tokenizer) |
| `build_tokenizers.py` | Build the real HuggingFace tokenizers via `add_tokens(special_tokens=True)` — GPT-NeoX v2 (+SNAC) and Qwen3 variants |
| `check_vocab.py` | Verify vocab size/token ranges (rounds to nearest 128 for Megatron) |

```bash
python tools/tokenizer/expand_vocab.py
python tools/tokenizer/build_tokenizers.py --mode all
python tools/tokenizer/check_vocab.py
```

---

## `inventory/` — Token & Dataset Counting

Core module: **`data_inventory.py`** — regex-based VLA token counter (`PATTERNS`, `count_tokens()`),
HF streaming download helpers (`_hf_token()`, `hf_url()`), and an atomic-checkpoint pattern that the
other scripts here import and reuse. Not a one-off script — treat it as a shared library.

| Script | Purpose |
|--------|---------|
| `data_inventory.py` | Counts tokens per modality across FineVideo-VLA + external HF sources (valid_with_seed, valid_snac). Checkpoint: `inventory_checkpoint_v2.json`. Chart: `data_inventory_charts.png` |
| `peek_multimodal.py` | Structural probe for a new HF dataset folder — streams the first few records/members (no full download) to discover format before writing real parsing logic |
| `count_multimodal_tokens.py` | Sample-based streaming token counter for a new HF dataset (imports `data_inventory.py` + `peek_multimodal.py`) — caps bytes read per file, extrapolates to full size, resumable checkpoint |
| `check_dataset_overlap.py` | Compares video-ID sets across dataset shards to detect double-counting risk. Results: `dataset_overlap_results.json` |
| `check_flattened_data.py` | Validates a flattened Megatron JSONL file — JSON integrity, token coverage, structural completeness |
| `setup_env_inventory.sh` | Creates the minimal venv these scripts need (`requests`, `tqdm` — no torch/datasets required) |

**Coupling note:** `count_multimodal_tokens.py` imports from `data_inventory.py` and `peek_multimodal.py`
via same-directory sibling import (not a package import) — all three must stay in this folder together.

```bash
python tools/inventory/data_inventory.py
python tools/inventory/peek_multimodal.py --only some_file.jsonl.gz
python tools/inventory/count_multimodal_tokens.py --sample-mb 75
python tools/inventory/check_flattened_data.py
```

---

## `eval/` — Model Sanity Checks

| Script | Purpose |
|--------|---------|
| `decode_agent_tokens.py` | Decodes agent (PCHIP) tokens back to 3D joint coordinates — reused as a library by `eval_vla_sanity.py` |
| `eval_vla_sanity.py` | Runs generation sanity checks against a trained checkpoint (token atomicity, agent-block completion, decodability) |

**Coupling note:** `eval_vla_sanity.py` imports `decode_agent_tokens.py` as a same-directory sibling — keep them together.

```bash
python tools/eval/decode_agent_tokens.py --seed 42
python tools/eval/eval_vla_sanity.py
```

---

## `visualize/` — Visual QA

| Script | Purpose |
|--------|---------|
| `visualize_skeleton_sidebyside.py` | Renders two skeleton sequences side by side (used to compare pose-pipeline output vs. raw/other sources) |
| `render_filtered_skeleton.py` | Renders a skeleton-only overlay video from a states JSONL |

```bash
python tools/visualize/render_filtered_skeleton.py \
    --video-real videos/sample.mp4 \
    --jsonl outputs/states_jsonl/sample_states.jsonl \
    --output outputs/skeleton.mp4
```

---

## `analysis/` — Compression / Tradeoff Analyses

One-off parametrized analysis scripts. Their specific runs are complete and the results are captured in
`PROGRESS.md`, but they're reusable (CLI args) if the compression scheme changes again.

| Script | Purpose |
|--------|---------|
| `analyze_pchip_compression.py` | Measures token savings of adaptive PCHIP vs. fixed 8-CP |
| `analyze_cp_tradeoff.py` | Validates the targeted 1-CP compression proposal (static-joint collapse) against reconstruction error |

---

## `extract/` — Data Extraction Helpers

| Script | Purpose |
|--------|---------|
| `extract_fps.py` | Reads native fps for all videos → `fps_lookup.json` |
| `fetch_data.py` | Fetches video data from the HuggingFace FineVideo dataset |
| `rebuild_parquet_fps.py` | Rebuilds parquet shards with 30fps poses + an fps column |

---

## Environment

Most scripts here run under the 3D pose pipeline environment:
```bash
source ../setup_motionbert.sh
```
`upload/` scripts additionally require `huggingface_hub` (included in `env_motion_final`).
`inventory/` scripts only need `requests`/`tqdm` — see `inventory/setup_env_inventory.sh` for a
minimal standalone venv (useful when working off-cluster, e.g. a local Windows machine).
