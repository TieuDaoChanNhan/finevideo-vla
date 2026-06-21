# tools — Standalone Utilities

Standalone scripts for vocabulary management, dataset flattening, quality checks,
data uploads, and visualization. These are not pipeline phases — they can be run
independently as needed.

---

## HuggingFace Upload

| Script | Purpose |
|--------|---------|
| `upload_tokenizer.py` | Create + upload the VLA tokenizer (GPT-NeoX-20b + 93,938 tokens via `add_tokens`) to [EmpathicRobotics/tokenizer-vla-adaptive](https://huggingface.co/EmpathicRobotics/tokenizer-vla-adaptive) |
| `upload_flattened_hf.py` | Compress + upload flattened adaptive Megatron-LM dataset to [EmpathicRobotics/FineVideo-Phase7-Flattened](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase7-Flattened). 160 shards → 152 train / 8 test (95/5, seed 42), gzip level 5 |
| `upload_vla_agent_hf.py` | Upload adaptive PCHIP merged shards to [EmpathicRobotics/FineVideo-Phase5-AgentTokens](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase5-AgentTokens) |
| `upload_phase4_hf.py` | Upload Phase 4 YOLO-cleaned pose data to [EmpathicRobotics/FineVideo-Phase4-YOLOPose](https://huggingface.co/datasets/EmpathicRobotics/FineVideo-Phase4-YOLOPose) |
| `upload_3d_npy_to_hf.py` | Upload raw 3D pose numpy arrays as parquet shards |
| `upload_parquet_hf.py` | Upload rebuilt parquet shards (resume-safe) |
| `rename_hf_repos.py` | Rename HF repos to phase-numbered naming convention |
| `cleanup_hf_repo.py` | Delete leftover `train/` and `test/` folders from HF repos |

All upload scripts require `HF_TOKEN`:
```bash
export HF_TOKEN='hf_...'
python tools/upload_flattened_hf.py          # compress + upload
python tools/upload_flattened_hf.py --skip-upload   # compress only
python tools/upload_flattened_hf.py --skip-compress  # upload only (reuse compressed)
```

---

## Vocabulary & Tokenizer

| Script | Purpose |
|--------|---------|
| `expand_vocab.py` | Extend GPT-NeoX-20b vocab (`vocab/vocab.json`) with all VLA tokens. Output: `vocab/vocab_expanded.json` (JSON lookup only) |
| `upload_tokenizer.py` | Create a proper HuggingFace tokenizer using `add_tokens(special_tokens=True)` and upload to [EmpathicRobotics/tokenizer-vla-adaptive](https://huggingface.co/EmpathicRobotics/tokenizer-vla-adaptive). This is the tokenizer that must be used for Megatron-LM tokenization — the base GPT-NeoX-20b tokenizer will split VLA tokens into sub-pieces |
| `check_vocab.py` | Verify expanded vocab size and token ranges (rounds to nearest 128 for Megatron) |

```bash
python tools/expand_vocab.py       # generate vocab JSON
python tools/upload_tokenizer.py   # create + upload HF tokenizer (requires HF_TOKEN)
python tools/check_vocab.py        # verify
```

---

## Data Inspection & Validation

| Script | Purpose |
|--------|---------|
| `check_flattened_data.py` | Validate flattened Megatron files — checks JSON integrity, token coverage, and structural completeness |
| `decode_agent_tokens.py` | Decode agent uint8 tokens back to 3D joint coordinates. Reads from a `final_vla_rank_*.jsonl` file |
| `extract_sample.py` | Extract sample records from dataset files for inspection |
| `extract_fps.py` | Read native fps for all videos → `fps_lookup.json` |
| `fetch_data.py` | Fetch video data from HuggingFace FineVideo dataset |
| `rebuild_parquet_fps.py` | Rebuild parquet shards with 30fps poses + fps column |
| `render_filtered_skeleton.py` | Render a skeleton overlay video from states JSONL |

```bash
# Decode a random agent token block
python tools/decode_agent_tokens.py --seed 42

# Validate flattened dataset
python tools/check_flattened_data.py

# Render skeleton video
python tools/render_filtered_skeleton.py \
    --video-real videos/sample.mp4 \
    --jsonl outputs/states_jsonl/sample_states.jsonl \
    --output outputs/skeleton.mp4
```

---

## Environment

Most tools run under the 3D pose pipeline environment:
```bash
source setup_motionbert.sh
```

Upload scripts additionally require `huggingface_hub` (included in `env_motion_final`).
