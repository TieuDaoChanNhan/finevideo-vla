# Decoder portability test (2026-07-23)

Real output of the 3 now-public-portable decoders in `tools/decode/` and
`tools/eval/`, run on real tokens `vla-1.7b-qwen3-v2` actually generated
(`samples/qwen3_1.7b_vla_v2_eval/*_raw_ids*.txt`). Each ran with **no cluster
access** -- `cosmos_decoded.mp4` specifically went through the
`hf_hub_download("nvidia/Cosmos-Tokenizer-DV8x16x16")` fallback path, not the
internal-cluster checkpoint copy, to prove the path an external evaluator
would actually take.

| File | Decoder | Command |
|---|---|---|
| `cosmos_decoded.mp4` | `tools/decode/decode_cosmos.py` | `--tokens <200 ids> --output cosmos_decoded.mp4` |
| `agent_decoded.json` | `tools/eval/decode_agent_tokens.py` | `--input <agent tokens> --output agent_decoded.json` |
| `snac_decoded.wav` | `tools/decode/decode_snac.py` | `--tokens <ids> --format listen --output snac_decoded.wav` |

seed2 decode intentionally not tested here -- needs a ~2.6GB vendored
checkpoint not yet packaged for public hosting (see
`tools/upload/upload_vla_v2_model.py`'s model card for the full note).
