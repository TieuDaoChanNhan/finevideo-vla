# manual_checks/

**Not an automated test suite** — there's no `pytest`/`unittest`/CI here. These are interactive,
print-based scripts a human runs by hand against a real trained checkpoint and real data paths to
sanity-check the pipeline. Run them manually, read the printed output, judge for yourself.

| Script | Purpose |
|--------|---------|
| `check_raw_shards.py` | Prints info about raw Megatron shard files |
| `test_vla_inference.py` | Loads a trained checkpoint, runs `.generate()`, prints decoded output |
| `vla_diff_test.py` | Loads a Megatron `.bin` shard + tokenizer, manually decodes/prints tokens to visually check for BPE fragmentation |

**Path warning:** as of this writing, all three hardcode paths pointing at the **v1 model**
(`vla_25b_test/hf/iter_0006000` — the broken-tokenizer model, deprecated per `../PROGRESS.md`).
Update the `MODEL_PATH`/`model_path` constants before using these against the current v2 model
(`vla-1.7b-pab-spline-adaptive`).

For the equivalent checks against the current model, see `../tools/eval/eval_vla_sanity.py` instead —
that one's kept current.
