Vendored from NVIDIA's [Cosmos-Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer),
Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES, licensed under Apache-2.0
(see individual file headers). Vendored here (2026-07-23) so
`tools/decode/decode_cosmos.py` doesn't require the internal cluster's
`prototype/` directory (not part of this repo) -- only inference code
(`video_lib.py` + its direct dependencies) is included, not training code.
Model checkpoints are downloaded separately from
[nvidia/Cosmos-Tokenizer-DV8x16x16](https://huggingface.co/nvidia/Cosmos-Tokenizer-DV8x16x16)
on first use, not vendored here.
