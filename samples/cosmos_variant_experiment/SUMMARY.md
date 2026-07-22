# Cosmos variant comparison

Source: `/e/project1/reformo/nguyen38/3d-human-pose/videos/good.mp4`, frames 40-47, resized to 448x256 (target_size=256, aspect-preserving)

Reference: [00_original_reference.mp4](00_original_reference.mp4)

| variant | tokens/chunk | ratio vs current | file |
|---|---|---|---|
| Cosmos-Tokenizer-DV8x16x16 | 896 | 1.0x | [Cosmos-Tokenizer-DV8x16x16_896tok.mp4](Cosmos-Tokenizer-DV8x16x16_896tok.mp4) |
| Cosmos-Tokenizer-DV8x8x8 | 3584 | 4.0x | [Cosmos-Tokenizer-DV8x8x8_3584tok.mp4](Cosmos-Tokenizer-DV8x8x8_3584tok.mp4) |
| Cosmos-Tokenizer-DV4x8x8 | 5376 | 6.0x | [Cosmos-Tokenizer-DV4x8x8_5376tok.mp4](Cosmos-Tokenizer-DV4x8x8_5376tok.mp4) |
