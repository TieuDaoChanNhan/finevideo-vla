# qwen3_1.7b_vla_v2 any-mode-to-any-mode eval (REAL pretrain records) -- 2026-07-22

Model: `/e/project1/reformo/nguyen38/output_vla/qwen3_1.7b_vla_v2/hf/iter_0007632`

Requested directly by Huu in the 2026-07-22 Discord thread (REPORT.md #33, item 9).

| Test | Category | Mode | seed2 | cosmos | snac | agent |
|---|---|---|---|---|---|---|
| [01_chemical_bond_real_greedy](01_chemical_bond_real_greedy/input_output.txt) | image_to_text | greedy | PASS: decoded_seed2.png | FAIL: only 138 cosmos tokens, need >= 200 for 1 chunk | PASS: decoded_snac.wav | - |
| [01_chemical_bond_real_sample](01_chemical_bond_real_sample/input_output.txt) | image_to_text | sample | PASS: decoded_seed2.png | FAIL: only 149 cosmos tokens, need >= 200 for 1 chunk | - | - |
| [02_woman_walking_real_greedy](02_woman_walking_real_greedy/input_output.txt) | any_to_any | greedy | PASS: decoded_seed2_0.png, decoded_seed2_1.png | PASS: 1/1 chunks: decoded_cosmos_chunk0.mp4 | PASS: decoded_snac.wav | - |
| [02_woman_walking_real_sample](02_woman_walking_real_sample/input_output.txt) | any_to_any | sample | PASS: decoded_seed2.png | PASS: 1/1 chunks: decoded_cosmos_chunk0.mp4 | PASS: decoded_snac.wav | - |
| [03_roleplay_speech_real_greedy](03_roleplay_speech_real_greedy/input_output.txt) | text_to_audio | greedy | - | - | - | - |
| [03_roleplay_speech_real_sample](03_roleplay_speech_real_sample/input_output.txt) | text_to_audio | sample | - | - | - | - |
