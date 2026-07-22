# qwen3_1.7b_vla_v2 media eval -- 2026-07-22

Model: `/e/project1/reformo/nguyen38/output_vla/qwen3_1.7b_vla_v2/hf/iter_0007632`

| Test | Category | Mode | seed2 | cosmos | snac | agent |
|---|---|---|---|---|---|---|
| [01_image_caption_greedy](01_image_caption_greedy/input_output.txt) | continuation | greedy | PASS: decoded_seed2.png | - | - | - |
| [01_image_caption_sample](01_image_caption_sample/input_output.txt) | continuation | sample | PASS: decoded_seed2.png | - | - | - |
| [02_agent_continuation_greedy](02_agent_continuation_greedy/input_output.txt) | continuation | greedy | FAIL: RuntimeError: The size of tensor a (3) must match the size of tensor b (32) at non-singleton dimension 1 | FAIL: only 2 cosmos tokens, need >= 200 for 1 chunk | PASS: decoded_snac.wav | PASS: decoded_agent_pose.mp4 |
| [02_agent_continuation_sample](02_agent_continuation_sample/input_output.txt) | continuation | sample | FAIL: RuntimeError: The size of tensor a (3) must match the size of tensor b (32) at non-singleton dimension 1 | FAIL: only 46 cosmos tokens, need >= 200 for 1 chunk | PASS: decoded_snac.wav | FAIL: ValueError: The length of `y` along `axis`=0 doesn't match the length of `x` |
| [03_roleplay_speech_greedy](03_roleplay_speech_greedy/input_output.txt) | continuation | greedy | PASS: decoded_seed2.png | PASS: 1/1 chunks: decoded_cosmos_chunk0.mp4 | PASS: decoded_snac.wav | - |
| [03_roleplay_speech_sample](03_roleplay_speech_sample/input_output.txt) | continuation | sample | PASS: decoded_seed2.png | PASS: 1/1 chunks: decoded_cosmos_chunk0.mp4 | FAIL:  | - |
| [04_seed2_from_scratch_greedy](04_seed2_from_scratch_greedy/input_output.txt) | from_scratch | greedy | PASS: decoded_seed2.png | FAIL: only 130 cosmos tokens, need >= 200 for 1 chunk | PASS: decoded_snac.wav | - |
| [04_seed2_from_scratch_sample](04_seed2_from_scratch_sample/input_output.txt) | from_scratch | sample | PASS: decoded_seed2.png | FAIL: only 108 cosmos tokens, need >= 200 for 1 chunk | PASS: decoded_snac.wav | - |
| [05_cosmos_from_scratch_greedy](05_cosmos_from_scratch_greedy/input_output.txt) | from_scratch | greedy | - | FAIL: only 125 cosmos tokens, need >= 200 for 1 chunk | - | - |
| [05_cosmos_from_scratch_sample](05_cosmos_from_scratch_sample/input_output.txt) | from_scratch | sample | - | FAIL: only 125 cosmos tokens, need >= 200 for 1 chunk | - | - |
| [06_agent_from_scratch_greedy](06_agent_from_scratch_greedy/input_output.txt) | from_scratch | greedy | PASS: decoded_seed2.png | FAIL: only 138 cosmos tokens, need >= 200 for 1 chunk | - | PASS: decoded_agent_pose.mp4 |
| [06_agent_from_scratch_sample](06_agent_from_scratch_sample/input_output.txt) | from_scratch | sample | PASS: decoded_seed2.png | FAIL: only 90 cosmos tokens, need >= 200 for 1 chunk | PASS: decoded_snac.wav | PASS: decoded_agent_pose.mp4 |
| [07_full_chain_from_scratch_greedy](07_full_chain_from_scratch_greedy/input_output.txt) | full_chain | greedy | FAIL: RuntimeError: The size of tensor a (96) must match the size of tensor b (32) at non-singleton dimension 1 | PASS: 3/3 chunks: decoded_cosmos_chunk0.mp4; decoded_cosmos_chunk1.mp4; decoded_cosmos_chunk2.mp4 | PASS: decoded_snac.wav | - |
| [07_full_chain_from_scratch_sample](07_full_chain_from_scratch_sample/input_output.txt) | full_chain | sample | FAIL: RuntimeError: The size of tensor a (96) must match the size of tensor b (32) at non-singleton dimension 1 | PASS: 3/3 chunks: decoded_cosmos_chunk0.mp4; decoded_cosmos_chunk1.mp4; decoded_cosmos_chunk2.mp4 | PASS: decoded_snac.wav | - |
