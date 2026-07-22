# Cosmos stride (window-duration) experiment

Source: `/e/project1/reformo/nguyen38/3d-human-pose/videos/boxing.mp4`, target_size=256 (fixed, so only stride varies)

| stride | native fps | real time/chunk | tokens | original | decoded |
|---|---|---|---|---|---|
| 1x | 30.0 | 0.233s | 896 | [stride1_00_original_frames.mp4](stride1_00_original_frames.mp4) | [stride1_decoded_896tok.mp4](stride1_decoded_896tok.mp4) |
| 2x | 30.0 | 0.467s | 896 | [stride2_00_original_frames.mp4](stride2_00_original_frames.mp4) | [stride2_decoded_896tok.mp4](stride2_decoded_896tok.mp4) |
| 3x | 30.0 | 0.700s | 896 | [stride3_00_original_frames.mp4](stride3_00_original_frames.mp4) | [stride3_decoded_896tok.mp4](stride3_decoded_896tok.mp4) |
| 4x | 30.0 | 0.933s | 896 | [stride4_00_original_frames.mp4](stride4_00_original_frames.mp4) | [stride4_decoded_896tok.mp4](stride4_decoded_896tok.mp4) |
