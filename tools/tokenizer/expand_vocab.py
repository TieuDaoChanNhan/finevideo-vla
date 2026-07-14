import json
import os

def expand_vocab(input_vocab_path="vocab/vocab.json", output_vocab_path="vocab/vocab_expanded.json"):
    if not os.path.exists(input_vocab_path):
        print(f"❌ {input_vocab_path} not found. Download the base vocab from EleutherAI/gpt-neox-20b.")
        return

    with open(input_vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    current_max_id = max(vocab.values())
    next_id = current_max_id + 1

    # 1. Wrapper tokens and special tokens
    special_tokens = [
        "<seed2>", "</seed2>",
        "<cosmos>", "</cosmos>",
        "<avc_lm>", "</avc_lm>",
        "<agent>", "</agent>",
        "<caption>", "</caption>",
        "<speech>", "</speech>",
        "<start_cosmo>", "</start_cosmo>",
        "<start_avclm>", "</start_avclm>",
    ]

    # 2. Auto-generate numeric value tokens
    regular_tokens = []
    regular_tokens.extend([f"<agent_{i}>" for i in range(256)])
    regular_tokens.extend([f"<avclm_{i}>" for i in range(8192)])
    regular_tokens.extend([f"<seed2_{i}>" for i in range(8192)])
    regular_tokens.extend([f"<cosmos_{i}>" for i in range(64000)])

    # 3. Phase 5 adaptive PCHIP per-joint tokens (named joints)
    # <fps_N>             — frame-rate prefix (1–60)
    # <{joint}>/</{joint}> — per-joint wrapper (17 joints)
    # <{joint}_x_N>       — x coordinate, uint8 [0,255]
    # <{joint}_y_N>       — y coordinate, uint8 [0,255]
    # <{joint}_z_N>       — z coordinate, uint8 [0,255]
    # <{joint}_t_N>       — frame index within 8-frame window [0,7]
    joint_names = [
        "pelvis", "r_hip", "r_knee", "r_ankle",
        "l_hip", "l_knee", "l_ankle",
        "spine", "thorax", "nose", "head_top",
        "l_shoulder", "l_elbow", "l_wrist",
        "r_shoulder", "r_elbow", "r_wrist",
    ]
    regular_tokens.extend([f"<fps_{i}>" for i in range(1, 61)])
    for name in joint_names:
        special_tokens.extend([f"<{name}>", f"</{name}>"])
        regular_tokens.extend([f"<{name}_x_{n}>" for n in range(256)])
        regular_tokens.extend([f"<{name}_y_{n}>" for n in range(256)])
        regular_tokens.extend([f"<{name}_z_{n}>" for n in range(256)])
        regular_tokens.extend([f"<{name}_t_{n}>" for n in range(8)])

    all_new_tokens = special_tokens + regular_tokens

    added_count = 0
    for token in all_new_tokens:
        if token not in vocab:
            vocab[token] = next_id
            next_id += 1
            added_count += 1

    with open(output_vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"✅ Added {added_count} new tokens.")
    print(f"✅ Total tokens in vocabulary: {len(vocab)}. Max ID: {next_id - 1}")
    print(f"👉 Saved to: {output_vocab_path}")

if __name__ == "__main__":
    expand_vocab()
