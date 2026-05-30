import json
import os

def expand_vocab(input_vocab_path="vocab.json", output_vocab_path="vocab_expanded.json"):
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
        "<start_cosmo>", "</start_cosmo>",
        "<start_avclm>", "</start_avclm>",
    ]

    # 2. Auto-generate numeric value tokens
    regular_tokens = []
    regular_tokens.extend([f"<agent_{i}>" for i in range(256)])
    regular_tokens.extend([f"<avclm_{i}>" for i in range(8192)])
    regular_tokens.extend([f"<seed2_{i}>" for i in range(8192)])
    regular_tokens.extend([f"<cosmos_{i}>" for i in range(64000)])

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
