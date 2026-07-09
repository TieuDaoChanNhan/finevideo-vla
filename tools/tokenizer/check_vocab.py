import json
import math
import os

def check_megatron_vocab(vocab_path="/p/data1/mmlaion/shared/vla/vocab_expanded.json"):
    if not os.path.exists(vocab_path):
        print(f"❌ Vocabulary file not found: {vocab_path}")
        return

    print(f"🔍 Analysing vocabulary file: {vocab_path}")

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    # 1. Actual token count
    actual_tokens = len(vocab)

    # 2. Largest ID in use
    # (token IDs start at 0, so minimum vocab size = max_id + 1)
    max_id = max(vocab.values())
    required_size = max_id + 1

    # 3. Compute padding (round up to nearest multiple of 128 for Tensor Cores)
    MULTIPLE = 128
    padded_size = math.ceil(required_size / MULTIPLE) * MULTIPLE

    print("=" * 60)
    print("📊 VLA VOCABULARY REPORT FOR MEGATRON TRAINING")
    print("=" * 60)
    print(f"  - Total tokens in file        : {actual_tokens:,}")
    print(f"  - Largest token ID (max_id)   : {max_id:,}")
    print(f"  - Minimum vocab size (max_id+1): {required_size:,}")
    print("-" * 60)
    print(f"🚀 PARAMETER FOR YOUR YAML FILE (multiple of 128):")
    print(f"👉 vocab_size: {padded_size}")
    print("-" * 60)

    padding_added = padded_size - required_size
    if padding_added > 0:
        print(f"💡 Note: {padding_added} dummy padding token(s) added")
        print(f"   to reach {padded_size}, enabling optimal NVIDIA matrix multiplication.")
    else:
        print("💡 Note: Vocabulary size is already a perfect multiple of 128.")
    print("=" * 60)

if __name__ == "__main__":
    check_megatron_vocab()
