import os
import json
import random
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Suppress warnings for clean cluster logging
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =====================================================================
# PATH CONFIGURATION
# =====================================================================
MODEL_PATH = "/e/project1/reformo/nguyen38/output_vla/vla_25b_test/hf/iter_0006000"
DATA_DIR = "/e/project1/reformo/nguyen38/prototype/FineVideo-VLA/target_vla_25b"
# =====================================================================

def get_raw_training_sample(data_dir):
    """Automatically finds the first JSONL shard and extracts a valid VLA sample."""
    jsonl_files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl')]
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in target directory: {data_dir}")
        
    target_file = os.path.join(data_dir, jsonl_files[0])
    print(f"📡 Reading raw training shard: {os.path.basename(target_file)}")
    
    samples = []
    with open(target_file, 'r', encoding='utf-8') as f:
        # Scan the first 1000 rows to find an action-heavy sequence
        for _ in range(1000):
            line = f.readline()
            if not line:
                break
            try:
                data = json.loads(line)
                text = data.get('text', '')
                if '<seed2' in text or '<cosmos' in text:
                    samples.append(text)
            except:
                continue
                
    if not samples:
        raise ValueError("No valid sequences containing action tokens found in the sample block.")
        
    return random.choice(samples)

def main():
    # 1. Fetch raw training ground truth
    full_sequence = get_raw_training_sample(DATA_DIR)
    
    # Locate the action boundary token to split into prompt and expected suffix
    split_token = '<seed2' if '<seed2' in full_sequence else '<cosmos'
    split_idx = full_sequence.find(split_token)
    
    # We prime the prompt with the text and the initial few characters of the first token
    # to force the model to instantly latch onto the sequence completion distribution
    prompt_prefix = full_sequence[:split_idx + 15]
    expected_suffix = full_sequence[split_idx + 15:]

    # 2. Initialize Model and Tokenizer
    print("⏳ Initializing VLA 1.7B Backbone & Expanded Vocabulary...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    inputs = tokenizer(prompt_prefix, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Execute Greedy Generation (Deterministic Evaluation)
    print("🚀 Running Sequence Replication Test (Greedy Decoding, Temp=0.0)...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50,  # Generate up to 50 tokens for deeper sequence verification
            do_sample=False,    # Disables random sampling
            pad_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    generated_suffix = output_text[len(prompt_prefix):]

    # 4. Clean Output Presentation for Visual Inspection
    print("\n" + "="*80)
    print("                   VLA SEQUENCE REPLICATION TEST LOG")
    print("="*80)
    print(f"PROMPT INPUT   :\n{prompt_prefix}\n")
    print(f"GROUND TRUTH   :\n...{expected_suffix[:150]}...\n")
    print(f"MODEL GENERATED:\n...{generated_suffix.strip()}...\n")
    print("="*80)
    print("💡 Note: Review the output sequence above for numerical structure and token overlap.")

if __name__ == "__main__":
    main()