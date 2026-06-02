import os
import sys
import torch

# 1. Point HF cache to scratch (matches the offline runtime config)
SCRATCH_CACHE_DIR = "/p/scratch/laionize/nguyen38/hf_cache"
os.environ["HF_HOME"] = SCRATCH_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = SCRATCH_CACHE_DIR

# 2. Ensure network is enabled (remove offline flags if stuck)
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.pop("HF_DATASETS_OFFLINE", None)

print("🌐 Downloading all Seed2 weights and configs to scratch cache...")

# 3. Import and init Seed2 to force it to download all dependencies
sys.path.append("./seed2")
try:
    from seed2_tokenizer import Seed2Tokenizer

    # Forces download of all missing weights, configs, and sub-models
    # into SCRATCH_CACHE_DIR
    tokenizer = Seed2Tokenizer.from_pretrained("./seed2", torch_dtype=torch.float16)

    print("✅ Done. All Seed2 cache files are now in scratch.")
    print("💡 You can now move to a Booster node (offline) and run benchmark_pipeline.py.")

except Exception as e:
    print(f"❌ Error: {e}")
