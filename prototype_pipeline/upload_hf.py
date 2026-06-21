import os
import gzip
import shutil
import random
from huggingface_hub import HfApi, login
import multiprocessing as mp

def compress_worker(args):
    old_path, new_path, new_name = args
    try:
        if not os.path.exists(new_path):
            with open(old_path, 'rb') as f_in:
                with gzip.open(new_path, 'wb', compresslevel=5) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return f"   ✔️ Done: {new_name}"
        else:
            return f"   ⏩ Skipped: {new_name}"
    except Exception as e:
        return f"   ❌ Error at {new_name}: {str(e)}"

def process_and_compress(file_list, prefix, target_dir, source_dir):
    total = len(file_list)
    print(f"📦 Compressing group {prefix} ({total} files) with multiprocessing...")

    tasks = []
    for i, old_name in enumerate(file_list):
        old_path = os.path.join(source_dir, old_name)
        new_name = f"{prefix}-{i:05d}-of-{total:05d}.jsonl.gz"
        new_path = os.path.join(target_dir, new_name)
        tasks.append((old_path, new_path, new_name))

    num_cores = min(mp.cpu_count(), 16)
    print(f"🚀 Using {num_cores} cores for parallel compression")

    with mp.Pool(num_cores) as pool:
        for res in pool.imap_unordered(compress_worker, tasks):
            print(res)

    print(f"🎉 Finished group {prefix}!")

def main():
    if "HF_TOKEN" not in os.environ:
        raise EnvironmentError("❌ HF_TOKEN not set. Run: export HF_TOKEN='your_token_here'")

    login(token=os.environ["HF_TOKEN"])

    # 1. Source directory containing merged final_vla_rank_*.jsonl files
    SOURCE_DIR = "./FineVideo-VLA/final_dataset"
    UPLOAD_DIR = "./hf_upload_ready"
    TRAIN_DIR = os.path.join(UPLOAD_DIR, "train")
    TEST_DIR = os.path.join(UPLOAD_DIR, "test")

    TOTAL_SHARDS = 160
    TEST_RATIO = 0.05
    SEED = 42

    # 2. Target HuggingFace repo
    REPO_ID = "EmpathicRobotics/FineVideo-Phase5-AgentTokens"

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # 3. Build shard filename list
    all_shards = [f"final_vla_rank_{i}.jsonl" for i in range(TOTAL_SHARDS)]

    print("🔍 Verifying all shards exist...")
    for f in all_shards:
        if not os.path.exists(os.path.join(SOURCE_DIR, f)):
            raise FileNotFoundError(f"❌ Missing shard: {os.path.join(SOURCE_DIR, f)}")
    print("✅ All 160 shards found.")

    random.seed(SEED)
    random.shuffle(all_shards)

    test_count = int(TOTAL_SHARDS * TEST_RATIO)
    test_files = all_shards[:test_count]
    train_files = all_shards[test_count:]

    print(f"Train: {len(train_files)} | Test: {len(test_files)}")

    process_and_compress(train_files, "train", TRAIN_DIR, SOURCE_DIR)
    process_and_compress(test_files, "test", TEST_DIR, SOURCE_DIR)

    print("✅ Compression complete.")

    if len(os.listdir(TRAIN_DIR)) != len(train_files):
        raise ValueError("Missing train shards!")
    if len(os.listdir(TEST_DIR)) != len(test_files):
        raise ValueError("Missing test shards!")

    print(f"🚀 Uploading to {REPO_ID} ...")
    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=UPLOAD_DIR, repo_id=REPO_ID, repo_type="dataset")

    print("✨ Done! Dataset uploaded successfully to HuggingFace.")

if __name__ == "__main__":
    main()
