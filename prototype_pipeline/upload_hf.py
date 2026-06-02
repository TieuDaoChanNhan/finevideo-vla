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
            return f"   ✔️ Đã xong: {new_name}"
        else:
            return f"   ⏩ Bỏ qua: {new_name}"
    except Exception as e:
        return f"   ❌ Lỗi tại {new_name}: {str(e)}"

def process_and_compress(file_list, prefix, target_dir, source_dir):
    total = len(file_list)
    print(f"📦 Đang xử lý nhóm {prefix} ({total} files) bằng Multiprocessing...")

    tasks = []
    for i, old_name in enumerate(file_list):
        old_path = os.path.join(source_dir, old_name)
        new_name = f"{prefix}-{i:05d}-of-{total:05d}.jsonl.gz"
        new_path = os.path.join(target_dir, new_name)
        tasks.append((old_path, new_path, new_name))

    num_cores = min(mp.cpu_count(), 16)
    print(f"🚀 Dùng {num_cores} core nén song song")

    with mp.Pool(num_cores) as pool:
        for res in pool.imap_unordered(compress_worker, tasks):
            print(res)

    print(f"🎉 Hoàn tất nhóm {prefix}!")

def main():
    if "HF_TOKEN" not in os.environ:
        raise EnvironmentError("❌ Chưa export HF_TOKEN! Chạy lệnh: export HF_TOKEN='your_token_here'")

    login(token=os.environ["HF_TOKEN"])

    # 1. CẬP NHẬT ĐƯỜNG DẪN SOURCE
    SOURCE_DIR = "./FineVideo-VLA/final_dataset" # Trỏ vào thư mục chứa file đã merge
    UPLOAD_DIR = "./hf_upload_ready"
    TRAIN_DIR = os.path.join(UPLOAD_DIR, "train")
    TEST_DIR = os.path.join(UPLOAD_DIR, "test")

    TOTAL_SHARDS = 160
    TEST_RATIO = 0.05
    SEED = 42
    
    # 2. ĐỔI TÊN REPO MỚI
    REPO_ID = "EmpathicRobotics/FineVideo-VLA-Agent" 

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # 3. CẬP NHẬT TÊN FILE THEO ĐỊNH DẠNG MỚI
    all_shards = [f"final_vla_rank_{i}.jsonl" for i in range(TOTAL_SHARDS)]
    
    print("🔍 Kiểm tra đủ shard...")
    for f in all_shards:
        if not os.path.exists(os.path.join(SOURCE_DIR, f)):
            raise FileNotFoundError(f"❌ Thiếu file: {os.path.join(SOURCE_DIR, f)}")
    print("✅ Đủ 160 shard!")

    # Shuffle
    random.seed(SEED)
    random.shuffle(all_shards)

    # Split
    test_count = int(TOTAL_SHARDS * TEST_RATIO)
    test_files = all_shards[:test_count]
    train_files = all_shards[test_count:]

    print(f"Train: {len(train_files)} | Test: {len(test_files)}")

    # Compress
    process_and_compress(train_files, "train", TRAIN_DIR, SOURCE_DIR)
    process_and_compress(test_files, "test", TEST_DIR, SOURCE_DIR)

    print("✅ Nén xong!")

    # Verify
    if len(os.listdir(TRAIN_DIR)) != len(train_files):
        raise ValueError("Thiếu shard train!")
    if len(os.listdir(TEST_DIR)) != len(test_files):
        raise ValueError("Thiếu shard test!")

    print(f"🚀 Uploading lên kho {REPO_ID} ...")
    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=UPLOAD_DIR, repo_id=REPO_ID, repo_type="dataset")

    print("✨ DONE! Chúc mừng cậu đã hoàn thành mảnh ghép dữ liệu đầu tiên cho VLA!")

if __name__ == "__main__":
    main()