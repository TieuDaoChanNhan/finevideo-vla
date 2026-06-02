import os
from huggingface_hub import HfApi

# Cấu hình thông tin
token = os.environ["HF_TOKEN"]
repo_id = "mixture-vitae-backup/MixtureVitae-Backup"
folder_path = "data/stack_images3_gzip_recover"

api = HfApi(token=token)

try:
    api.delete_folder(
        repo_id=repo_id,
        path_in_repo=folder_path,
        repo_type="dataset"  # Thay bằng "dataset" nếu repo này là Dataset
    )
    print(f"✅ Đã xóa thành công folder: {folder_path}")
except Exception as e:
    print(f"❌ Có lỗi xảy ra: {e}")