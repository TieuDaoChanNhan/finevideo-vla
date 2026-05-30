import os
from huggingface_hub import HfApi

# =====================================================================
# CONFIGURATION
# =====================================================================
# 1. Paste your Hugging Face Access Token (Write permission) here:
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # set via: export HF_TOKEN=hf_...

# 2. Repository settings
USER_ID = "EmpathicRobotics"
MODEL_NAME = "vla-1.7b-pab-spline-25b-test"
REPO_ID = f"{USER_ID}/{MODEL_NAME}"

# 3. Path to your local converted Hugging Face checkpoint directory
LOCAL_DIR = "/e/project1/reformo/nguyen38/output_vla/vla_25b_test/hf/iter_0006000"
# =====================================================================

def upload_vla_model():
    if HF_TOKEN == "YOUR_HF_TOKEN_HERE" or not HF_TOKEN:
        print("❌ Error: Please paste your Hugging Face WRITE token into the HF_TOKEN variable.")
        return

    if not os.path.exists(LOCAL_DIR):
        print(f"❌ Error: Local directory not found at {LOCAL_DIR}")
        return

    api = HfApi()
    
    print(f"⏳ Creating repository '{REPO_ID}' if it doesn't exist...")
    try:
        api.create_repo(
            repo_id=REPO_ID,
            token=HF_TOKEN,
            private=True,  # Set to True for internal lab safety, change to False if public
            exist_ok=True
        )
        print("✅ Repository ready.")
    except Exception as e:
        print(f"❌ Failed to create/verify repository: {e}")
        return

    print(f"🚀 Uploading all files from {LOCAL_DIR} to HF Hub...")
    print("💡 This will upload model.safetensors (3.2 GB) and the correct expanded tokenizer files.")
    
    try:
        api.upload_folder(
            folder_path=LOCAL_DIR,
            repo_id=REPO_ID,
            token=HF_TOKEN,
            commit_message="Update"
        )
        print("\n" + "="*70)
        print("🎉 SUCCESS! Your VLA model has been successfully uploaded.")
        print(f"👉 Model URL: https://huggingface.co/{REPO_ID}")
        print("="*70)
    except Exception as e:
        print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    upload_vla_model()