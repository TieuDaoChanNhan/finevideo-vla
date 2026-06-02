from huggingface_hub import login, snapshot_download
import os
# You could get your Hugging Face token from https://huggingface.co/settings/tokens
login(token=os.environ["HF_TOKEN"], add_to_git_credential=True)
# You could specify the tokenizers you want to download.
model_names = [
        "Cosmos-Tokenizer-DV8x16x16",
]
for model_name in model_names:
    hf_repo = "nvidia/" + model_name
    local_dir = "pretrained_ckpts/" + model_name
    os.makedirs(local_dir, exist_ok=True)
    print(f"downloading {model_name} to {local_dir}...")
    snapshot_download(repo_id=hf_repo, local_dir=local_dir)
