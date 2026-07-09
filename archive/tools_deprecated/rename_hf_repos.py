"""
Rename HuggingFace dataset repos under EmpathicRobotics to follow
a consistent phase-numbered naming convention.

Renames:
  FineVideo-Tokenized       → FineVideo-Prototype-Tokenized
  finevideo-3d-pose         → FineVideo-Phase2-3DPose
  FineVideo-Phase4-Pose     → FineVideo-Phase4-YOLOPose
  FineVideo-VLA-Agent       → FineVideo-Phase5-AgentTokens
  FineVideo-VLA-flattened   → FineVideo-Phase7-Flattened

Usage:
  export HF_TOKEN=<your_token>
  python rename_hf_repos.py
"""
from huggingface_hub import HfApi

ORG = "EmpathicRobotics"

RENAMES = [
    ("FineVideo-Tokenized",     "FineVideo-Prototype-Tokenized"),
    ("finevideo-3d-pose",       "FineVideo-Phase2-3DPose"),
    ("FineVideo-Phase4-Pose",   "FineVideo-Phase4-YOLOPose"),
    ("FineVideo-VLA-Agent",     "FineVideo-Phase5-AgentTokens"),
    ("FineVideo-VLA-flattened", "FineVideo-Phase7-Flattened"),
]

api = HfApi()

for old_name, new_name in RENAMES:
    old_repo = f"{ORG}/{old_name}"
    new_repo = f"{ORG}/{new_name}"
    print(f"  {old_repo}  →  {new_repo}")
    try:
        api.move_repo(from_id=old_repo, to_id=new_repo, repo_type="dataset")
        print(f"    ✓ done")
    except Exception as e:
        try:
            api.move_repo(from_id=old_repo, to_id=new_repo, repo_type="model")
            print(f"    ✓ done (model repo)")
        except Exception as e2:
            print(f"    ✗ failed: {e2}")

print("\nAll done.")
