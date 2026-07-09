#!/usr/bin/env python3
"""
Push dataset cards (README.md) to all five HuggingFace dataset repos.

Usage:
    export HF_TOKEN='hf_...'
    python tools/upload/push_dataset_cards.py
"""

import os

from huggingface_hub import HfApi, login

CARDS = [
    {
        "repo_id": "EmpathicRobotics/FineVideo-Prototype-Tokenized",
        "card_file": os.path.join(os.path.dirname(__file__), "prototype_tokenized_dataset_card.md"),
    },
    {
        "repo_id": "EmpathicRobotics/FineVideo-Phase2-3DPose",
        "card_file": os.path.join(os.path.dirname(__file__), "phase2_3dpose_dataset_card.md"),
    },
    {
        "repo_id": "EmpathicRobotics/FineVideo-Phase4-YOLOPose",
        "card_file": os.path.join(os.path.dirname(__file__), "phase4_dataset_card.md"),
    },
    {
        "repo_id": "EmpathicRobotics/FineVideo-Phase5-AgentTokens",
        "card_file": os.path.join(os.path.dirname(__file__), "vla_agent_dataset_card.md"),
    },
    {
        "repo_id": "EmpathicRobotics/FineVideo-Phase7-Flattened",
        "card_file": os.path.join(os.path.dirname(__file__), "vla_flattened_dataset_card.md"),
    },
]


def main():
    if "HF_TOKEN" not in os.environ:
        raise EnvironmentError("HF_TOKEN not set. Run: export HF_TOKEN='hf_...'")

    login(token=os.environ["HF_TOKEN"])
    api = HfApi()

    for entry in CARDS:
        repo_id = entry["repo_id"]
        card_file = entry["card_file"]

        if not os.path.exists(card_file):
            print(f"SKIP {repo_id} — card file not found: {card_file}")
            continue

        print(f"Pushing README.md to {repo_id} ...")
        api.upload_file(
            path_or_fileobj=card_file,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update dataset card",
        )
        print(f"  Done: https://huggingface.co/datasets/{repo_id}")

    print("\nAll dataset cards pushed.")


if __name__ == "__main__":
    main()
