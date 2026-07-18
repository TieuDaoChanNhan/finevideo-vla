#!/usr/bin/env python3
"""
Flatten RoboVQA's json/{train,val}/*.json shards into Megatron-ready flat
JSONL ({"text": ...} per line, matching the shape mv_preprocess_data.py reads).

Source records already look like:
    {"uid": "...", "text": "<task:...>\n...<PRED>...</PRED>...", "video": "....mp4"}

The `text` field is already a well-formed multi-task instruction string (no
per-video context header needed the way FineVideo's ### Title:/### Context:
is, since each RoboVQA record is self-contained). This script does the
minimal transform: drop `uid`/`video` (not needed for a text-only pretrain
signal), keep `text` as-is.

The `video` field (mp4 filename) and the `tfrecord/` shards are intentionally
NOT touched here -- video-token extraction for RoboVQA is a separate,
deprioritized decision (see datasets.md), not part of this pass.

Usage:
    python3 data_prep/robovqa/flatten_text.py
"""
import json
import os

SRC_DIRS = [
    "/p/data1/mmlaion/shared/vla/robovqa/json/train",
    "/p/data1/mmlaion/shared/vla/robovqa/json/val",
]
OUT_PATH = "/p/data1/mmlaion/shared/vla/robovqa_flat/robovqa_flat.jsonl"


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    n_in = 0
    n_out = 0
    n_empty_text = 0

    with open(OUT_PATH, "w") as out_f:
        for src_dir in SRC_DIRS:
            shard_files = sorted(f for f in os.listdir(src_dir) if f.endswith(".json"))
            print(f"{src_dir}: {len(shard_files)} shards")
            for fname in shard_files:
                path = os.path.join(src_dir, fname)
                with open(path) as in_f:
                    for line in in_f:
                        line = line.strip()
                        if not line:
                            continue
                        n_in += 1
                        rec = json.loads(line)
                        text = rec.get("text", "")
                        if not text:
                            n_empty_text += 1
                            continue
                        out_f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                        n_out += 1

    print(f"\nRead {n_in} records, wrote {n_out} flat lines "
          f"({n_empty_text} skipped for empty text) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
