"""seed2 tokenize for synth_llava / synth_llava2 (image+caption, from
mixture-vitae-backup/MixtureVitae-Backup/data/multimodal), interleaved into
the existing <image_0> placeholder.

Source (see PROGRESS_VI.md 20/07 entry for the peek/download investigation):
  /p/data1/mmlaion/shared/vla/synth_llava/extracted/{synth_llava,synth_llava2}/
  shard-NNNNNNN.jsonl  -- one row/image: {"text", "metadata", "language", "media"}
    text:  "<caption><image_0>The image shows ...</caption>"
    media: '{"<image_0>": "_0_0.image_0.png"}'  -- placeholder -> filename inside .wds
  shard-NNNNNNN.wds     -- POSIX tar, <id>.image_0.png + <id>.metadata.json per row

Verified (20/07/2026, shard-0000000, 4000 rows): always exactly 1 image/row,
key always literally "<image_0>" -- no multi-image rows in this dataset, so
no need to handle >1 placeholder per row.

Static images -> seed2 only (cosmos needs an 8-frame chunk, avc_lm needs an
H.264 bitstream; neither applies to a single PNG). Reuses the exact working
Seed2Tokenizer import/shim from step_a_tokenize_video.py (fixes 2 real bugs:
a transformers-version helper-function move, and a Qformer.cls=None crash --
see that file's docstring) rather than re-deriving it.

Output: replace the <image_0> placeholder in `text` in place with
"<seed2_N> <seed2_N> ..." (atomic-token string form, not the raw int-list
`pipeline_video/image_pipeline.py` prototype used), drop metadata/media
(not needed once the placeholder is substituted) -> {"id", "text"}.

Usage:
    python data_prep/synth_llava/tokenize_seed2.py [--limit N] [--shard-glob PATTERN]
"""
import argparse
import glob
import io
import json
import os
import re
import sys
import tarfile

PROTOTYPE_DIR = "/e/project1/reformo/nguyen38/prototype"

# Exact same import/shim sequence as data_prep/omnivideo_100k/step_a/step_a_tokenize_video.py
sys.path.insert(0, PROTOTYPE_DIR)
os.chdir(PROTOTYPE_DIR)

import transformers.modeling_utils as _modeling_utils  # noqa: E402
import transformers.pytorch_utils as _pytorch_utils  # noqa: E402
for _name in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"):
    if not hasattr(_modeling_utils, _name):
        setattr(_modeling_utils, _name, getattr(_pytorch_utils, _name))

sys.path.insert(0, os.path.join(PROTOTYPE_DIR, "seed2"))  # noqa: E402
import seed2_tokenizer as _seed2_tokenizer  # noqa: E402


def _safe_get_output_embeddings(self):
    return None if self.cls is None else self.cls.predictions.decoder


def _safe_set_output_embeddings(self, new_embeddings):
    if self.cls is not None:
        self.cls.predictions.decoder = new_embeddings


for _cls in (_seed2_tokenizer.BertLMHeadModel, _seed2_tokenizer.BertForMaskedLM):
    _cls.get_output_embeddings = _safe_get_output_embeddings
    _cls.set_output_embeddings = _safe_set_output_embeddings

from pipeline import Seed2Tokenizer  # noqa: E402

DATA_ROOT = "/p/data1/mmlaion/shared/vla/synth_llava/extracted"
DEFAULT_OUTPUT_DIR = "/p/data1/mmlaion/shared/vla/synth_llava_flat"
SOURCES = ["synth_llava", "synth_llava2"]

_PLACEHOLDER_RE = re.compile(r"<image_\d+>")


def find_shard_pairs(shard_glob=None):
    pairs = []
    for source in SOURCES:
        src_dir = os.path.join(DATA_ROOT, source)
        if not os.path.isdir(src_dir):
            continue
        pattern = shard_glob or os.path.join(src_dir, "shard-*.jsonl")
        if shard_glob:
            pattern = os.path.join(src_dir, shard_glob)
        for jsonl_path in sorted(glob.glob(pattern)):
            wds_path = jsonl_path[:-len(".jsonl")] + ".wds"
            if os.path.exists(wds_path):
                pairs.append((source, jsonl_path, wds_path))
    return pairs


def process_shard(source, jsonl_path, wds_path, output_dir, seed2, limit=None):
    base = os.path.basename(jsonl_path)
    out_path = os.path.join(output_dir, f"{source}_{base}")
    if os.path.exists(out_path):
        return {"file": base, "skipped": True}

    tar = tarfile.open(wds_path, "r")
    # Build a name -> tarinfo index once (avoids O(n) getmember() scans per row)
    members = {m.name: m for m in tar.getmembers()}

    n_in = n_out = n_no_image = n_encode_fail = 0
    total_seed2_tokens = 0
    tmp_path = out_path + ".tmp"

    with open(jsonl_path, encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if limit is not None and n_in >= limit:
                break
            n_in += 1
            rec = json.loads(line)
            media = json.loads(rec.get("media", "{}"))
            metadata = json.loads(rec.get("metadata", "{}"))
            rec_id = f"{source}_{metadata.get('params', {}).get('id', n_in)}"

            m = _PLACEHOLDER_RE.search(rec["text"])
            if not m or m.group(0) not in media:
                n_no_image += 1
                continue
            placeholder = m.group(0)
            filename = media[placeholder]

            member = members.get(filename)
            if member is None:
                n_encode_fail += 1
                continue
            img_bytes = tar.extractfile(member).read()

            tmp_img_path = f"/tmp/synth_llava_seed2_{os.getpid()}.png"
            with open(tmp_img_path, "wb") as f:
                f.write(img_bytes)
            try:
                ids = seed2.encode_image(tmp_img_path)
            except Exception:
                ids = []
            finally:
                if os.path.exists(tmp_img_path):
                    os.remove(tmp_img_path)

            if not ids:
                n_encode_fail += 1
                continue

            seed2_str = " ".join(f"<seed2_{i}>" for i in ids)
            total_seed2_tokens += len(ids)
            # Source text is "<caption><image_0>...caption text...</caption>" --
            # a bare in-place substitute would nest seed2 tokens inside <caption>,
            # which is wrong: project convention (step_a_tokenize_video.py) treats
            # <caption> as a sibling of the modality block, wrapping only the
            # caption text, not the tokens. Strip the source's outer <caption>
            # wrapper and placeholder, then rebuild in the correct order.
            caption_text = rec["text"]
            if caption_text.startswith("<caption>") and caption_text.endswith("</caption>"):
                caption_text = caption_text[len("<caption>"):-len("</caption>")]
            caption_text = caption_text.replace(placeholder, "", 1).strip()
            final_text = f"{seed2_str} <caption> {caption_text} </caption>"

            fout.write(json.dumps({"id": rec_id, "text": final_text}, ensure_ascii=False) + "\n")
            n_out += 1

    tar.close()
    os.replace(tmp_path, out_path)
    return {
        "file": base, "n_in": n_in, "n_out": n_out,
        "n_no_image": n_no_image, "n_encode_fail": n_encode_fail,
        "total_seed2_tokens": total_seed2_tokens,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--shard-glob", default=None, help="e.g. 'shard-0000000.jsonl' to test on 1 shard")
    ap.add_argument("--limit", type=int, default=None, help="Only process first N rows per shard (for testing)")
    ap.add_argument("--max-shards", type=int, default=None, help="Only process first N shards total (for testing)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pairs = find_shard_pairs(args.shard_glob)
    if args.max_shards:
        pairs = pairs[:args.max_shards]
    print(f"{len(pairs)} shard pairs found")

    print("Loading Seed2Tokenizer...")
    seed2 = Seed2Tokenizer()
    print("Seed2Tokenizer loaded.")

    total_in = total_out = total_no_image = total_encode_fail = total_tokens = 0
    for source, jsonl_path, wds_path in pairs:
        stats = process_shard(source, jsonl_path, wds_path, args.output_dir, seed2, limit=args.limit)
        if stats.get("skipped"):
            print(f"{stats['file']}: da co, bo qua")
            continue
        total_in += stats["n_in"]
        total_out += stats["n_out"]
        total_no_image += stats["n_no_image"]
        total_encode_fail += stats["n_encode_fail"]
        total_tokens += stats["total_seed2_tokens"]
        print(f"{stats['file']}: {stats['n_in']} -> {stats['n_out']} "
              f"(no_image: {stats['n_no_image']}, encode_fail: {stats['n_encode_fail']}, "
              f"seed2_tokens: {stats['total_seed2_tokens']:,})")

    print(f"\nTONG: {total_in} -> {total_out} | no_image: {total_no_image} | "
          f"encode_fail: {total_encode_fail} | seed2_tokens: {total_tokens:,}")


if __name__ == "__main__":
    main()
