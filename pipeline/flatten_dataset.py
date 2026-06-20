"""
Flatten the merged adaptive dataset into Megatron-LM JSONL.

Reads final_dataset_adaptive (hierarchical scenes/activities) and produces
one {"text": "..."} record per activity that contains <agent> tokens.

Includes data augmentation: synonym replacement, stopword dropout,
sentence permutation, modality dropout, and speech/token interleaving.

Modality dropout balances the token ratio across modalities:
    AVC-LM tokens outnumber agent tokens ~373x, so 99% are dropped.
    Cosmos tokens outnumber agent tokens ~19x, so 90% are dropped.
    Seed2 and Agent tokens are kept at 100%.

Token flattening:
    <seed2> 3758 2157 </seed2>                         → <seed2_3758> <seed2_2157>
    <cosmos> 58567 </cosmos>                            → <cosmos_58567>
    <avc_lm> 100 200 </avc_lm>                         → <avclm_100> <avclm_200>
    <agent> <fps_30> <pelvis> ... </pelvis> </agent>   → <fps_30> <pelvis> ... </pelvis>

Input:  .../FineVideo-VLA/final_dataset_adaptive/final_vla_adaptive_rank_*.jsonl
Output: .../FineVideo-VLA/megatron_dataset_adaptive/flat_*.jsonl

Usage:
    python pipeline/flatten_dataset.py [--drop_avc 0.99] [--drop_cosmos 0.9]
"""

import json
import re
import random
import argparse
import glob
import multiprocessing as mp
import os
from functools import partial

import wn

try:
    _WORDNET = wn.Wordnet('oewn:2024')
except Exception:
    _WORDNET = wn.Wordnet()

DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's",
    "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't",
    "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's",
    "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves"
}


def get_wordnet_synonym(word):
    word_lower = word.lower()
    synonyms = set()

    for synset in _WORDNET.synsets(word_lower, pos='n'):
        for w in synset.words():
            syn_name = w.lemma().replace('_', ' ')
            if syn_name[0] == syn_name[0].upper() and word[0].upper() != word[0]:
                continue
            if syn_name.lower() != word_lower:
                synonyms.add(syn_name)

    if not synonyms:
        for synset in _WORDNET.synsets(word_lower, pos='v'):
            for w in synset.words():
                syn_name = w.lemma().replace('_', ' ')
                if syn_name.lower() != word_lower:
                    synonyms.add(syn_name)

    if synonyms:
        chosen_syn = random.choice(list(synonyms))
        if word.istitle():
            return chosen_syn.title()
        elif word.isupper():
            return chosen_syn.upper()
        return chosen_syn

    return word


def augment_text_string(text, synonym_rate=0.15, stopword_drop_rate=0.05):
    if not text:
        return ""
    words = text.split()
    augmented_words = []
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word).lower()
        if clean_word in DEFAULT_STOPWORDS and random.random() < stopword_drop_rate:
            continue
        if len(clean_word) > 5:
            core_match = re.match(r'^([^\w]*)(.*?)([^\w]*)$', word)
            if core_match:
                prefix, core, suffix = core_match.groups()
                augmented_words.append(prefix + get_wordnet_synonym(core) + suffix)
            else:
                augmented_words.append(get_wordnet_synonym(word))
        else:
            augmented_words.append(word)
    return " ".join(augmented_words)


def permute_chunks_list(chunks, permutation_rate=0.10):
    if len(chunks) < 2:
        return chunks
    chunks_copy = list(chunks)
    num_to_permute = max(1, int(len(chunks_copy) * permutation_rate))
    for _ in range(num_to_permute):
        idx_a = random.randint(0, len(chunks_copy) - 1)
        idx_b = random.randint(0, len(chunks_copy) - 1)
        chunks_copy[idx_a], chunks_copy[idx_b] = chunks_copy[idx_b], chunks_copy[idx_a]
    return chunks_copy


def process_tokens_to_individual_tags(token_str, drop_rate_avc=1.0, drop_rate_cosmos=0.5, drop_rate_seed=0.5):
    """Flatten <tag> payload </tag> blocks into individual vocab tokens.

    Standard modalities:  <tag> N1 N2 </tag> → <prefix_N1> <prefix_N2>
    Agent blocks:         <agent> <fps_30> <pelvis> ... </agent> → kept as-is
    """
    if not isinstance(token_str, str):
        return [], ""

    pattern = r'(<\w+>.*?</\w+>)'
    segments = re.findall(pattern, token_str, re.DOTALL)
    trailing_text = re.sub(pattern, '', token_str, flags=re.DOTALL).strip()
    all_final_tokens = []

    for seg in segments:
        match = re.match(r'<([^>]+)>(.*?)</\1>', seg, re.DOTALL)
        if not match:
            continue

        tag_name = match.group(1).strip()
        payload = match.group(2).strip()
        tag_lower = tag_name.lower()

        if "agent" in tag_lower:
            keep = True
        elif tag_lower.startswith("avc"):
            keep = random.random() > drop_rate_avc
        elif tag_lower.startswith("cosmos"):
            keep = random.random() > drop_rate_cosmos
        elif tag_lower.startswith("seed"):
            keep = random.random() > drop_rate_seed
        else:
            keep = True

        if not keep:
            continue

        if "agent" in tag_lower:
            inner_tokens = re.findall(r'<[^>]+>', payload)
            if inner_tokens:
                all_final_tokens.extend(inner_tokens)
            else:
                nums = payload.split()
                all_final_tokens.extend(f"<agent_{n}>" for n in nums if n.isdigit())
        else:
            prefix = "avclm" if tag_name == "avc_lm" else tag_name
            nums = payload.split()
            all_final_tokens.extend(f"<{prefix}_{n}>" for n in nums if n.isdigit())

    return all_final_tokens, trailing_text


def process_transcript_into_chunks(text, max_words=20, permute_rate=0.10, syn_rate=0.15, stop_rate=0.05):
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s]
    raw_chunks = []
    for sentence in sentences:
        augmented_sentence = augment_text_string(sentence, synonym_rate=syn_rate, stopword_drop_rate=stop_rate)
        words = augmented_sentence.split()
        if len(words) <= max_words:
            if augmented_sentence:
                raw_chunks.append(augmented_sentence)
        else:
            for i in range(0, len(words), max_words):
                sub_chunk = " ".join(words[i:i + max_words])
                if sub_chunk:
                    raw_chunks.append(sub_chunk)
    return permute_chunks_list(raw_chunks, permutation_rate=permute_rate)


def interleave_speech_and_tokens(chunks, tokens):
    if not chunks:
        return " ".join(tokens)
    if not tokens:
        return " ".join(chunks)
    num_chunks, num_tokens = len(chunks), len(tokens)
    result = list(tokens)
    if num_chunks == 1:
        insert_positions = [random.choice([0, num_tokens // 2, num_tokens])]
    else:
        insert_positions = [int(i * num_tokens / (num_chunks - 1)) for i in range(num_chunks)]
    if random.choice([True, False]):
        insert_positions = [min(num_tokens, pos + (num_tokens // (num_chunks * 2))) for pos in insert_positions]
    for chunk_idx, token_pos in reversed(list(enumerate(insert_positions))):
        result.insert(token_pos, chunks[chunk_idx])
    return " ".join(result)


def main():
    parser = argparse.ArgumentParser(description="Flatten adaptive merged dataset into Megatron-LM JSONL.")
    parser.add_argument("--input-glob",
                        default="/e/data1/datasets/playground/mmlaion/shared/nguyen38/FineVideo-VLA/final_dataset_adaptive/final_vla_adaptive_rank_*.jsonl")
    parser.add_argument("--output-dir",
                        default="/e/data1/datasets/playground/mmlaion/shared/nguyen38/FineVideo-VLA/megatron_dataset_adaptive")
    parser.add_argument("--drop_avc", type=float, default=0.99, help="Dropout rate for AVC tags (373x agent, keep ~1%%)")
    parser.add_argument("--drop_cosmos", type=float, default=0.9, help="Dropout rate for Cosmos tags (19x agent, keep ~10%%)")
    parser.add_argument("--drop_seed", type=float, default=0.0, help="Dropout rate for Seed tags (balanced with agent)")
    parser.add_argument("--synonym_rate", type=float, default=0.15, help="Synonym mutation chance")
    parser.add_argument("--stopword_drop", type=float, default=0.05, help="Stopword removal chance")
    parser.add_argument("--permute_sentences", type=float, default=0.10, help="Sentence swap chance")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    input_paths = sorted(glob.glob(args.input_glob))
    if not input_paths:
        raise FileNotFoundError(f"No files matched: {args.input_glob!r}")

    os.makedirs(args.output_dir, exist_ok=True)

    worker_fn = partial(
        flatten_one_file,
        output_dir=args.output_dir,
        skip_existing=args.skip_existing,
        drop_avc=args.drop_avc,
        drop_cosmos=args.drop_cosmos,
        drop_seed=args.drop_seed,
        synonym_rate=args.synonym_rate,
        stopword_drop=args.stopword_drop,
        permute_sentences=args.permute_sentences,
    )

    print(f"Flattening {len(input_paths)} files with {args.workers} workers")
    num_workers = min(args.workers, len(input_paths))

    if num_workers <= 1:
        for p in input_paths:
            print(worker_fn(p))
    else:
        with mp.Pool(num_workers) as pool:
            for result in pool.imap_unordered(worker_fn, input_paths):
                print(result)

    print("Done.")


def flatten_one_file(in_path, output_dir, skip_existing,
                     drop_avc, drop_cosmos, drop_seed,
                     synonym_rate, stopword_drop, permute_sentences):
    base = os.path.basename(in_path)
    out_path = os.path.join(output_dir, f"flat_{base}")

    if skip_existing and os.path.exists(out_path):
        return f"[SKIP] {out_path}"

    h_prefix = chr(35) * 3
    written = 0

    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            global_context = data.get("global_context", "")

            for scene in data.get("scenes", []):
                scene_title = scene.get("scene_title", "")
                scene_thematic = scene.get("scene_thematic", "")
                scene_mood = scene.get("scene_mood", "")

                for activity in scene.get("activities", []):
                    raw_tokens = activity.get("video_tokens", "")
                    if "<agent>" not in raw_tokens:
                        continue

                    speech = activity.get("speech_transcript", "")
                    text_prompt = activity.get("text_prompt", "")

                    text_chunks = process_transcript_into_chunks(
                        speech, max_words=20,
                        permute_rate=permute_sentences,
                        syn_rate=synonym_rate,
                        stop_rate=stopword_drop,
                    )

                    kept_tokens, trailing_text = process_tokens_to_individual_tags(
                        raw_tokens,
                        drop_rate_avc=drop_avc,
                        drop_rate_cosmos=drop_cosmos,
                        drop_rate_seed=drop_seed,
                    )

                    interleaved = interleave_speech_and_tokens(text_chunks, kept_tokens)
                    if trailing_text:
                        interleaved += f" {trailing_text}"

                    aug_title = augment_text_string(scene_title, synonym_rate, stopword_drop)
                    combined_ctx = f"{global_context} {text_prompt}".strip()
                    aug_ctx = augment_text_string(combined_ctx, synonym_rate, stopword_drop)
                    combined_kw = f"{scene_thematic}, {scene_mood}".strip()
                    aug_kw = augment_text_string(combined_kw, synonym_rate, stopword_drop)

                    layout_blocks = [
                        f"{h_prefix} Title: {aug_title}",
                        f"{h_prefix} Context: {aug_ctx}",
                        f"{h_prefix} Keywords: {aug_kw}",
                        interleaved,
                    ]
                    random.shuffle(layout_blocks)

                    output = "\n".join(layout_blocks)
                    output = output.replace(" , ", "").replace(",.", ".").replace(".,", ".").replace(":", ": ").replace(":  ", ": ")

                    record = {"text": output}
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1

    return f"[DONE] {out_path} | {written} records"


if __name__ == "__main__":
    main()
