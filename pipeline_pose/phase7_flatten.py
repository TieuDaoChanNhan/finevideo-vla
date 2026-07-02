#!/usr/bin/env python3
"""
Phase 7 v4 — Flatten Phase 6 v2 merged dataset into Megatron-LM JSONL.

Key changes from v3:
  1. Per-chunk temporal ordering.
     Old: [all cosmos ...][all agent ...][all snac ...]
     New: for each 8-frame chunk → [<seed2_N>...?][<cosmos_N>...?][agent tokens?][<snac_N>...?]
     State machine walks events in document order; AVC-LM fires the per-chunk flush.

  2. Speech in a dedicated ### Speech: header block, NOT scattered into the token sequence.
     Previously speech words were randomly inserted among cosmos/agent/snac tokens, breaking
     agent joint grammar in 43% of full-chain records.

  3. Text header blocks (Title / Context / Keywords / Speech) are shuffled among
     themselves.  The token sequence is ALWAYS placed after all text blocks to
     guarantee a consistent text-prompt → token-generation order for inference.

  4. Per-file token counts printed to stdout so you can sum them for a total
     dataset size without running a separate Megatron tokenisation pass.

Record filter (unchanged from v3):
  Emit any activity that has <agent> OR <snac> tokens.
  Pure seed2+cosmos activities are skipped.

Drop rates (unchanged from v3):
  AVC-LM  100%   (removed pending ablation)
  Cosmos   50%   (per-chunk independent decision)
  Seed2     0%   (always keep)
  Agent     0%   (always keep)
  SNAC      0%   (always keep)

Input:  .../final_dataset_adaptive_v2/final_vla_adaptive_v2_rank_*.jsonl   (Phase 6 v2 output)
Output: .../megatron_dataset_v4/flat_*.jsonl

Usage:
    python pipeline_pose/phase7_flatten.py [options]
    python pipeline_pose/phase7_flatten.py --skip-existing   # resume interrupted run
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

# ── Text augmentation helpers (unchanged from v3) ────────────────────────────

DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's",
    "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not",
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll",
    "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
    "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when",
    "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
    "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves",
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
        chosen = random.choice(list(synonyms))
        if word.istitle():
            return chosen.title()
        if word.isupper():
            return chosen.upper()
        return chosen
    return word


def augment_text_string(text, synonym_rate=0.15, stopword_drop_rate=0.05):
    if not text:
        return ""
    words = text.split()
    out = []
    for word in words:
        clean = re.sub(r'[^\w]', '', word).lower()
        if clean in DEFAULT_STOPWORDS and random.random() < stopword_drop_rate:
            continue
        if len(clean) > 5:
            m = re.match(r'^([^\w]*)(.*?)([^\w]*)$', word)
            if m and random.random() < synonym_rate:
                prefix, core, suffix = m.groups()
                out.append(prefix + get_wordnet_synonym(core) + suffix)
            else:
                out.append(word)
        else:
            out.append(word)
    return " ".join(out)


def permute_chunks_list(chunks, permutation_rate=0.10):
    if len(chunks) < 2:
        return chunks
    c = list(chunks)
    n = max(1, int(len(c) * permutation_rate))
    for _ in range(n):
        a = random.randint(0, len(c) - 1)
        b = random.randint(0, len(c) - 1)
        c[a], c[b] = c[b], c[a]
    return c


def process_transcript_into_chunks(text, max_words=20, permute_rate=0.10,
                                   syn_rate=0.15, stop_rate=0.05):
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s]
    raw = []
    for sent in sentences:
        aug = augment_text_string(sent, synonym_rate=syn_rate, stopword_drop_rate=stop_rate)
        words = aug.split()
        if len(words) <= max_words:
            if aug:
                raw.append(aug)
        else:
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i:i + max_words])
                if chunk:
                    raw.append(chunk)
    return permute_chunks_list(raw, permutation_rate=permute_rate)


# ── Pre-compiled patterns ────────────────────────────────────────────────────

# Simple blocks with numeric payloads
_RE_SIMPLE = re.compile(r'<(seed2|cosmos|avc_lm)>(.*?)</\1>', re.DOTALL)
# Agent blocks: contain nested named-joint tags
_RE_AGENT  = re.compile(r'<agent>(.*?)</agent>', re.DOTALL)
# SNAC blocks: contain <snac_N> tokens
_RE_SNAC   = re.compile(r'<snac>(.*?)</snac>', re.DOTALL)
# Any single tag <...>
_RE_TAG    = re.compile(r'<[^>]+>')


# ── Core: per-chunk temporal flatten ────────────────────────────────────────

def process_activity_per_chunk(token_str,
                                drop_rate_cosmos=0.5,
                                drop_rate_seed=0.0,
                                drop_rate_snac=0.0):
    """
    Flatten video_tokens into a temporally-ordered list of vocab token strings.

    Phase 6 v2 writes chunks in this order for each 8-frame window:
        [<seed2>N...</seed2>?]  <cosmos>N...</cosmos>  <avc_lm>N...</avc_lm>
        [<agent><fps_30>...</agent>?]  [<snac><snac_N>...</snac>?]

    This function emits per chunk:
        [<seed2_N>...?]  [<cosmos_N>...?]  [agent inner tokens?]  [<snac_N>...?]

    State machine (processes events in document order):
        seed2   →  buffer pending_seed2
        cosmos  →  buffer pending_cosmos
        avc_lm  →  flush: emit pending_seed2 (dropout), emit pending_cosmos (dropout)
        agent   →  emit inner named-joint tags directly (no dropout)
        snac    →  emit <snac_N> tags directly (optional dropout)

    AVC-LM payload is always discarded.
    """
    if not isinstance(token_str, str) or not token_str.strip():
        return []

    # ── Collect all named block events in document order ─────────────────────
    events = []   # (position, block_type, payload_string)

    for m in _RE_SIMPLE.finditer(token_str):
        events.append((m.start(), m.group(1), m.group(2)))

    for m in _RE_AGENT.finditer(token_str):
        events.append((m.start(), 'agent', m.group(1)))

    for m in _RE_SNAC.finditer(token_str):
        events.append((m.start(), 'snac', m.group(1)))

    if not events:
        return []

    events.sort(key=lambda x: x[0])

    # ── State machine ─────────────────────────────────────────────────────────
    all_output     = []
    pending_seed2  = None   # seed2 payload waiting for its avc_lm trigger
    pending_cosmos = None   # cosmos payload waiting for its avc_lm trigger

    for _, etype, payload in events:

        if etype == 'seed2':
            pending_seed2 = payload

        elif etype == 'cosmos':
            pending_cosmos = payload

        elif etype == 'avc_lm':
            # avc_lm fires → flush pending video tokens for this chunk

            if pending_seed2 is not None and random.random() > drop_rate_seed:
                all_output.extend(
                    f'<seed2_{n}>' for n in pending_seed2.split() if n.isdigit()
                )
            pending_seed2 = None

            if pending_cosmos is not None and random.random() > drop_rate_cosmos:
                all_output.extend(
                    f'<cosmos_{n}>' for n in pending_cosmos.split() if n.isdigit()
                )
            pending_cosmos = None
            # avc_lm payload: discarded (100% drop)

        elif etype == 'agent':
            # Pass through inner named-joint tokens as-is
            all_output.extend(_RE_TAG.findall(payload))

        elif etype == 'snac':
            if random.random() > drop_rate_snac:
                all_output.extend(_RE_TAG.findall(payload))

    return all_output


# ── Token type counter ───────────────────────────────────────────────────────

def count_token_types(tokens):
    """Return (n_seed2, n_cosmos, n_agent, n_snac) counts."""
    s2 = co = ag = sn = 0
    for tok in tokens:
        if tok.startswith('<seed2_'):
            s2 += 1
        elif tok.startswith('<cosmos_'):
            co += 1
        elif tok.startswith('<snac_'):
            sn += 1
        else:
            ag += 1   # fps_N, joint open/close, joint_t_N, joint_x/y/z_N
    return s2, co, ag, sn


# ── Per-file worker ──────────────────────────────────────────────────────────

def flatten_one_file(in_path, output_dir, skip_existing,
                     drop_avc, drop_cosmos, drop_seed, drop_snac,
                     synonym_rate, stopword_drop, permute_sentences):
    """
    Flatten one Phase 6 v2 JSONL shard into flat Megatron-LM JSONL.

    Returns a dict with per-file stats for aggregate reporting.
    """
    base     = os.path.basename(in_path)
    out_path = os.path.join(output_dir, f"flat_{base}")

    if skip_existing and os.path.exists(out_path):
        return {"status": "skip", "path": out_path,
                "records": 0, "seed2": 0, "cosmos": 0, "agent": 0, "snac": 0}

    H = "###"
    written = 0
    total_seed2 = total_cosmos = total_agent = total_snac = 0

    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        for raw_line in fin:
            if not raw_line.strip():
                continue
            try:
                data = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            global_context = data.get("global_context", "")

            for scene in data.get("scenes", []):
                scene_title    = scene.get("scene_title", "")
                scene_thematic = scene.get("scene_thematic", "")
                scene_mood     = scene.get("scene_mood", "")

                for activity in scene.get("activities", []):
                    raw_tokens = activity.get("video_tokens", "")

                    # Filter: only emit activities with agent OR snac tokens
                    if "<agent>" not in raw_tokens and "<snac>" not in raw_tokens:
                        continue

                    speech      = activity.get("speech_transcript", "")
                    text_prompt = activity.get("text_prompt", "")

                    # ── Token processing (per-chunk temporal ordering) ──────
                    kept_tokens = process_activity_per_chunk(
                        raw_tokens,
                        drop_rate_cosmos=drop_cosmos,
                        drop_rate_seed=drop_seed,
                        drop_rate_snac=drop_snac,
                    )
                    if not kept_tokens:
                        continue

                    s2, co, ag, sn = count_token_types(kept_tokens)
                    total_seed2  += s2
                    total_cosmos += co
                    total_agent  += ag
                    total_snac   += sn

                    # ── Text header blocks ──────────────────────────────────
                    aug_title = augment_text_string(
                        scene_title, synonym_rate, stopword_drop)
                    aug_ctx = augment_text_string(
                        f"{global_context} {text_prompt}".strip(),
                        synonym_rate, stopword_drop)
                    aug_kw = augment_text_string(
                        f"{scene_thematic}, {scene_mood}".strip(),
                        synonym_rate, stopword_drop)

                    # Speech: augmented and placed in a dedicated header block.
                    # NOT interleaved into the token sequence (fixed v3 bug).
                    speech_chunks = process_transcript_into_chunks(
                        speech, max_words=20,
                        permute_rate=permute_sentences,
                        syn_rate=synonym_rate,
                        stop_rate=stopword_drop,
                    )
                    aug_speech = " ".join(speech_chunks)

                    text_blocks = [
                        f"{H} Title: {aug_title}",
                        f"{H} Context: {aug_ctx}",
                        f"{H} Keywords: {aug_kw}",
                    ]
                    if aug_speech.strip():
                        text_blocks.append(f"{H} Speech: {aug_speech}")

                    # Shuffle text header blocks for layout diversity.
                    # Token sequence is ALWAYS placed after all text blocks.
                    random.shuffle(text_blocks)

                    token_str = " ".join(kept_tokens)
                    output = "\n".join(text_blocks) + "\n" + token_str
                    # Light cleanup of punctuation artefacts from augmentation
                    output = (output
                              .replace(" , ", " ")
                              .replace(",.", ".")
                              .replace(".,", ".")
                              .replace(":  ", ": "))

                    fout.write(json.dumps({"text": output}, ensure_ascii=False) + "\n")
                    written += 1

    total = total_seed2 + total_cosmos + total_agent + total_snac
    return {
        "status":  "done",
        "path":    out_path,
        "records": written,
        "seed2":   total_seed2,
        "cosmos":  total_cosmos,
        "agent":   total_agent,
        "snac":    total_snac,
        "total":   total,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 7 v4 — per-chunk temporal flatten for Megatron-LM pretraining."
    )
    parser.add_argument(
        "--input-glob",
        default=(
            "/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA"
            "/final_dataset_adaptive_v2/final_vla_adaptive_v2_rank_*.jsonl"
        ),
        help="Glob pattern for Phase 6 v2 JSONL shards",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA"
            "/megatron_dataset_v4"
        ),
        help="Output directory for flat Megatron-LM JSONL",
    )
    parser.add_argument("--drop_avc",    type=float, default=1.0,
                        help="AVC-LM drop rate (always 1.0 in v4; argument kept for compat)")
    parser.add_argument("--drop_cosmos", type=float, default=0.5,
                        help="Cosmos per-chunk dropout rate")
    parser.add_argument("--drop_seed",   type=float, default=0.0,
                        help="Seed2 dropout rate (0.0 = keep all)")
    parser.add_argument("--drop_snac",   type=float, default=0.0,
                        help="SNAC dropout rate (0.0 = keep all)")
    parser.add_argument("--synonym_rate",      type=float, default=0.15)
    parser.add_argument("--stopword_drop",     type=float, default=0.05)
    parser.add_argument("--permute_sentences", type=float, default=0.10)
    parser.add_argument("--workers",    type=int, default=32,
                        help="Number of parallel worker processes")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip output files that already exist (resume support)")
    args = parser.parse_args()

    input_paths = sorted(glob.glob(args.input_glob))
    if not input_paths:
        raise FileNotFoundError(f"No input files matched: {args.input_glob!r}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[Phase7 v4] {len(input_paths)} input files → {args.output_dir}")
    print(f"[Phase7 v4] drop: cosmos={args.drop_cosmos} seed={args.drop_seed} "
          f"snac={args.drop_snac} avc=1.0(fixed)")
    print(f"[Phase7 v4] workers={min(args.workers, len(input_paths))}")

    worker_fn = partial(
        flatten_one_file,
        output_dir=args.output_dir,
        skip_existing=args.skip_existing,
        drop_avc=args.drop_avc,
        drop_cosmos=args.drop_cosmos,
        drop_seed=args.drop_seed,
        drop_snac=args.drop_snac,
        synonym_rate=args.synonym_rate,
        stopword_drop=args.stopword_drop,
        permute_sentences=args.permute_sentences,
    )

    # Aggregate counters
    total_records = total_seed2 = total_cosmos = total_agent = total_snac = 0
    n_skipped = n_done = 0

    num_workers = min(args.workers, len(input_paths))
    with mp.Pool(num_workers) as pool:
        for result in pool.imap_unordered(worker_fn, input_paths):
            if result["status"] == "skip":
                n_skipped += 1
                print(f"[SKIP] {os.path.basename(result['path'])}")
            else:
                n_done += 1
                total_records += result["records"]
                total_seed2   += result["seed2"]
                total_cosmos  += result["cosmos"]
                total_agent   += result["agent"]
                total_snac    += result["snac"]
                print(
                    f"[DONE] {os.path.basename(result['path'])} | "
                    f"{result['records']:5d} records | "
                    f"seed2={result['seed2']:,} "
                    f"cosmos={result['cosmos']:,} "
                    f"agent={result['agent']:,} "
                    f"snac={result['snac']:,} "
                    f"total={result['total']:,}"
                )

    grand_total = total_seed2 + total_cosmos + total_agent + total_snac
    denom = max(grand_total, 1)

    print()
    print("=" * 72)
    print(f"Phase 7 v4 — DONE")
    print(f"  Files processed : {n_done}  ({n_skipped} skipped)")
    print(f"  Total records   : {total_records:,}")
    print(f"  Token counts:")
    print(f"    seed2   : {total_seed2:>15,}  ({total_seed2 / denom * 100:.1f}%)")
    print(f"    cosmos  : {total_cosmos:>15,}  ({total_cosmos / denom * 100:.1f}%)")
    print(f"    agent   : {total_agent:>15,}  ({total_agent / denom * 100:.1f}%)")
    print(f"    snac    : {total_snac:>15,}  ({total_snac / denom * 100:.1f}%)")
    print(f"    TOTAL   : {grand_total:>15,}  ({grand_total / 1e9:.3f}B)")
    print("=" * 72)


if __name__ == "__main__":
    main()
