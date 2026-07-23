#!/usr/bin/env python3
"""
Phase 7 v5 — Flatten Phase 6 v4 merged dataset into Megatron-LM JSONL.

Key changes from v4:
  1. Two new token types from the caption+speech language-anchor pipeline
     (Phase 6 v4's --captions-dir / --speech-segments-dir):
       <caption>...</caption>  — Qwen2.5-VL frame caption, anchored before
                                  its chunk's <cosmos> block. Buffered like
                                  seed2/cosmos, flushed at avc_lm (in the
                                  order caption -> seed2 -> cosmos, matching
                                  source document order).
       <speech>...</speech>    — inline ASR segment snapped to its chunk,
                                  anchored after </avc_lm> alongside
                                  agent/snac. Emitted immediately, no
                                  dropout, like agent/snac.
     Neither is text-augmented (no synonym replacement / stopword drop):
     both are anchored to an exact chunk, so paraphrasing them would break
     the token-to-moment correspondence that's the point of adding them.
     This is a different treatment from the existing "### Speech:" header
     block below, which is untouched and still augmented/permuted as before
     — the two are intentionally redundant (whole-activity dump vs.
     precisely-timed anchor), a deliberate decision, not an oversight.

  2. count_token_types() gained a `mode` tracker so caption/speech words
     (which have no distinguishing '<...>' prefix) are attributed to their
     own stat buckets instead of silently landing in the catch-all "agent"
     bucket. Stats-only change; does not affect training text content.

Carried over from v4 (unchanged):
  - Per-chunk temporal ordering via a document-order event state machine;
    AVC-LM fires the per-chunk flush.
  - Speech in a dedicated ### Speech: header block (in addition to the new
    inline <speech> above), NOT otherwise scattered into the token sequence.
  - Text header blocks (Title / Context / Keywords / Speech) are shuffled
    among themselves. The token sequence is ALWAYS placed after all text
    blocks to guarantee a consistent text-prompt -> token-generation order.
  - Per-file token counts printed to stdout.

Record filter (unchanged from v3):
  Emit any activity that has <agent> OR <snac> tokens.
  Pure seed2+cosmos activities are skipped.

Drop rates (unchanged from v4):
  AVC-LM          100%   (removed pending ablation)
  Cosmos           50%   (per-chunk independent decision)
  Seed2             0%   (always keep)
  Agent             0%   (always keep)
  SNAC              0%   (always keep)
  Caption           0%   (always keep -- sparse anchor signal)
  Speech (inline)   0%   (always keep -- sparse anchor signal)

Input:  .../final_dataset_adaptive_v4/final_vla_adaptive_rank_*.jsonl   (Phase 6 v4 output)
Output: .../megatron_dataset_v5/flat_*.jsonl

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
    if len(chunks) < 2 or permutation_rate <= 0:
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
# 2026-07-23: FineVideo/OmniVideo-100K's SNAC wrapper is <listen> (ambient audio
# the model perceives, never a reply); roleplay-style data uses <speak> (a
# generated reply). Both hold <snac_N> tokens exactly like the older generic
# <snac> wrapper above -- kept as a separate pattern (not folded into
# _RE_SNAC) so the output preserves which wrapper tag was actually used,
# since that distinction is the whole point of the 2026-07-23 convention.
_RE_LISTEN = re.compile(r'<listen>(.*?)</listen>', re.DOTALL)
_RE_SPEAK  = re.compile(r'<speak>(.*?)</speak>', re.DOTALL)
# Caption blocks (Phase 6 v4): free-text Qwen2.5-VL caption, injected before <cosmos>
_RE_CAPTION = re.compile(r'<caption>(.*?)</caption>', re.DOTALL)
# Inline speech blocks (Phase 6 v4): free-text ASR segment snapped to a chunk,
# injected after </avc_lm> alongside agent/snac. Distinct from the whole-activity
# "### Speech:" header block built from activity["speech_transcript"] below --
# that one is unchanged and still augmented/permuted as before.
_RE_SPEECH_INLINE = re.compile(r'<speech>(.*?)</speech>', re.DOTALL)
# Any single tag <...>
_RE_TAG    = re.compile(r'<[^>]+>')


# ── Core: per-chunk temporal flatten ────────────────────────────────────────

def process_activity_per_chunk(token_str,
                                drop_rate_cosmos=0.5,
                                drop_rate_seed=0.0,
                                drop_rate_snac=0.0):
    """
    Flatten video_tokens into a temporally-ordered list of vocab token strings.

    Phase 6 v4 writes chunks in this order for each 8-frame window:
        [<seed2>N...</seed2>?]  [<caption>text</caption>?]  <cosmos>N...</cosmos>
        <avc_lm>N...</avc_lm>  [<agent><fps_30>...</agent>?]  [<snac><snac_N>...</snac>?]
        [<speech>text</speech>?]

    This function emits per chunk:
        [<caption> word... </caption>?]  [<seed2> <seed2_N>... </seed2>?]
        [<cosmos> <cosmos_N>... </cosmos>?]  [<agent> agent inner tokens </agent>?]
        [<snac> <snac_N>... </snac>?]  [<speech> word... </speech>?]

    seed2/cosmos/agent/snac keep their open/close wrapper tags in the output
    (unlike an earlier version of this function, which dropped them) --
    matching tokenizer_vla_qwen3's registered <seed2>/</seed2> etc. special
    tokens and the laion_emotional_roleplay <snac>/</snac> convention. Decided
    with Van Khue 2026-07-21: explicit boundaries give the model an
    unambiguous "this span is over" signal decoupled from "what modality
    comes next" -- plausibly relevant to the known modality-transition
    failure (model never autonomously switches out of seed2 mode), and the
    tokens were already paid for in vocab (registered, never trained on).

    State machine (processes events in document order):
        caption →  buffer pending_caption
        seed2   →  buffer pending_seed2
        cosmos  →  buffer pending_cosmos
        avc_lm  →  flush, in this order: pending_caption (no dropout, verbatim
                    text), pending_seed2 (dropout), pending_cosmos (dropout)
        agent   →  emit inner named-joint tags directly (no dropout)
        snac    →  emit <snac_N> tags directly (optional dropout)
        speech  →  emit verbatim text wrapped in <speech>/</speech> (no dropout,
                    no augmentation -- same reasoning as caption: this text is
                    snapped to an exact chunk, so paraphrasing it would break
                    the token-to-moment correspondence that's the whole point
                    of adding it)

    Caption is buffered rather than emitted immediately (unlike agent/snac/
    speech) because it appears before <cosmos> in the source string but must
    still respect per-chunk dropout timing semantics the same way seed2/cosmos
    do -- flushing all three together at avc_lm keeps their relative order
    (caption, then seed2, then cosmos) matching the source document order.

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

    for m in _RE_LISTEN.finditer(token_str):
        events.append((m.start(), 'listen', m.group(1)))

    for m in _RE_SPEAK.finditer(token_str):
        events.append((m.start(), 'speak', m.group(1)))

    for m in _RE_CAPTION.finditer(token_str):
        events.append((m.start(), 'caption', m.group(1)))

    for m in _RE_SPEECH_INLINE.finditer(token_str):
        events.append((m.start(), 'speech', m.group(1)))

    if not events:
        return []

    events.sort(key=lambda x: x[0])

    # ── State machine ─────────────────────────────────────────────────────────
    all_output      = []
    pending_seed2   = None   # seed2 payload waiting for its avc_lm trigger
    pending_cosmos  = None   # cosmos payload waiting for its avc_lm trigger
    pending_caption = None   # caption payload waiting for its avc_lm trigger

    for _, etype, payload in events:

        if etype == 'caption':
            pending_caption = payload

        elif etype == 'seed2':
            pending_seed2 = payload

        elif etype == 'cosmos':
            pending_cosmos = payload

        elif etype == 'avc_lm':
            # avc_lm fires → flush pending video tokens for this chunk

            if pending_caption is not None:
                text = pending_caption.strip()
                if text:
                    all_output.append('<caption>')
                    all_output.extend(text.split())
                    all_output.append('</caption>')
            pending_caption = None

            if pending_seed2 is not None and random.random() > drop_rate_seed:
                seed2_toks = [f'<seed2_{n}>' for n in pending_seed2.split() if n.isdigit()]
                if seed2_toks:
                    all_output.append('<seed2>')
                    all_output.extend(seed2_toks)
                    all_output.append('</seed2>')
            pending_seed2 = None

            if pending_cosmos is not None and random.random() > drop_rate_cosmos:
                cosmos_toks = [f'<cosmos_{n}>' for n in pending_cosmos.split() if n.isdigit()]
                if cosmos_toks:
                    all_output.append('<cosmos>')
                    all_output.extend(cosmos_toks)
                    all_output.append('</cosmos>')
            pending_cosmos = None
            # avc_lm payload: discarded (100% drop)

        elif etype == 'agent':
            # Pass through inner named-joint tokens as-is, wrapped
            inner = _RE_TAG.findall(payload)
            if inner:
                all_output.append('<agent>')
                all_output.extend(inner)
                all_output.append('</agent>')

        elif etype == 'snac':
            if random.random() > drop_rate_snac:
                inner = _RE_TAG.findall(payload)
                if inner:
                    all_output.append('<snac>')
                    all_output.extend(inner)
                    all_output.append('</snac>')

        elif etype == 'listen':
            if random.random() > drop_rate_snac:
                inner = _RE_TAG.findall(payload)
                if inner:
                    all_output.append('<listen>')
                    all_output.extend(inner)
                    all_output.append('</listen>')

        elif etype == 'speak':
            if random.random() > drop_rate_snac:
                inner = _RE_TAG.findall(payload)
                if inner:
                    all_output.append('<speak>')
                    all_output.extend(inner)
                    all_output.append('</speak>')

        elif etype == 'speech':
            text = payload.strip()
            if text:
                all_output.append('<speech>')
                all_output.extend(text.split())
                all_output.append('</speech>')

    return all_output


# ── Token type counter ───────────────────────────────────────────────────────

def count_token_types(tokens):
    """Return (n_seed2, n_cosmos, n_agent, n_snac, n_caption, n_speech) counts.

    caption/speech words don't carry a distinguishing '<...>' prefix (they're
    plain English text between '<caption>'/'</caption>' or '<speech>'/'</speech>'
    sentinels), so a running `mode` tracks whether we're currently inside one
    of those wrappers -- otherwise every caption/speech word would silently
    fall into the catch-all 'agent' bucket below and corrupt the reported
    token-type breakdown (it would NOT corrupt the actual training text,
    which is unaffected -- this only affects the printed stats).
    """
    s2 = co = ag = sn = cap = sp = 0
    mode = None  # None | 'caption' | 'speech'
    for tok in tokens:
        if tok == '<caption>':
            mode = 'caption'
            cap += 1
        elif tok == '</caption>':
            mode = None
            cap += 1
        elif tok == '<speech>':
            mode = 'speech'
            sp += 1
        elif tok == '</speech>':
            mode = None
            sp += 1
        elif mode == 'caption':
            cap += 1
        elif mode == 'speech':
            sp += 1
        elif tok in ('<seed2>', '</seed2>') or tok.startswith('<seed2_'):
            s2 += 1
        elif tok in ('<cosmos>', '</cosmos>') or tok.startswith('<cosmos_'):
            co += 1
        elif tok in ('<snac>', '</snac>', '<listen>', '</listen>', '<speak>', '</speak>') or tok.startswith('<snac_'):
            sn += 1
        else:
            ag += 1   # <agent>/</agent>, fps_N, joint open/close, joint_t_N, joint_x/y/z_N
    return s2, co, ag, sn, cap, sp


# ── Per-file worker ──────────────────────────────────────────────────────────

def flatten_one_file(in_path, output_dir, skip_existing,
                     drop_avc, drop_cosmos, drop_seed, drop_snac,
                     synonym_rate, stopword_drop, permute_sentences):
    """
    Flatten one Phase 6 v4 JSONL shard into flat Megatron-LM JSONL.

    Returns a dict with per-file stats for aggregate reporting.
    """
    base     = os.path.basename(in_path)
    out_path = os.path.join(output_dir, f"flat_{base}")

    if skip_existing and os.path.exists(out_path):
        return {"status": "skip", "path": out_path,
                "records": 0, "seed2": 0, "cosmos": 0, "agent": 0, "snac": 0,
                "caption": 0, "speech_inline": 0}

    H = "###"
    written = 0
    total_seed2 = total_cosmos = total_agent = total_snac = total_caption = total_speech_inline = 0

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

                    # Filter: only emit activities with agent OR some audio (snac/listen/speak) tokens.
                    # 2026-07-23: added listen/speak -- FineVideo/OmniVideo-100K now write <listen>,
                    # not <snac>, for their SNAC audio (see _RE_LISTEN/_RE_SPEAK above). Without this,
                    # every audio-only activity (no <agent>) was silently dropped whole, and every
                    # agent-bearing activity kept its agent tokens but lost 100% of its <listen> audio
                    # -- caught via smoke test on final_dataset_adaptive_w24 before the real run
                    # (snac=0 in output despite 254 real <listen> blocks in the matching Phase 6 input).
                    if ("<agent>" not in raw_tokens and "<snac>" not in raw_tokens
                            and "<listen>" not in raw_tokens and "<speak>" not in raw_tokens):
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

                    s2, co, ag, sn, cap, sp_inline = count_token_types(kept_tokens)
                    total_seed2  += s2
                    total_cosmos += co
                    total_agent  += ag
                    total_snac   += sn
                    total_caption += cap
                    total_speech_inline += sp_inline

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
                    # Sentence permutation is skipped when real SNAC audio tokens
                    # are present: SNAC preserves true temporal order, so shuffling
                    # the speech text would mismatch what the model "hears" vs "reads".
                    effective_permute_rate = 0.0 if sn > 0 else permute_sentences
                    speech_chunks = process_transcript_into_chunks(
                        speech, max_words=20,
                        permute_rate=effective_permute_rate,
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

    total = total_seed2 + total_cosmos + total_agent + total_snac + total_caption + total_speech_inline
    return {
        "status":  "done",
        "path":    out_path,
        "records": written,
        "seed2":   total_seed2,
        "cosmos":  total_cosmos,
        "agent":   total_agent,
        "snac":    total_snac,
        "caption": total_caption,
        "speech_inline": total_speech_inline,
        "total":   total,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 7 v5 — per-chunk temporal flatten for Megatron-LM pretraining."
    )
    parser.add_argument(
        "--input-glob",
        default=(
            "/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA"
            "/final_dataset_adaptive_v4/final_vla_adaptive_rank_*.jsonl"
        ),
        help="Glob pattern for Phase 6 v4 JSONL shards (v3 input + caption/speech injected)",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA"
            "/megatron_dataset_v5"
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
    print(f"[Phase7 v5] {len(input_paths)} input files → {args.output_dir}")
    print(f"[Phase7 v5] drop: cosmos={args.drop_cosmos} seed={args.drop_seed} "
          f"snac={args.drop_snac} avc=1.0(fixed) caption=0.0(fixed) speech_inline=0.0(fixed)")
    print(f"[Phase7 v5] workers={min(args.workers, len(input_paths))}")

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
    total_caption = total_speech_inline = 0
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
                total_caption += result["caption"]
                total_speech_inline += result["speech_inline"]
                print(
                    f"[DONE] {os.path.basename(result['path'])} | "
                    f"{result['records']:5d} records | "
                    f"seed2={result['seed2']:,} "
                    f"cosmos={result['cosmos']:,} "
                    f"agent={result['agent']:,} "
                    f"snac={result['snac']:,} "
                    f"caption={result['caption']:,} "
                    f"speech_inline={result['speech_inline']:,} "
                    f"total={result['total']:,}"
                )

    grand_total = (total_seed2 + total_cosmos + total_agent + total_snac
                   + total_caption + total_speech_inline)
    denom = max(grand_total, 1)

    print()
    print("=" * 72)
    print(f"Phase 7 v5 — DONE")
    print(f"  Files processed : {n_done}  ({n_skipped} skipped)")
    print(f"  Total records   : {total_records:,}")
    print(f"  Token counts:")
    print(f"    seed2         : {total_seed2:>15,}  ({total_seed2 / denom * 100:.1f}%)")
    print(f"    cosmos        : {total_cosmos:>15,}  ({total_cosmos / denom * 100:.1f}%)")
    print(f"    agent         : {total_agent:>15,}  ({total_agent / denom * 100:.1f}%)")
    print(f"    snac          : {total_snac:>15,}  ({total_snac / denom * 100:.1f}%)")
    print(f"    caption       : {total_caption:>15,}  ({total_caption / denom * 100:.1f}%)")
    print(f"    speech_inline : {total_speech_inline:>15,}  ({total_speech_inline / denom * 100:.1f}%)")
    print(f"    TOTAL         : {grand_total:>15,}  ({grand_total / 1e9:.3f}B)")
    print("=" * 72)


if __name__ == "__main__":
    main()
