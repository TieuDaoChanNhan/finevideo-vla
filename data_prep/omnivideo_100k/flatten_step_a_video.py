#!/usr/bin/env python3
"""
Flatten OmniVideo-100K Step A raw output into Megatron-ready JSONL.

Step A (step_a_tokenize_video.py, run on JUPITER) writes, per video, a single
"text" field: a "### Context: ..." header followed by a chunk-ordered stream
of <seed2>N N N</seed2> [<caption>...</caption>] <cosmos>N N N</cosmos>
<avc_lm>N N N</avc_lm> [<speech>...</speech>] blocks. The payloads inside
<seed2>/<cosmos>/<avc_lm> are raw space-separated integer IDs -- NOT yet the
atomic <seed2_N>/<cosmos_N> vocab entries that tokenizer_vla_qwen3 actually
registers as special tokens. This is the exact same intermediate format
FineVideo's Step A (pipeline_video/pipeline.py) produces; this script is the
OmniVideo-100K equivalent of pipeline_pose/phase7_flatten.py's
process_activity_per_chunk() -- same regexes, same drop-rate convention
(avc_lm payload always discarded, cosmos 50% dropped per chunk, seed2/
caption/speech always kept), adapted for OmniVideo-100K's flatter
one-record-per-video schema (no scenes/activities nesting, no agent/snac
blocks).

Usage:
    python data_prep/omnivideo_100k/flatten_step_a_video.py [--skip-existing]
"""

import argparse
import glob
import json
import os
import random
import re

_RE_SIMPLE = re.compile(r'<(seed2|cosmos|avc_lm)>(.*?)</\1>', re.DOTALL)
_RE_CAPTION = re.compile(r'<caption>(.*?)</caption>', re.DOTALL)
_RE_SPEECH_INLINE = re.compile(r'<speech>(.*?)</speech>', re.DOTALL)
_RE_FIRST_TAG = re.compile(r'<(seed2|cosmos|avc_lm|caption|speech)>')

DROP_RATE_COSMOS = 0.5   # same default as pipeline_pose/phase7_flatten.py
DROP_RATE_SEED = 0.0     # always keep


def flatten_token_stream(token_str):
    events = []
    for m in _RE_SIMPLE.finditer(token_str):
        events.append((m.start(), m.group(1), m.group(2)))
    for m in _RE_CAPTION.finditer(token_str):
        events.append((m.start(), 'caption', m.group(1)))
    for m in _RE_SPEECH_INLINE.finditer(token_str):
        events.append((m.start(), 'speech', m.group(1)))
    events.sort(key=lambda x: x[0])

    out = []
    pending_seed2 = pending_cosmos = pending_caption = None

    for _, etype, payload in events:
        if etype == 'caption':
            pending_caption = payload

        elif etype == 'seed2':
            pending_seed2 = payload

        elif etype == 'cosmos':
            pending_cosmos = payload

        elif etype == 'avc_lm':
            # avc_lm fires -> flush pending video tokens for this chunk
            if pending_caption is not None:
                text = pending_caption.strip()
                if text:
                    out.append('<caption>')
                    out.extend(text.split())
                    out.append('</caption>')
            pending_caption = None

            if pending_seed2 is not None and random.random() > DROP_RATE_SEED:
                out.extend(f'<seed2_{n}>' for n in pending_seed2.split() if n.isdigit())
            pending_seed2 = None

            if pending_cosmos is not None and random.random() > DROP_RATE_COSMOS:
                out.extend(f'<cosmos_{n}>' for n in pending_cosmos.split() if n.isdigit())
            pending_cosmos = None
            # avc_lm payload: always discarded, same as FineVideo Phase 7

        elif etype == 'speech':
            text = payload.strip()
            if text:
                out.append('<speech>')
                out.extend(text.split())
                out.append('</speech>')

    return out


def flatten_record(text):
    """Return flattened text, or None if malformed (no recognizable tag)."""
    m = _RE_FIRST_TAG.search(text)
    if not m:
        return None
    header = text[:m.start()].strip()
    tokens = flatten_token_stream(text[m.start():])
    if not tokens:
        return None
    return (header + ' ' + ' '.join(tokens)).strip()


def flatten_one_file(in_path, output_dir, skip_existing):
    base = os.path.basename(in_path)
    out_path = os.path.join(output_dir, base)
    if skip_existing and os.path.exists(out_path):
        return {'file': base, 'skipped': True}

    n_in = n_out = n_malformed = 0
    tmp_path = out_path + '.tmp'
    with open(in_path) as fin, open(tmp_path, 'w') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            d = json.loads(line)
            flat_text = flatten_record(d.get('text', ''))
            if flat_text is None:
                n_malformed += 1
                continue
            fout.write(json.dumps({'video_id': d['video_id'], 'text': flat_text}) + '\n')
            n_out += 1
    os.replace(tmp_path, out_path)
    return {'file': base, 'n_in': n_in, 'n_out': n_out, 'n_malformed': n_malformed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='/p/data1/mmlaion/shared/vla/omnivideo_100k_video_flat')
    ap.add_argument('--output-dir', default='/p/data1/mmlaion/shared/vla/omnivideo_100k_video_flattened')
    ap.add_argument('--skip-existing', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, 'step_a_rank_*.jsonl')))
    print(f'{len(files)} file dau vao tu {args.input_dir}')

    total_in = total_out = total_malformed = 0
    for fp in files:
        stats = flatten_one_file(fp, args.output_dir, args.skip_existing)
        if stats.get('skipped'):
            print(f"{stats['file']}: da co, bo qua")
            continue
        total_in += stats['n_in']
        total_out += stats['n_out']
        total_malformed += stats['n_malformed']
        print(f"{stats['file']}: {stats['n_in']} -> {stats['n_out']} "
              f"(malformed: {stats['n_malformed']})")

    print(f'TONG: {total_in} -> {total_out} (malformed: {total_malformed})')


if __name__ == '__main__':
    main()
