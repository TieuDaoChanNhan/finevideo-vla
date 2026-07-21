#!/usr/bin/env python3
"""
Merge pose Phase 5 agent tokens into OmniVideo-100K's Step A video-token
stream, chunk-aligned.

Step A (step_a/step_a_tokenize_video.py) tokenizes video in CHUNK_SIZE=8-frame
chunks at TARGET_FPS=30 -- one <avc_lm> block per chunk, in chunk order. The
pose pipeline (pose/phase1..4_*_omnivideo.py + pipeline_pose/phase5_adaptive_
pchip.py, shared with FineVideo) also uses stride=8 at fps=30, and labels
each output window with `window_id` = its starting frame index. These two
numbering schemes are therefore identical: the Nth <avc_lm> block (0-indexed)
in a video's Step A stream corresponds exactly to window_id = N * 8 in that
video's Phase 5 agent-token file. No time-based interpolation needed, unlike
the FineVideo/OmniVideo fps-mismatch bugs fixed elsewhere in this project --
here both producers already use the same fixed 30fps/stride-8 grid.

Reuses flatten_step_a_video.py's flatten_token_stream()/flatten_record() logic
(same regexes, same seed2/cosmos drop-rate convention) rather than
duplicating it, and adds one thing: after each chunk's cosmos flush, if a
Phase 5 window exists for that chunk's video_id, inject
<agent>{token_str}</agent> -- mirroring the modality order FineVideo's
pipeline_pose/phase6_merge_adaptive.py uses (agent immediately after
seed2/cosmos, avc_lm already discarded).

Only ~799/5,214 videos have any agent tokens at all (Phase 5 only ran on the
1,126-video sports subset, and only produced non-empty output for videos with
at least one clean 8-frame window after the hallucination/YOLO filters) --
every other video's record is unchanged from flatten_step_a_video.py's output.
This mirrors FineVideo's own partial (~12-18%) agent coverage; not a bug.

Usage:
    python data_prep/omnivideo_100k/phase6_merge_omnivideo.py [--skip-existing]
"""
import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_a.flatten_step_a_video import _RE_SIMPLE, _RE_CAPTION, _RE_SPEECH_INLINE, _RE_FIRST_TAG  # noqa: E402

DROP_RATE_COSMOS = 0.5   # same default as flatten_step_a_video.py
DROP_RATE_SEED = 0.0     # always keep
STRIDE = 8               # must match pipeline_pose/phase5_adaptive_pchip.py's --stride

DEFAULT_INPUT_DIR = "/p/data1/mmlaion/shared/vla/omnivideo_100k_video_flat"
DEFAULT_AGENT_DIR = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/omnivideo_100k/pose_agent_tokens_adaptive"
DEFAULT_OUTPUT_DIR = "/p/data1/mmlaion/shared/vla/omnivideo_100k_video_agent_merged"


def load_agent_windows(video_id, agent_dir):
    """Return {window_id: token_str} for this video, or {} if no Phase 5 output."""
    path = os.path.join(agent_dir, f"{video_id}_tokens.jsonl")
    if not os.path.exists(path):
        return {}
    windows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            windows[rec["window_id"]] = rec["token_str"]
    return windows


def flatten_token_stream_with_agent(token_str, agent_windows):
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
    chunk_idx = 0
    n_agent_injected = 0

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
                seed2_toks = [f'<seed2_{n}>' for n in pending_seed2.split() if n.isdigit()]
                if seed2_toks:
                    out.append('<seed2>')
                    out.extend(seed2_toks)
                    out.append('</seed2>')
            pending_seed2 = None

            if pending_cosmos is not None and random.random() > DROP_RATE_COSMOS:
                cosmos_toks = [f'<cosmos_{n}>' for n in pending_cosmos.split() if n.isdigit()]
                if cosmos_toks:
                    out.append('<cosmos>')
                    out.extend(cosmos_toks)
                    out.append('</cosmos>')
            pending_cosmos = None
            # avc_lm payload: always discarded, same as flatten_step_a_video.py

            window_id = chunk_idx * STRIDE
            token_str_agent = agent_windows.get(window_id)
            if token_str_agent:
                out.append('<agent>')
                out.extend(token_str_agent.split())
                out.append('</agent>')
                n_agent_injected += 1
            chunk_idx += 1

        elif etype == 'speech':
            text = payload.strip()
            if text:
                out.append('<speech>')
                out.extend(text.split())
                out.append('</speech>')

    return out, n_agent_injected


def merge_record(text, agent_windows):
    """Return (flat_text, n_agent_injected), or (None, 0) if malformed."""
    m = _RE_FIRST_TAG.search(text)
    if not m:
        return None, 0
    header = text[:m.start()].strip()
    tokens, n_agent = flatten_token_stream_with_agent(text[m.start():], agent_windows)
    if not tokens:
        return None, 0
    return (header + ' ' + ' '.join(tokens)).strip(), n_agent


def merge_one_file(in_path, output_dir, agent_dir, skip_existing):
    base = os.path.basename(in_path)
    out_path = os.path.join(output_dir, base)
    if skip_existing and os.path.exists(out_path):
        return {'file': base, 'skipped': True}

    n_in = n_out = n_malformed = n_videos_with_agent = n_windows_injected = 0
    tmp_path = out_path + '.tmp'
    with open(in_path) as fin, open(tmp_path, 'w') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            d = json.loads(line)
            video_id = d['video_id']
            agent_windows = load_agent_windows(video_id, agent_dir)
            flat_text, n_agent = merge_record(d.get('text', ''), agent_windows)
            if flat_text is None:
                n_malformed += 1
                continue
            fout.write(json.dumps({'video_id': video_id, 'text': flat_text}, ensure_ascii=False) + '\n')
            n_out += 1
            if n_agent > 0:
                n_videos_with_agent += 1
                n_windows_injected += n_agent
    os.replace(tmp_path, out_path)
    return {
        'file': base, 'n_in': n_in, 'n_out': n_out, 'n_malformed': n_malformed,
        'n_videos_with_agent': n_videos_with_agent, 'n_windows_injected': n_windows_injected,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default=DEFAULT_INPUT_DIR)
    ap.add_argument('--agent-dir', default=DEFAULT_AGENT_DIR)
    ap.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    ap.add_argument('--skip-existing', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, 'step_a_rank_*.jsonl')))
    print(f'{len(files)} file dau vao tu {args.input_dir}')

    total_in = total_out = total_malformed = 0
    total_videos_with_agent = total_windows_injected = 0
    for fp in files:
        stats = merge_one_file(fp, args.output_dir, args.agent_dir, args.skip_existing)
        if stats.get('skipped'):
            print(f"{stats['file']}: da co, bo qua")
            continue
        total_in += stats['n_in']
        total_out += stats['n_out']
        total_malformed += stats['n_malformed']
        total_videos_with_agent += stats['n_videos_with_agent']
        total_windows_injected += stats['n_windows_injected']
        print(f"{stats['file']}: {stats['n_in']} -> {stats['n_out']} "
              f"(malformed: {stats['n_malformed']}, videos_with_agent: {stats['n_videos_with_agent']}, "
              f"windows_injected: {stats['n_windows_injected']})")

    print(f'TONG: {total_in} -> {total_out} (malformed: {total_malformed}) | '
          f'videos_with_agent: {total_videos_with_agent} | windows_injected: {total_windows_injected}')


if __name__ == '__main__':
    main()
