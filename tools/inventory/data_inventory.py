#!/usr/bin/env python3
"""
data_inventory.py — Count tokens across all multimodal datasets.

Datasets:
  1. FineVideo-VLA flat JSONL (local)
  2. valid_with_seed — VALID + seed2 (HF: 64 shards, download-process-delete)
  3. stack_images3_gzip — StackExchange + seed2 (local tar.gz)
  4. valid_snac — VALID + SNAC audio tokens (HF: MixtureVitae-Omni, 6 files)

Checkpoint: inventory_checkpoint.json (resume after interruption)
Charts:     data_inventory_charts.png

Usage:
  python data_inventory.py                      # full run
  python data_inventory.py --skip-valid-with-seed --skip-valid-snac   # fast test
  python data_inventory.py --only-shard 48      # test with cached shard 48
"""

import argparse
import collections
import glob
import gzip
import json
import os
import re
import sys
import tarfile
import time
from pathlib import Path

import requests

# ─── Paths ───────────────────────────────────────────────────────────────────

FINEVIDEO_DIR   = "/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/megatron_dataset_v2"
STACK_LOCAL_DIR = ("/p/data1/mmlaion/mixture-vitae/shared/"
                   "mixture-vitae-backup-MixtureVitae-Backup/data/stack_images3_gzip")
CACHE_BASE      = "/p/data1/mmlaion/nguyen38/inventory_cache"
TMP_DIR         = os.path.join(CACHE_BASE, "tmp")
SNAC_CACHE_DIR  = os.path.join(CACHE_BASE, "hf_snac")
# Already-downloaded shard from previous investigations
HF_SHARD_CACHE  = os.path.join(CACHE_BASE, "hf_shards")

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "inventory_checkpoint_v2.json")
CHARTS_OUT      = os.path.join(SCRIPT_DIR, "data_inventory_charts.png")

HF_BACKUP_REPO  = "mixture-vitae-backup/MixtureVitae-Backup"
HF_OMNI_REPO    = "mixture-vitae/MixtureVitae-Omni"
HF_BASE_URL     = "https://huggingface.co/datasets"

VWS_NUM_SHARDS  = 64

# ─── Token patterns ──────────────────────────────────────────────────────────

PATTERNS = {
    'seed2':  re.compile(r'<seed2_\d+>'),
    'seed':   re.compile(r'<seed_\d+>'),       # MV-Omni uses <seed_N>, not <seed2_N>
    'cosmos': re.compile(r'<cosmos_\d+>'),
    'avclm':  re.compile(r'<avclm_\d+>'),
    'snac':   re.compile(r'<snac_\d+>'),
    'agent':  re.compile(r'<fps_\d+>|<[a-z_]+_[txyz]_\d+>'),
}
TOKEN_TYPES = list(PATTERNS.keys()) + ['text']
_ANGLE_RE = re.compile(r'^<[^>]+>$')


def count_tokens(text: str) -> dict:
    """Count all VLA token types + text words in one text string."""
    c = {}
    for k, pat in PATTERNS.items():
        c[k] = len(pat.findall(text))
    c['text'] = sum(1 for w in text.split() if w and not _ANGLE_RE.match(w))
    return c


def add_counts(dst: dict, src: dict):
    for k, v in src.items():
        if isinstance(v, (int, float)):
            dst[k] = dst.get(k, 0) + v


def zero_counts() -> dict:
    return {k: 0 for k in TOKEN_TYPES}


def fmt(n: int) -> str:
    if n == 0:  return "—"
    if n >= 1e9: return f"{n/1e9:.2f}B"
    if n >= 1e6: return f"{n/1e6:.1f}M"
    if n >= 1e3: return f"{n/1e3:.1f}K"
    return str(n)


def elapsed(t0: float) -> str:
    s = int(time.time() - t0)
    if s >= 3600:
        return f"{s//3600}h{(s%3600)//60:02d}m{s%60:02d}s"
    return f"{s//60}m{s%60:02d}s"


# ─── Checkpoint ──────────────────────────────────────────────────────────────

def load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            n = len(data.get('completed', {}))
            print(f"Loaded checkpoint ({n} completed files): {path}")
            return data
        except Exception as e:
            print(f"Warning: checkpoint unreadable ({e}), starting fresh")
    return {'completed': {}}


def save_checkpoint(path: str, state: dict):
    import datetime
    state['last_updated'] = datetime.datetime.now().isoformat()
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


# ─── HuggingFace download ─────────────────────────────────────────────────────

def _hf_token() -> str:
    tok = os.environ.get('HF_TOKEN', '')
    if not tok:
        local_token_file = os.path.join(SCRIPT_DIR, '.hf_token')
        for candidate in [local_token_file,
                          os.path.expanduser('~/.huggingface/token'),
                          os.path.expanduser('~/.cache/huggingface/token')]:
            if os.path.exists(candidate):
                lines = [ln.strip() for ln in open(candidate).read().splitlines()]
                lines = [ln for ln in lines if ln and not ln.startswith('#')]
                if lines:
                    tok = lines[0]
                    break
    return tok


def hf_url(repo: str, path: str) -> str:
    return f"{HF_BASE_URL}/{repo}/resolve/main/{path}?download=true"


def download_with_progress(url: str, dest: str) -> bool:
    """Download url to dest, print progress every ~256 MB. Returns True on success."""
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    token = _hf_token()
    headers = {'Authorization': f'Bearer {token}'} if token else {}
    try:
        r = requests.get(url, stream=True, headers=headers, timeout=120)
        r.raise_for_status()
    except Exception as e:
        print(f"  DOWNLOAD FAILED {url}: {e}", flush=True)
        return False

    total = int(r.headers.get('content-length', 0))
    downloaded = 0
    t0 = time.time()
    chunk_size = 1 << 17        # 128 KB
    report_every = 1 << 28     # 256 MB
    next_report = report_every

    try:
        with open(dest + '.tmp', 'wb') as f:
            for chunk in r.iter_content(chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    spd = downloaded / max(time.time() - t0, 0.01) / 1e6
                    pct = downloaded / total * 100 if total else 0
                    eta = (total - downloaded) / max(spd * 1e6, 1) if total else 0
                    print(f"  ↓ {downloaded/1e9:.2f}/{total/1e9:.2f} GB  "
                          f"({pct:.0f}%)  {spd:.1f} MB/s  ETA {eta:.0f}s", flush=True)
                    next_report = downloaded + report_every
    except Exception as e:
        print(f"  DOWNLOAD ERROR: {e}", flush=True)
        try:
            os.remove(dest + '.tmp')
        except OSError:
            pass
        return False

    os.replace(dest + '.tmp', dest)
    spd = downloaded / max(time.time() - t0, 0.01) / 1e6
    print(f"  ↓ done: {downloaded/1e9:.2f} GB  {spd:.1f} MB/s", flush=True)
    return True


# ─── Tar scanner ─────────────────────────────────────────────────────────────

def _process_seed2_jsonl_bytes(raw: bytes) -> dict:
    """Parse JSONL bytes and count seed2 tokens."""
    c = zero_counts()
    try:
        text_content = raw.decode('utf-8', errors='replace')
    except Exception:
        return c
    for line in text_content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(rec, list):
            c['seed2'] = c.get('seed2', 0) + len(rec)
        elif isinstance(rec, dict):
            text = rec.get('text', rec.get('seed2_text', ''))
            if text:
                add_counts(c, count_tokens(text))
    return c


def scan_tar_for_seed2(path: str, label: str = '') -> dict:
    """
    Stream a tar.gz archive, find *_seed2.jsonl members, count tokens.
    Recursively handles inner tar.gz archives (one level deep).
    Skips .png, .ogg, and all other non-JSONL members without reading their bytes.
    Returns count dict with extra keys: _seed2_files, _total_members.
    """
    counts = zero_counts()
    counts['_seed2_files'] = 0
    counts['_total_members'] = 0
    t0 = time.time()

    def _scan(tar_obj: tarfile.TarFile, depth: int):
        for member in tar_obj:
            counts['_total_members'] += 1
            if not member.isfile() or member.size == 0:
                continue
            name = member.name

            if name.endswith('_seed2.jsonl'):
                counts['_seed2_files'] += 1
                try:
                    fobj = tar_obj.extractfile(member)
                    if fobj is not None:
                        add_counts(counts, _process_seed2_jsonl_bytes(fobj.read()))
                except Exception as e:
                    print(f"    SKIP member {name}: {e}", flush=True)

                if counts['_seed2_files'] % 500 == 0:
                    print(f"    {label}members: {counts['_total_members']:,}  "
                          f"seed2_files: {counts['_seed2_files']:,}  "
                          f"seed2_tokens: {fmt(counts['seed2'])}  "
                          f"elapsed: {elapsed(t0)}", flush=True)

            elif depth == 0 and name.endswith(('.tar.gz', '.tgz', '.tar')):
                mode = 'r|gz' if name.endswith(('.tar.gz', '.tgz')) else 'r|'
                try:
                    fobj = tar_obj.extractfile(member)
                    if fobj is not None:
                        with tarfile.open(fileobj=fobj, mode=mode) as inner:
                            _scan(inner, depth=1)
                except Exception as e:
                    print(f"    SKIP inner tar {name}: {e}", flush=True)
            # .png, .ogg, etc. — advance without reading data

    try:
        with tarfile.open(path, 'r:gz') as tar:
            _scan(tar, depth=0)
    except Exception as e:
        print(f"  SKIP broken archive {path}: {e}", flush=True)

    return counts


# ─── Gzip JSONL scanner ──────────────────────────────────────────────────────

def scan_gzip_jsonl(path: str, label: str = '') -> dict:
    """Count all token types in a gzip-compressed JSONL file."""
    counts = zero_counts()
    counts['_records'] = 0
    t0 = time.time()
    report_every = 50_000

    try:
        with gzip.open(path, 'rt', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(rec, dict):
                    continue
                text = rec.get('text', '')
                if text:
                    add_counts(counts, count_tokens(text))
                counts['_records'] += 1
                if counts['_records'] % report_every == 0:
                    print(f"    {label}records: {counts['_records']:,}  "
                          f"snac: {fmt(counts['snac'])}  seed: {fmt(counts['seed'])}  "
                          f"elapsed: {elapsed(t0)}", flush=True)
    except EOFError:
        print(f"  WARNING: truncated gzip {path} — "
              f"processed {counts.get('_records',0):,} records")
    except Exception as e:
        print(f"  SKIP broken gzip {path}: {e}")

    return counts


# ─── Section A: FineVideo ─────────────────────────────────────────────────────

def process_finevideo(state: dict, checkpoint_path: str) -> dict:
    print('\n' + '='*72)
    print('SECTION A: FineVideo-VLA flat JSONL (local)')
    print('='*72, flush=True)

    files = sorted(glob.glob(os.path.join(FINEVIDEO_DIR, 'flat_*.jsonl')))
    if not files:
        print(f'  No flat_*.jsonl files in {FINEVIDEO_DIR}')
        return zero_counts()
    print(f'  {len(files)} files in {FINEVIDEO_DIR}')

    total = zero_counts()
    t0 = time.time()
    n = len(files)

    for i, fpath in enumerate(files, 1):
        key = 'fv:' + os.path.basename(fpath)
        if key in state['completed']:
            add_counts(total, state['completed'][key])
            print(f'  [{i:3d}/{n}] SKIP {os.path.basename(fpath)} (done)', flush=True)
            continue

        counts = zero_counts()
        counts['_records'] = 0
        ft0 = time.time()
        try:
            with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = rec.get('text', '')
                    if text:
                        add_counts(counts, count_tokens(text))
                    counts['_records'] += 1
        except Exception as e:
            print(f'  SKIP broken file {fpath}: {e}')

        row = {k: counts[k] for k in TOKEN_TYPES}
        add_counts(total, row)
        state['completed'][key] = row
        save_checkpoint(checkpoint_path, state)

        recs = counts.get('_records', 0)
        print(f'  [{i:3d}/{n}] {os.path.basename(fpath)}  '
              f'records: {recs:,}  seed2: {fmt(counts["seed2"])}  '
              f'cosmos: {fmt(counts["cosmos"])}  avclm: {fmt(counts["avclm"])}  '
              f'agent: {fmt(counts["agent"])}  text: {fmt(counts["text"])}  '
              f'file: {time.time()-ft0:.0f}s  total: {elapsed(t0)}', flush=True)

    _print_section_total('FineVideo-VLA', total)
    return total


# ─── Section B: valid_with_seed ───────────────────────────────────────────────

def process_valid_with_seed(state: dict, checkpoint_path: str,
                             only_shard: int = None) -> dict:
    print('\n' + '='*72)
    print(f'SECTION B: valid_with_seed ({VWS_NUM_SHARDS} HF shards, '
          'download → scan → keep)')
    print(f'  Saving shards to: {HF_SHARD_CACHE}')
    print('='*72, flush=True)

    os.makedirs(HF_SHARD_CACHE, exist_ok=True)

    if only_shard is not None:
        indices = [only_shard]
        print(f'  Test mode: only shard {only_shard}')
    else:
        indices = list(range(VWS_NUM_SHARDS))

    total = zero_counts()
    t0 = time.time()

    for i in indices:
        shard_name = f'valid_with_seed_shard_{i:05d}.tar.gz'
        hf_path    = f'data/valid_with_seed/{shard_name}'
        key        = 'vws:' + shard_name

        if key in state['completed']:
            add_counts(total, state['completed'][key])
            print(f'  [{i+1:2d}/{VWS_NUM_SHARDS}] SKIP {shard_name} (done)', flush=True)
            continue

        # All shards saved permanently to HF_SHARD_CACHE
        dest = os.path.join(HF_SHARD_CACHE, shard_name)
        if os.path.exists(dest):
            print(f'  [{i+1:2d}/{VWS_NUM_SHARDS}] Using cached: {dest}', flush=True)
        else:
            url = hf_url(HF_BACKUP_REPO, hf_path)
            print(f'  [{i+1:2d}/{VWS_NUM_SHARDS}] Downloading {shard_name} '
                  f'to {HF_SHARD_CACHE} ...', flush=True)
            if not download_with_progress(url, dest):
                print(f'  SKIP {shard_name}: download failed')
                continue

        print(f'  [{i+1:2d}/{VWS_NUM_SHARDS}] Scanning {shard_name} ...', flush=True)
        ft0 = time.time()
        counts = scan_tar_for_seed2(dest, label=f'shard {i:05d}: ')

        row = {k: counts[k] for k in TOKEN_TYPES}
        add_counts(total, row)
        state['completed'][key] = row
        save_checkpoint(checkpoint_path, state)

        sf = counts.get('_seed2_files', 0)
        print(f'  [{i+1:2d}/{VWS_NUM_SHARDS}] {shard_name} DONE  '
              f'seed2_files: {sf:,}  seed2: {fmt(counts["seed2"])}  '
              f'scan: {time.time()-ft0:.0f}s  total: {elapsed(t0)}', flush=True)

    _print_section_total('valid_with_seed', total)
    return total


# ─── Section C: stack_images3_gzip ───────────────────────────────────────────

def process_stack_images3(state: dict, checkpoint_path: str) -> dict:
    print('\n' + '='*72)
    print('SECTION C: stack_images3_gzip (local tar.gz archives)')
    print('='*72, flush=True)

    archives = sorted(glob.glob(os.path.join(STACK_LOCAL_DIR, '*.tar.gz')))
    if not archives:
        print(f'  No *.tar.gz files in {STACK_LOCAL_DIR}')
        return zero_counts()
    print(f'  {len(archives)} archives found')

    total = zero_counts()
    t0 = time.time()
    n = len(archives)

    for i, apath in enumerate(archives, 1):
        key = 'stack:' + os.path.basename(apath)
        if key in state['completed']:
            add_counts(total, state['completed'][key])
            print(f'  [{i:2d}/{n}] SKIP {os.path.basename(apath)} (done)', flush=True)
            continue

        size_gb = os.path.getsize(apath) / 1e9
        print(f'  [{i:2d}/{n}] {os.path.basename(apath)} ({size_gb:.1f} GB) ...',
              flush=True)
        ft0 = time.time()
        counts = scan_tar_for_seed2(apath, label=f'{os.path.basename(apath)}: ')

        row = {k: counts[k] for k in TOKEN_TYPES}
        add_counts(total, row)
        state['completed'][key] = row
        save_checkpoint(checkpoint_path, state)

        sf = counts.get('_seed2_files', 0)
        print(f'  [{i:2d}/{n}] {os.path.basename(apath)} DONE  '
              f'seed2_files: {sf:,}  seed2: {fmt(counts["seed2"])}  '
              f'scan: {time.time()-ft0:.0f}s  total: {elapsed(t0)}', flush=True)

    _print_section_total('stack_images3_gzip', total)
    return total


# ─── Section D: valid_snac ────────────────────────────────────────────────────

def process_valid_snac(state: dict, checkpoint_path: str) -> dict:
    print('\n' + '='*72)
    print('SECTION D: valid_snac — MixtureVitae-Omni HF download')
    print('='*72, flush=True)

    os.makedirs(SNAC_CACHE_DIR, exist_ok=True)
    snac_files = [f'data/data/valid_snac_{i}.jsonl.gz' for i in range(6)]
    total = zero_counts()
    t0 = time.time()
    n = len(snac_files)

    for i, hf_path in enumerate(snac_files, 1):
        fname = os.path.basename(hf_path)
        key   = 'snac:' + fname
        dest  = os.path.join(SNAC_CACHE_DIR, fname)

        if key in state['completed']:
            add_counts(total, state['completed'][key])
            print(f'  [{i}/{n}] SKIP {fname} (done)', flush=True)
            continue

        if not os.path.exists(dest):
            url = hf_url(HF_OMNI_REPO, hf_path)
            print(f'  [{i}/{n}] Downloading {fname} ...', flush=True)
            if not download_with_progress(url, dest):
                print(f'  SKIP {fname}: download failed')
                continue
        else:
            size_gb = os.path.getsize(dest) / 1e9
            print(f'  [{i}/{n}] Cached {fname} ({size_gb:.1f} GB)', flush=True)

        print(f'  [{i}/{n}] Scanning {fname} ...', flush=True)
        ft0 = time.time()
        counts = scan_gzip_jsonl(dest, label=f'{fname}: ')

        row = {k: counts[k] for k in TOKEN_TYPES}
        add_counts(total, row)
        state['completed'][key] = row
        save_checkpoint(checkpoint_path, state)

        recs = counts.get('_records', 0)
        print(f'  [{i}/{n}] {fname} DONE  '
              f'records: {recs:,}  snac: {fmt(counts["snac"])}  '
              f'seed: {fmt(counts["seed"])}  text: {fmt(counts["text"])}  '
              f'scan: {time.time()-ft0:.0f}s  total: {elapsed(t0)}', flush=True)

    _print_section_total('valid_snac', total)
    return total


# ─── Summary & charts ─────────────────────────────────────────────────────────

def _print_section_total(label: str, counts: dict):
    parts = [f'{k}={fmt(counts[k])}' for k in TOKEN_TYPES if counts.get(k, 0) > 0]
    print(f'\n  {label} TOTAL: {", ".join(parts) if parts else "(empty)"}')


DATASET_META = [
    ('finevideo',       'FineVideo-VLA',               'fv:'),
    ('valid_with_seed', 'valid_with_seed (64 HF shards)', 'vws:'),
    ('stack_images3',   'stack_images3_gzip',           'stack:'),
    ('valid_snac',      'valid_snac (MV-Omni)',         'snac:'),
]

# Display types: seed + seed2 are the same family — merged under 'seed2' in all charts/summaries
DISPLAY_TYPES = ['seed2', 'cosmos', 'avclm', 'snac', 'agent', 'text']

TOKEN_COLORS = {
    'seed2':  '#4e79a7',
    'cosmos': '#f28e2b',
    'avclm':  '#e15759',
    'snac':   '#59a14f',
    'agent':  '#b07aa1',
    'text':   '#9c755f',
}


def _merge_seed(c: dict) -> dict:
    """Collapse 'seed' (<seed_N> from MV-Omni) into 'seed2' for display/charting."""
    out = {k: c.get(k, 0) for k in DISPLAY_TYPES}
    out['seed2'] += c.get('seed', 0)
    return out


def totals_from_checkpoint(state: dict) -> dict:
    """Rebuild per-dataset totals from checkpoint data (works even for skipped sections)."""
    result = {ds: zero_counts() for ds, _, _ in DATASET_META}
    for key, row in state.get('completed', {}).items():
        for ds, _, prefix in DATASET_META:
            if key.startswith(prefix):
                add_counts(result[ds], row)
                break
    return result


def print_summary(totals: dict):
    print('\n' + '='*110)
    print('DATA INVENTORY SUMMARY  (seed + seed2 merged as seed2)')
    print('='*110)
    cols = DISPLAY_TYPES
    header = f"  {'Dataset':<38s}" + ''.join(f"{c:>9s}" for c in cols) + f"  {'TOTAL':>10s}"
    print(header)
    print('-'*110)

    grand = {k: 0 for k in cols}
    for ds, label, _ in DATASET_META:
        c = _merge_seed(totals.get(ds, zero_counts()))
        row_total = sum(c.get(k, 0) for k in cols)
        print(f"  {label:<38s}" + ''.join(f"{fmt(c.get(k,0)):>9s}" for k in cols)
              + f"  {fmt(row_total):>10s}")
        for k in cols:
            grand[k] += c.get(k, 0)

    print('-'*110)
    grand_total = sum(grand.values())
    print(f"  {'TOTAL':<38s}" + ''.join(f"{fmt(grand.get(k,0)):>9s}" for k in cols)
          + f"  {fmt(grand_total):>10s}")
    print('='*110)


def save_charts(totals: dict):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        import matplotlib.colors as mcolors
    except Exception as e:
        print(f'WARNING: matplotlib unavailable, skipping charts: {e}')
        return

    # Merge seed → seed2 for all datasets
    display = {ds: _merge_seed(totals.get(ds, zero_counts())) for ds, _, _ in DATASET_META}

    # Active display types (have data anywhere)
    active = [k for k in DISPLAY_TYPES
              if any(display[ds].get(k, 0) > 0 for ds, _, _ in DATASET_META)]

    # Grand totals
    overall = {k: sum(display[ds].get(k, 0) for ds, _, _ in DATASET_META) for k in active}
    grand_total = sum(overall.values())

    # ── Layout: left = overall pie, right = summary table ────────────────────────
    fig = plt.figure(figsize=(20, 9))
    fig.patch.set_facecolor('#ffffff')
    gs = GridSpec(1, 2, figure=fig, wspace=0.08, width_ratios=[1, 1.6])

    # ── Left: overall pie ─────────────────────────────────────────────────────────
    ax_pie = fig.add_subplot(gs[0, 0])
    pie_vals = [overall[k] for k in active]
    pct_lbls = [f'{overall[k]/grand_total*100:.1f}%\n{fmt(overall[k])}' for k in active]
    pie_cols = [TOKEN_COLORS.get(k, '#aaa') for k in active]
    wedges, _, autotexts = ax_pie.pie(
        pie_vals,
        colors=pie_cols,
        autopct='',          # we use custom labels instead
        startangle=140,
        wedgeprops={'linewidth': 1.2, 'edgecolor': 'white'},
        pctdistance=0.78,
    )
    # Legend with token type + count
    legend_labels = [f'{k}  —  {fmt(overall[k])}  ({overall[k]/grand_total*100:.1f}%)'
                     for k in active]
    ax_pie.legend(wedges, legend_labels, loc='lower center',
                  bbox_to_anchor=(0.5, -0.18), fontsize=9.5,
                  frameon=False, ncol=1)
    ax_pie.set_title(f'All Datasets Combined\n{fmt(grand_total)} tokens total',
                     fontsize=13, fontweight='bold', pad=12)

    # ── Right: per-dataset table ──────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[0, 1])
    ax_tbl.axis('off')

    # Build table data
    col_headers = active + ['TOTAL']
    row_labels   = [label for _, label, _ in DATASET_META] + ['TOTAL']

    grand_row = {k: overall[k] for k in active}

    cell_text = []
    cell_colors = []

    for ds, label, _ in DATASET_META:
        c = display[ds]
        ds_total = sum(c.get(k, 0) for k in active)
        row_vals = [fmt(c.get(k, 0)) for k in active] + [fmt(ds_total)]
        cell_text.append(row_vals)

        # Color each cell: light tint of the token color; grey for zero; last col white
        row_colors = []
        for k in active:
            v = c.get(k, 0)
            if v == 0:
                row_colors.append('#f0f0f0')
            else:
                base = mcolors.to_rgb(TOKEN_COLORS.get(k, '#aaa'))
                # blend toward white: 0.25 saturation
                tint = tuple(0.25 * b + 0.75 for b in base)
                row_colors.append(tint)
        row_colors.append('#e8e8e8')   # TOTAL column
        cell_colors.append(row_colors)

    # TOTAL row
    total_vals = [fmt(overall[k]) for k in active] + [fmt(grand_total)]
    cell_text.append(total_vals)
    total_colors = []
    for k in active:
        base = mcolors.to_rgb(TOKEN_COLORS.get(k, '#aaa'))
        tint = tuple(0.4 * b + 0.6 for b in base)   # slightly deeper for total row
        total_colors.append(tint)
    total_colors.append('#c8c8c8')
    cell_colors.append(total_colors)

    tbl = ax_tbl.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_headers,
        cellColours=cell_colors,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 2.0)   # taller rows

    # Bold header row and row-label column
    for (r, c_), cell in tbl.get_celld().items():
        cell.set_edgecolor('#bbbbbb')
        if r == 0 or c_ == -1:
            cell.set_text_props(fontweight='bold')
        # TOTAL row (last data row)
        if r == len(DATASET_META) + 1:
            cell.set_text_props(fontweight='bold')

    ax_tbl.set_title('Token counts by dataset and type\n(seed and seed2 merged as seed2)',
                     fontsize=11, fontweight='bold', pad=14)

    fig.suptitle('Data Inventory — Multimodal Token Distribution',
                 fontsize=15, fontweight='bold', y=1.02)

    try:
        plt.savefig(CHARTS_OUT, dpi=150, bbox_inches='tight')
        print(f'\nChart saved → {CHARTS_OUT}')
    except Exception as e:
        print(f'WARNING: could not save chart: {e}')
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Count tokens across all multimodal datasets.')
    p.add_argument('--skip-finevideo',       action='store_true',
                   help='Skip FineVideo section')
    p.add_argument('--skip-valid-with-seed', action='store_true',
                   help='Skip valid_with_seed section (64 HF shards)')
    p.add_argument('--skip-stack',           action='store_true',
                   help='Skip stack_images3_gzip section')
    p.add_argument('--skip-valid-snac',      action='store_true',
                   help='Skip valid_snac section')
    p.add_argument('--only-shard',           type=int, default=None,
                   help='valid_with_seed: process only this shard index (0–63). '
                        'Implies --skip-finevideo --skip-stack --skip-valid-snac '
                        'unless those are explicitly requested.')
    p.add_argument('--checkpoint', default=CHECKPOINT_PATH,
                   help=f'Checkpoint file (default: {CHECKPOINT_PATH})')
    args = p.parse_args()

    t_start = time.time()
    state = load_checkpoint(args.checkpoint)

    session_totals = {}  # only what ran this session

    if not args.skip_finevideo:
        session_totals['finevideo'] = process_finevideo(state, args.checkpoint)

    if not args.skip_valid_with_seed:
        session_totals['valid_with_seed'] = process_valid_with_seed(
            state, args.checkpoint, args.only_shard)

    if not args.skip_stack:
        session_totals['stack_images3'] = process_stack_images3(state, args.checkpoint)

    if not args.skip_valid_snac:
        session_totals['valid_snac'] = process_valid_snac(state, args.checkpoint)

    # Always rebuild totals from the full checkpoint so the summary includes
    # sections processed in previous runs (even if skipped this session).
    all_totals = totals_from_checkpoint(state)

    print_summary(all_totals)
    save_charts(all_totals)

    print(f'\nTotal wall time: {elapsed(t_start)}')
    print(f'Checkpoint: {args.checkpoint}')


if __name__ == '__main__':
    main()
