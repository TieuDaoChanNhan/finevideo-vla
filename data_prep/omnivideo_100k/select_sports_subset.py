#!/usr/bin/env python3
"""
Select the sports/physical-activity subset of OmniVideo-100K by keyword-matching
the video-level `video_summary` field. Coarse heuristic (title/summary text, not
a visual check) -- meant to narrow down which videos are worth spending pose-
pipeline GPU compute on, not a precise content classifier.

Usage:
    python data_prep/omnivideo_100k/select_sports_subset.py
"""

import json
import re

INPUT = '/p/data1/mmlaion/shared/vla/omnivideo_100k_flat/omnivideo_100k_segment_captions.jsonl'
OUTPUT_IDS = '/p/data1/mmlaion/nguyen38/3d-human-pose/data_prep/omnivideo_100k/sports_subset_video_ids.txt'

SPORTS_PATTERN = re.compile(
    r'\b(basketball|soccer|football|boxing|dance|dancing|gym|workout|running|'
    r'fight|fighting|wrestl|tennis|martial art|gymnast|athlete)\b',
    re.IGNORECASE,
)


def main():
    matched = []
    total = 0
    with open(INPUT) as f:
        for line in f:
            total += 1
            d = json.loads(line)
            if SPORTS_PATTERN.search(d.get('video_summary', '')):
                matched.append(d['video_id'])

    with open(OUTPUT_IDS, 'w') as f:
        f.write('\n'.join(matched) + '\n')

    print(f'{len(matched)}/{total} video khop keyword the thao ({100*len(matched)/total:.1f}%)')
    print(f'Da ghi danh sach video_id vao: {OUTPUT_IDS}')


if __name__ == '__main__':
    main()
