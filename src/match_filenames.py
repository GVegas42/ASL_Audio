import json
import os
import re
import csv
from urllib.parse import urlparse, parse_qs


def extract_youtube_id(url: str):
    try:
        p = urlparse(url)
        if 'youtube' in p.netloc or 'youtu.be' in p.netloc:
            if 'v=' in p.query:
                return parse_qs(p.query)['v'][0]
            # short link youtu.be/ID
            m = re.search(r'youtu\.be/([A-Za-z0-9_-]+)', url)
            if m:
                return m.group(1)
        return None
    except Exception:
        return None


def build_video_index(wlasl_path: str):
    with open(wlasl_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    vid_index = {}  # video_id -> (gloss, split, url)
    for entry in data:
        gloss = entry.get('gloss')
        instances = entry.get('instances') or []
        for inst in instances:
            vid = inst.get('video_id')
            if not vid:
                continue
            vid_index[str(vid)] = {
                'gloss': gloss,
                'split': inst.get('split'),
                'url': inst.get('url')
            }
    return vid_index


def match_files(raw_dir: str, vid_index: dict, out_csv: str):
    files = [f for f in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, f))]
    matched = {}
    video_ids = set(vid_index.keys())

    rows = []

    for fn in files:
        name, ext = os.path.splitext(fn)
        name_lower = name.lower()
        match_type = None
        matched_vid = None

        # Exact numeric name match
        if name in vid_index:
            matched_vid = name
            match_type = 'exact'

        # filename contains video_id
        if matched_vid is None:
            for vid in video_ids:
                if vid in name:
                    matched_vid = vid
                    match_type = 'contains'
                    break

        # try matching via youtube id in url
        if matched_vid is None:
            for vid, meta in vid_index.items():
                yt = extract_youtube_id(meta.get('url') or '')
                if yt and yt in name:
                    matched_vid = vid
                    match_type = 'youtube_id'
                    break

        # try matching numeric sequences in filename
        if matched_vid is None:
            nums = re.findall(r'\d+', name)
            for n in nums:
                if n in video_ids:
                    matched_vid = n
                    match_type = 'numeric_seq'
                    break

        confidence = 0.0
        gloss = ''
        if matched_vid:
            if match_type == 'exact':
                confidence = 1.0
            elif match_type == 'contains':
                confidence = 0.8
            elif match_type == 'youtube_id':
                confidence = 0.7
            elif match_type == 'numeric_seq':
                confidence = 0.6
            meta = vid_index.get(matched_vid, {})
            gloss = meta.get('gloss')
            matched[matched_vid] = matched.get(matched_vid, []) + [fn]

        rows.append({'filename': fn, 'video_id': matched_vid or '', 'gloss': gloss or '', 'match_type': match_type or '', 'confidence': confidence})

    # write csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=['filename', 'video_id', 'gloss', 'match_type', 'confidence'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    unmatched_files = [r for r in rows if not r['video_id']]
    unmatched_vids = [v for v in video_ids if v not in matched]

    print(f'Total raw files: {len(files)}')
    print(f'Matched files: {len(files) - len(unmatched_files)} | Unmatched files: {len(unmatched_files)}')
    print(f'Total video_ids in WLASL: {len(video_ids)} | Unmatched video_ids: {len(unmatched_vids)}')

    # save some reports
    with open(os.path.join(os.path.dirname(out_csv), 'unmatched_files.txt'), 'w', encoding='utf-8') as f:
        for r in unmatched_files:
            f.write(r['filename'] + '\n')

    with open(os.path.join(os.path.dirname(out_csv), 'unmatched_video_ids.txt'), 'w', encoding='utf-8') as f:
        for v in unmatched_vids:
            f.write(v + '\n')

    return rows, unmatched_files, unmatched_vids


if __name__ == '__main__':
    wlasl_path = 'data/WLASL_v0.3.json'
    raw_dir = 'data/raw_videos'
    out_csv = 'data/manifests/mapping.csv'

    vid_index = build_video_index(wlasl_path)
    rows, unmatched_files, unmatched_vids = match_files(raw_dir, vid_index, out_csv)

    print('Mapping written to', out_csv)
    print(f'Example matches (first 10):')
    for r in rows[:10]:
        print(r)
