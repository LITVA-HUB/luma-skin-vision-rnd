"""Freeze and sparsely acquire original Beyond RGB phone records, no pixel decoding."""
import argparse
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

from download_cc_v2_fresh import byte_record, safe_destination, unpack_member

from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT/'docs/data/provenance/mobile_screen_2026_09_11'
LOCK = PROVENANCE/'beyond_rgb_selection.json'
PREFIX = 'beyond-unzip/beyondRGB/'
REQUIRED = ('CRI/CRI.txt',) + tuple(
    f'{mode}/{camera}{suffix}' for camera in ('samsung', 'oppo')
    for mode in ('NT', 'WT') for suffix in (('.h5', '_tags.json', '_cc_detection.json') if mode == 'WT' else ('.h5', '_tags.json')))


def split_ranges(sizes, start, count):
    if not sizes or any(s <= 0 for s in sizes) or start < 0 or count <= 0 or start+count > sum(sizes):
        raise ValueError('Range must lie inside concatenated archive')
    output, base = [], 0
    end = start+count
    for part, size in enumerate(sizes):
        left, right = max(start, base), min(end, base+size)
        if left < right:
            output.append((part, left-base, right-base-1))
        base += size
    return output


def scene_members(rows, scene):
    prefix = PREFIX+scene+'/'
    if not all(prefix+name in rows for name in REQUIRED):
        return None
    chosen = [rows[prefix+name] for name in REQUIRED]
    chosen += [row for name, row in rows.items() if name.startswith(prefix) and
               name.endswith(('/samsung_blurred_areas_detection.json', '/oppo_blurred_areas_detection.json'))]
    return sorted(chosen, key=lambda row: row['path'])


def freeze():
    if LOCK.exists():
        raise FileExistsError('Existing phone selection lock is immutable')
    metadata = json.loads((PROVENANCE/'beyond_rgb_zenodo.json').read_text(encoding='utf-8'))
    if metadata['metadata']['license']['id'] != 'cc-by-4.0':
        raise ValueError('Original attribution license missing')
    directory = json.loads((PROVENANCE/'beyond_rgb_members.json').read_text(encoding='utf-8'))
    if sha256(PROVENANCE/'beyond_rgb_directory.bin') != directory['directory_sha256']:
        raise ValueError('Central directory changed')
    rows = {r['path']: r for r in directory['members']}
    candidates = {}
    for split in ('train', 'test'):
        names = (PROVENANCE/f'beyond_rgb_{split}_dataset_paper_git.txt').read_text().splitlines()
        candidates[split] = sorted({s for s in names if s.startswith('outdoor/')},
                                   key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    chosen, excluded = [], []
    for split in ('train', 'test'):
        for scene in candidates[split]:
            members = scene_members(rows, scene)
            if members is None:
                excluded.append({'scene': scene, 'split': split, 'reason': 'Required paired-phone reference record absent'})
                continue
            chosen.append({'scene': scene, 'role': 'loader' if split == 'train' else 'reserved_test', 'members': members})
            if split == 'train' and sum(s['role'] == 'loader' for s in chosen) == 3:
                break
    if len({s['scene'] for s in chosen}) != len(chosen):
        raise ValueError('Scene roles overlap')
    payload = sum(m['compressed_bytes'] + 1000 for s in chosen for m in s['members'])
    if payload > 4_000_000_000:
        raise ValueError('Predeclared4GB payload budget insufficient; do not silently change selection')
    source_names = ['beyond_rgb_zenodo.json', 'beyond_rgb_members.json', 'beyond_rgb_directory.bin',
                    'beyond_rgb_source.json', 'beyond_rgb_train_dataset_paper_git.txt',
                    'beyond_rgb_test_dataset_paper_git.txt']
    lock = {'status': 'FROZEN BEFORE ANY NEW IMAGE OR GT VALUE DECODING',
            'license': 'Original author-linked Zenodo CC BY4.0; attribution required',
            'parts': sorted(metadata['files'], key=lambda f: f['key']),
            'source_sha256': {name: sha256(PROVENANCE/name) for name in source_names},
            'source_code_sha256': sha256(Path(__file__)), 'scenes': chosen, 'excluded': excluded,
            'payload_upper_bound_bytes': payload, 'traffic_cap_bytes': 5_000_000_000,
            'note': 'Loader scenes are original TRAIN; paired-phone test scenes remain reserved, never architecture-selection rows.'}
    write_json(LOCK, lock)
    print(json.dumps({'lock_sha256': sha256(LOCK), 'scenes': len(chosen), 'payload_upper_bound_bytes': payload,
                      'loader': sum(s['role'] == 'loader' for s in chosen), 'excluded': excluded}))


def download(args):
    if sha256(LOCK) != args.lock_sha256:
        raise ValueError('Explicit frozen lock digest mismatch')
    lock = json.loads(LOCK.read_text(encoding='utf-8'))
    if sha256(Path(__file__)) != lock['source_code_sha256']:
        raise ValueError('Acquisition code changed after freeze')
    for name, digest in lock['source_sha256'].items():
        if sha256(PROVENANCE/name) != digest:
            raise ValueError('Acquisition source changed: '+name)
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    ledger = root/'traffic.jsonl'
    used = sum(json.loads(line)['bytes'] for line in ledger.read_text().splitlines()) if ledger.exists() else 0
    sizes = [p['size'] for p in lock['parts']]

    def fetch(start, count):
        nonlocal used
        result = []
        for part, left, right in split_ranges(sizes, start, count):
            size = right-left+1
            if used+size > lock['traffic_cap_bytes']:
                raise ValueError('Total traffic cap exhausted')
            url = lock['parts'][part]['links']['self']
            request = urllib.request.Request(url, headers={'Range': f'bytes={left}-{right}', 'Accept-Encoding': 'identity'})
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes {left}-{right}/{sizes[part]}':
                    raise ValueError('Exact HTTP206 range required; refusing full archive')
                raw = response.read(size+1)
            used += len(raw)
            with ledger.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'part': part, 'start': left, 'end': right, 'bytes': len(raw)})+'\n')
            if len(raw) != size:
                raise ValueError('Unexpected range response size')
            result.append(raw)
        return b''.join(result)

    records = []
    for scene in lock['scenes']:
        if args.role != 'all' and scene['role'] != args.role:
            continue
        for member in scene['members']:
            path = safe_destination(root, member['path'])
            if path.exists():
                record = byte_record(path.read_bytes(), member)
            else:
                header = fetch(member['local_header_offset'], 30)
                if header[:4] != b'PK\x03\x04':
                    raise ValueError('Local ZIP signature mismatch')
                name_len, extra_len = struct.unpack_from('<2H', header, 26)
                length = name_len+extra_len+member['compressed_bytes']
                if length > 100_000_000:
                    raise ValueError('Unexpected oversized phone record')
                blob = header+fetch(member['local_header_offset']+30, length)
                raw = unpack_member(blob, member['local_header_offset'], member)
                record = byte_record(raw, member)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            records.append({'scene': scene['scene'], 'role': scene['role'], **record})
        print(json.dumps({'scene': scene['scene'], 'role': scene['role'], 'traffic_bytes': used}), flush=True)
    write_json(root/f'verification_{args.role}.json', {'status': 'ALL REQUESTED RECORDS CRC/SIZE/SHA VERIFIED; NOT DECODED',
               'lock_sha256': args.lock_sha256, 'cumulative_payload_traffic_bytes': used, 'records': records})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'download'))
    parser.add_argument('--lock-sha256')
    parser.add_argument('--role', choices=('all', 'loader', 'reserved_test'), default='all')
    parser.add_argument('--out', default=str(ROOT/'data/public/beyond_rgb_phone'))
    options = parser.parse_args()
    if options.command == 'freeze':
        freeze()
    else:
        download(options)
